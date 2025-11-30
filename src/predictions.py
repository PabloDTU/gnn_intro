import argparse
import importlib
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from omegaconf import OmegaConf

from qm9 import QM9DataModule


def resolve_target(target_str: str):
    """Dynamically import a class from a string path."""
    module_name, cls_name = target_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, cls_name)


def _clean_cfg(cfg_dict: dict) -> dict:
    """Drop Hydra helper keys like _target_/_partial_."""
    return {k: v for k, v in cfg_dict.items() if not k.startswith('_')}


def load_model(ckpt_path: Path, model_cfg_path: Path, device: torch.device):
    cfg = OmegaConf.load(model_cfg_path)
    model_cfg = OmegaConf.to_container(cfg.init, resolve=True)
    target = model_cfg.pop('_target_', 'models.EdgeAwareGCNPlus')
    model_cfg = _clean_cfg(model_cfg)
    ModelCls = resolve_target(target)

    model = ModelCls(**model_cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()
    return model, state


def evaluate(model, loader, y_mean, y_std, device):
    mse = torch.nn.MSELoss(reduction='mean')
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, targets = batch
            else:
                # Unexpected batch structure
                continue
            x = x.to(device)
            targets = targets.to(device)

            preds = model(x)
            if y_mean is not None and y_std is not None:
                preds = preds * y_std.to(device) + y_mean.to(device)
            loss = mse(preds, targets)
            total_loss += loss.item() * targets.size(0)
            n += targets.size(0)
    return total_loss / n if n > 0 else float('nan')


def evaluate_ensemble(models, loader, y_mean, y_std, device):
    mse = torch.nn.MSELoss(reduction='mean')
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, targets = batch
            else:
                continue
            x = x.to(device)
            targets = targets.to(device)

            preds = [m(x) for m in models]
            avg_pred = torch.stack(preds).mean(0)
            if y_mean is not None and y_std is not None:
                avg_pred = avg_pred * y_std.to(device) + y_mean.to(device)
            loss = mse(avg_pred, targets)
            total_loss += loss.item() * targets.size(0)
            n += targets.size(0)
    return total_loss / n if n > 0 else float('nan')


def load_datamodule(dataset_cfg_path: Path):
    cfg = OmegaConf.load(dataset_cfg_path)
    dm_kwargs = OmegaConf.to_container(cfg.init, resolve=True)
    dm_kwargs = _clean_cfg(dm_kwargs)
    dm = QM9DataModule(**dm_kwargs)
    return dm


def pyg_to_rdkit(data):
    """Convert a PyG QM9 graph to an RDKit Mol object for visualization."""
    if not hasattr(data, "z") or data.z is None or not hasattr(data, "edge_index"):
        return None

    atom_list = data.z.tolist()
    edge_index = data.edge_index
    mol = Chem.RWMol()

    # Add atoms
    for atomic_num in atom_list:
        mol.AddAtom(Chem.Atom(int(atomic_num)))

    # Add bonds (undirected, so only add i<j)
    for i, j in edge_index.t().tolist():
        if i < j:
            try:
                mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
            except Exception:
                pass  # duplicates can happen

    mol = mol.GetMol()
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass
    return mol


def predict_single(model, data, y_mean=None, y_std=None, device="cpu"):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        pred = model(data)

    if y_mean is not None and y_std is not None:
        pred = pred * y_std.to(device) + y_mean.to(device)

    return float(pred.view(-1)[0].item())


def plot_molecules(model, batch, y_mean, y_std, device, title: str, n: int = 8, save_path: str | None = None):
    """Plot up to n molecules from a (Batch, targets) pair with predicted vs true values."""
    if not (isinstance(batch, (list, tuple)) and len(batch) == 2):
        print(f"[WARN] Unexpected batch structure for plotting: {type(batch)}")
        return

    x, targets = batch
    x_cpu = x.cpu()
    targets_cpu = targets.cpu()

    mols = []
    legends = []
    max_n = min(n, x_cpu.num_graphs)

    for i in range(max_n):
        g = x_cpu.get_example(i)
        try:
            g.y = targets_cpu[i].view(-1)
        except Exception:
            pass

        mol = pyg_to_rdkit(g)
        if mol is None:
            continue

        pred_val = predict_single(model, g, y_mean, y_std, device)
        true_val = float(g.y.view(-1)[0].item()) if hasattr(g, "y") and g.y is not None else None

        legends.append(f"P:{pred_val:.3f} | T:{true_val:.3f}" if true_val is not None else f"P:{pred_val:.3f}")
        mols.append(mol)

    if not mols:
        print(f"[WARN] No molecules to plot for {title}.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(4, len(mols)),
        legends=legends,
        subImgSize=(250, 250),
        useSVG=False,
    )

    plt.figure(figsize=(8, 2 + max_n // 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAT vs non-VAT checkpoints on the QM9 test split')
    parser.add_argument('--ckpt_vat', type=Path, default=Path('checkpoints/best_model_VAT_v2.pt'), help='Path to VAT-trained checkpoint (best_model.pt)')
    parser.add_argument('--ckpt_no_vat', type=Path, default=Path('checkpoints/best_model_noVAT.pt'), help='Path to non-VAT checkpoint (best_model.pt)')
    parser.add_argument('--model_cfg', type=Path, default=Path('configs/model/edge_aware_gcn.yaml'), help='Model config YAML')
    parser.add_argument('--dataset_cfg', type=Path, default=Path('configs/dataset/qm9.yaml'), help='Dataset config YAML')
    parser.add_argument('--device', type=str, default='auto', help='cuda, cpu, or auto')
    parser.add_argument('--num_mols', type=int, default=8, help='Number of molecules to visualize per model')
    parser.add_argument('--save_plots', action='store_true', help='Save the molecule grids as PNGs')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dm = load_datamodule(args.dataset_cfg)
    test_loader = dm.test_dataloader()
    y_mean, y_std = dm.target_stats

    model_vat, state_vat = load_model(args.ckpt_vat, args.model_cfg, device)
    model_no_vat, state_no_vat = load_model(args.ckpt_no_vat, args.model_cfg, device)

    vat_mse = evaluate(model_vat, test_loader, y_mean, y_std, device)
    no_vat_mse = evaluate(model_no_vat, test_loader, y_mean, y_std, device)
    ensemble_mse = evaluate_ensemble([model_vat, model_no_vat], test_loader, y_mean, y_std, device)

    print('=== Test MSE ===')
    print(f"VAT model      : {vat_mse:.6f} (epoch {state_vat.get('epoch', 'n/a')}, val {state_vat.get('val_MSE', 'n/a')})")
    print(f"Non-VAT model  : {no_vat_mse:.6f} (epoch {state_no_vat.get('epoch', 'n/a')}, val {state_no_vat.get('val_MSE', 'n/a')})")
    print(f"Ensemble (avg) : {ensemble_mse:.6f}")

    try:
        batch_for_viz = next(iter(test_loader))
        save_vat = "vat_mols.png" if args.save_plots else None
        save_no_vat = "no_vat_mols.png" if args.save_plots else None
        plot_molecules(model_vat, batch_for_viz, y_mean, y_std, device, title="VAT model", n=args.num_mols, save_path=save_vat)
        plot_molecules(model_no_vat, batch_for_viz, y_mean, y_std, device, title="Non-VAT model", n=args.num_mols, save_path=save_no_vat)
    except Exception as e:
        print(f"[WARN] Failed to generate molecule plots: {e}")


if __name__ == '__main__':
    main()
