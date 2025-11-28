import argparse
import importlib
from pathlib import Path

import torch
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


def main():
    parser = argparse.ArgumentParser(description='Evaluate VAT vs non-VAT checkpoints on the QM9 test split')
    parser.add_argument('--ckpt_vat', type=Path, default=Path('checkpoints/best_model_VAT.pt'), help='Path to VAT-trained checkpoint (best_model.pt)')
    parser.add_argument('--ckpt_no_vat', type=Path, default=Path('checkpoints/best_model_noVAT.pt'), help='Path to non-VAT checkpoint (best_model.pt)')
    parser.add_argument('--model_cfg', type=Path, default=Path('configs/model/edge_aware_gcn.yaml'), help='Model config YAML')
    parser.add_argument('--dataset_cfg', type=Path, default=Path('configs/dataset/qm9.yaml'), help='Dataset config YAML')
    parser.add_argument('--device', type=str, default='auto', help='cuda, cpu, or auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    dm = load_datamodule(args.dataset_cfg)
    test_loader = dm.test_dataloader()
    y_mean, y_std = dm.target_stats

    #model_vat, state_vat = load_model(args.ckpt_vat, args.model_cfg, device)
    model_no_vat, state_no_vat = load_model(args.ckpt_no_vat, args.model_cfg, device)

    #vat_mse = evaluate(model_vat, test_loader, y_mean, y_std, device)
    no_vat_mse = evaluate(model_no_vat, test_loader, y_mean, y_std, device)
    #ensemble_mse = evaluate_ensemble([model_vat, model_no_vat], test_loader, y_mean, y_std, device)

    print('=== Test MSE ===')
    #print(f"VAT model      : {vat_mse:.6f} (epoch {state_vat.get('epoch', 'n/a')}, val {state_vat.get('val_MSE', 'n/a')})")
    print(f"Non-VAT model  : {no_vat_mse:.6f} (epoch {state_no_vat.get('epoch', 'n/a')}, val {state_no_vat.get('val_MSE', 'n/a')})")
    #print(f"Ensemble (avg) : {ensemble_mse:.6f}")


if __name__ == '__main__':
    main()