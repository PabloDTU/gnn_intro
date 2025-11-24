import hydra
import torch
from omegaconf import OmegaConf
import os
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import torch
from utils import seed_everything
import matplotlib.pyplot as plt


os.environ["HYDRA_FULL_ERROR"] = "1"
@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)
    val_loader = dm.val_dataloader()
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))
    x, y = batch
    data = x.get_example(0)

    checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
    model = hydra.utils.instantiate(cfg.model.init)
    model.load_state_dict(checkpoint["models"][0])

    y_mean, y_std = checkpoint["y_stats"]

    print("\n[INFO] Running prediction on validation batch...")
    batch = next(iter(val_loader))

    show_prediction(model, data, y_mean, y_std, device="cpu")


def pyg_to_rdkit(data):
    """
    Convert a PyG QM9 graph to an RDKit Mol object.
    Works for visualization (2D drawing).
    """
    atom_list = data.z.tolist()  # atomic numbers
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
            except:
                pass  # some datasets create duplicated edges

    mol = mol.GetMol()

    # Compute 2D coordinates for drawing
    AllChem.Compute2DCoords(mol)
    return mol

def predict_single(model, data, y_mean=None, y_std=None, device="cpu"):
    model.eval()
    data = data.to(device)

    with torch.no_grad():
        pred = model(data)

    # Unscale prediction if stats exist
    if y_mean is not None and y_std is not None:
        pred = pred * y_std + y_mean

    return float(pred.cpu().item())

def show_prediction(model, data, y_mean, y_std, device="cpu"):
    mol = pyg_to_rdkit(data)
    pred = predict_single(model, data, y_mean, y_std, device)
    true_val = float(data.y.item())

    img = Draw.MolToImage(mol, size=(300, 300))

    print(f"Predicted energy: {pred:.4f}")
    print(f"True energy:      {true_val:.4f}")

    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted energy: {pred:.4f}\nTrue energy: {true_val:.4f}")
    plt.show()

if __name__ == "__main__":
    main()
