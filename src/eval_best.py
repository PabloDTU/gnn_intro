import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf


def load_config(config_path: Path):
    """Load a Hydra config.yaml from a past run."""
    return OmegaConf.load(config_path)


def build_datamodule(cfg, device):
    dm = hydra.utils.instantiate(cfg.dataset.init)
    return dm


def build_models(cfg, device, use_ncps: bool):
    # Reuse same logic as run.py: one model by default, two if N-CPS
    def _build_model():
        m = hydra.utils.instantiate(cfg.model.init).to(device)
        return m

    if use_ncps:
        model1 = _build_model()
        model2 = _build_model()
        models = [model1, model2]
    else:
        model = _build_model()
        models = [model]
    return models


def load_checkpoints(models, ckpt_dir: Path, use_ncps: bool, device):
    if use_ncps and len(models) >= 2:
        for idx, m in enumerate(models):
            ckpt_path = ckpt_dir / f"model_{idx}_best.pt"
            state = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(state)
    else:
        ckpt_path = ckpt_dir / "model_0_best.pt"
        state = torch.load(ckpt_path, map_location=device)
        models[0].load_state_dict(state)


def evaluate(config_path: Path):
    project_root = Path(__file__).resolve().parents[1]
    ckpt_dir = project_root / "checkpoints"

    cfg = load_config(config_path)

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    use_ncps = getattr(cfg.trainer.init, "use_ncps", False)

    dm = build_datamodule(cfg, device)
    models = build_models(cfg, device, use_ncps)

    load_checkpoints(models, ckpt_dir, use_ncps, device)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    trainer = hydra.utils.instantiate(
        cfg.trainer.init,
        models=models,
        logger=logger,
        datamodule=dm,
        device=device,
    )

    # Run validation metrics
    val_metrics = trainer.validate()
    print("Validation metrics from best checkpoint:", val_metrics)

    # Run test evaluation (simple loop over test_dataloader)
    try:
        for m in models:
            m.eval()
        test_loader = dm.test_dataloader()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, targets in test_loader:
                x, targets = x.to(device), targets.to(device)
                preds = [m(x) for m in models]
                avg_preds = torch.stack(preds).mean(0)
                y_true.append(targets.detach().cpu())
                y_pred.append(avg_preds.detach().cpu())

        if y_true:
            y_true_t = torch.cat(y_true, dim=0)
            y_pred_t = torch.cat(y_pred, dim=0)
            test_mse = torch.nn.functional.mse_loss(y_pred_t, y_true_t).item()
            print("Test MSE from best checkpoint:", test_mse)
        else:
            print("Test evaluation skipped: empty test dataloader.")
    except Exception as e:
        print(f"Test evaluation failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on validation set")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a .hydra/config.yaml from a previous run",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    evaluate(config_path)
