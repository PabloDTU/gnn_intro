
from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf
import os

from utils import seed_everything
os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    if cfg.device in ["unset", "auto"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    seed_everything(cfg.seed, cfg.force_deterministic)

    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    dm = hydra.utils.instantiate(cfg.dataset.init)

    model = hydra.utils.instantiate(cfg.model.init).to(device)
    # Sanity: verify that instantiated class matches config target
    try:
        expected_target = cfg.model.init._target_
        actual_target = f"{model.__class__.__module__}.{model.__class__.__name__}"
        print(f"[INFO] Model expected: {expected_target} | actual: {actual_target}")
    except Exception:
        pass

    # Optional debug instrumentation to verify which model runs and data shapes
    if getattr(cfg, "debug_mode", False):
        def _count_params(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[DEBUG] Model class: {model.__class__.__name__} | trainable params: {_count_params(model):,}")
        try:
            train_loader = dm.train_dataloader()
            batch = next(iter(train_loader))
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, targets = batch
            else:
                x, targets = batch, None
            print(f"[DEBUG] Batch type: {type(x)}")
            # Try to introspect PyG Data
            for attr in ["x", "edge_index", "edge_attr", "batch"]:
                if hasattr(x, attr) and getattr(x, attr) is not None:
                    try:
                        shape = tuple(getattr(x, attr).shape)
                    except Exception:
                        shape = "(unknown)"
                    print(f"[DEBUG]   data.{attr} shape: {shape}")
        except Exception as e:
            print(f"[DEBUG] Failed to sample a batch for inspection: {e}")

    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    trainer = hydra.utils.instantiate(
        cfg.trainer.init,
        models=models,
        logger=logger,
        datamodule=dm,
        device=device,
    )

    # Pull optional regularization params from config if present
    edge_p = getattr(cfg.trainer.init, 'edge_drop_prob', 0.0)
    feat_p = getattr(cfg.trainer.init, 'feature_mask_prob', 0.0)
    clip_n = getattr(cfg.trainer.init, 'grad_clip_norm', 0.0)
    results = trainer.train(
        **cfg.trainer.train,
        edge_drop_prob=edge_p,
        feature_mask_prob=feat_p,
        grad_clip_norm=clip_n if clip_n and clip_n > 0 else None,
    )
    #results = torch.Tensor(results)



if __name__ == "__main__":
    main()
