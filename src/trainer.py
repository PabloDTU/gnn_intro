from functools import partial

import numpy as np
import torch
from tqdm import tqdm
from regularizers import apply_edge_dropout, apply_feature_mask

class SemiSupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        scheduler_step_on: str = "epoch",  # 'epoch' or 'val_MSE' (plateau)
        # Regularization knobs (accepted here due to Hydra passing all init.* fields)
        edge_drop_prob: float = 0.0,
        feature_mask_prob: float = 0.0,
        grad_clip_norm: float = 0.0,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.scheduler_step_on = scheduler_step_on

        # Store regularization defaults (can be overridden per-train call)
        self.edge_drop_prob_default = edge_drop_prob
        self.feature_mask_prob_default = feature_mask_prob
        self.grad_clip_norm_default = grad_clip_norm

        # Dataloader setup
        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            # Ensure target stats available
            if not hasattr(self, 'y_stats') or self.y_stats is None:
                try:
                    y_mean, y_std = self.datamodule.target_stats  # type: ignore
                    self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
                except Exception:
                    self.y_stats = (None, None)
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                # If model trained on standardized targets, predictions are in std-space.
                # Try to get target stats from datamodule to compute unscaled MSE if available.
                y_mean, y_std = self.y_stats if hasattr(self, 'y_stats') else (None, None)
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                if y_mean is not None and y_std is not None:
                    # Unscale predictions back to original scale for reporting
                    avg_preds_unscaled = avg_preds * y_std + y_mean
                    val_loss = torch.nn.functional.mse_loss(avg_preds_unscaled, targets)
                else:
                    val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(
        self,
        total_epochs,
        validation_interval,
        edge_drop_prob: float | None = None,
        feature_mask_prob: float | None = None,
        grad_clip_norm: float | None = None,
    ):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            # Cache target stats once
            if not hasattr(self, 'y_stats'):
                self.y_stats = getattr(self, 'y_stats', None)
                if hasattr(self, 'logger') and hasattr(self, 'val_dataloader'):
                    pass
            # Resolve regularization values (prefer call-time overrides)
            eff_edge_p = self.edge_drop_prob_default if edge_drop_prob is None else edge_drop_prob
            eff_feat_p = self.feature_mask_prob_default if feature_mask_prob is None else feature_mask_prob
            eff_clip_n = self.grad_clip_norm_default if grad_clip_norm is None else grad_clip_norm

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Ensure target stats are available (compute once)
                if not hasattr(self, 'y_stats') or self.y_stats is None:
                    try:
                        y_mean, y_std = self.datamodule.target_stats  # type: ignore
                        self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
                    except Exception:
                        self.y_stats = (None, None)

                # Apply regularization transforms (train-time only)
                if (eff_edge_p or 0) > 0:
                    x = apply_edge_dropout(x, float(eff_edge_p))
                if (eff_feat_p or 0) > 0:
                    x = apply_feature_mask(x, float(eff_feat_p))

                # Supervised loss (use standardized targets if stats are available)
                y_mean, y_std = self.y_stats if hasattr(self, 'y_stats') else (None, None)
                if y_mean is not None and y_std is not None:
                    targets_std = (targets - y_mean) / y_std
                    supervised_losses = [self.supervised_criterion(model(x), targets_std) for model in self.models]
                else:
                    supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]

                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore

                if eff_clip_n is not None and eff_clip_n > 0:
                    params = []
                    for g in self.optimizer.param_groups:
                        params += list(g.get('params', []))
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, float(eff_clip_n))

                self.optimizer.step()
            # Scheduler step mode
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                # If ReduceLROnPlateau style
                if self.scheduler_step_on == "val_MSE":
                    if hasattr(self.scheduler, "step"):
                        self.scheduler.step(val_metrics.get("val_MSE"))
                pbar.set_postfix(summary_dict)
            else:
                # Epoch-based schedulers
                if self.scheduler_step_on == "epoch":
                    self.scheduler.step()
            # Log current LR
            try:
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                summary_dict["lr"] = np.mean(lrs)
            except Exception:
                pass
            self.logger.log_dict(summary_dict, step=epoch)
