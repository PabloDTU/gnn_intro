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
        scheduler_step_on: str = "epoch",
        # Regularization
        edge_drop_prob: float = 0.0,
        feature_mask_prob: float = 0.0,
        grad_clip_norm: float = 0.0,

        # ---------- PSEUDO-LABELING PARAMETERS ----------
        use_pseudo_labels: bool = True,
        unsup_weight: float = 0.1,
        unsup_rampup_epochs: int = 100,
        variance_threshold: float = 0.01,   # Model disagreement threshold
    ):
        self.device = device
        self.models = models

        # Loss + Optimizer + Scheduler
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.scheduler_step_on = scheduler_step_on

        # Regularization
        self.edge_drop_prob_default = edge_drop_prob
        self.feature_mask_prob_default = feature_mask_prob
        self.grad_clip_norm_default = grad_clip_norm

        # Data
        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Unlabeled loader for pseudo-labeling
        self.unlabeled_dataloader = datamodule.unsupervised_train_dataloader()
        self._unlabeled_iter = None

        # Pseudo-labeling settings
        self.use_pseudo_labels = use_pseudo_labels
        self.unsup_weight = unsup_weight
        self.unsup_rampup_epochs = unsup_rampup_epochs
        self.variance_threshold = variance_threshold

        # Logging
        self.logger = logger

        # Cache target stats
        self.y_stats = None

    # -------------------------------------------------------
    # Helper: fetch next unlabeled batch (cycles endlessly)
    # -------------------------------------------------------
    def _get_unlabeled_batch(self):
        if self.unlabeled_dataloader is None:
            return None
        if self._unlabeled_iter is None:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
        try:
            batch = next(self._unlabeled_iter)
        except StopIteration:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
            batch = next(self._unlabeled_iter)

        # batch could be (x,), (x, dummy) or x
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return batch.to(self.device)

    # -------------------------------------------------------
    # Smooth ramp-up used for unsupervised weight scheduling
    # -------------------------------------------------------
    def _rampup(self, epoch):
        if self.unsup_rampup_epochs <= 0:
            return 1.0
        t = min(epoch / float(self.unsup_rampup_epochs), 1.0)
        # Smooth curve used in semi-supervised literature
        return float(np.exp(-5.0 * (1.0 - t) ** 2))

    # -------------------------------------------------------
    # Validation (unchanged)
    # -------------------------------------------------------
    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            if not hasattr(self, 'y_stats') or self.y_stats is None:
                try:
                    y_mean, y_std = self.datamodule.target_stats
                    self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
                except:
                    self.y_stats = (None, None)

            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                y_mean, y_std = self.y_stats

                # Ensemble prediction (teacher = ensemble average)
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                # Unscale predictions if needed
                if y_mean is not None:
                    avg_preds = avg_preds * y_std + y_mean

                val_losses.append(
                    torch.nn.functional.mse_loss(avg_preds, targets).item()
                )

        return {"val_MSE": float(np.mean(val_losses))}

    # -------------------------------------------------------
    # TRAINING LOOP WITH PSEUDO-LABELING
    # -------------------------------------------------------
    def train(
        self,
        total_epochs,
        validation_interval,
        edge_drop_prob=None,
        feature_mask_prob=None,
        grad_clip_norm=None,
    ):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):

            # Put all student models into training mode
            for model in self.models:
                model.train()

            supervised_losses_logged = []

            # Resolve regularization params
            eff_edge_p = self.edge_drop_prob_default if edge_drop_prob is None else edge_drop_prob
            eff_feat_p = self.feature_mask_prob_default if feature_mask_prob is None else feature_mask_prob
            eff_clip_n = self.grad_clip_norm_default if grad_clip_norm is None else grad_clip_norm

            # -------------------------------------------------------
            # MAIN LABELED TRAINING LOOP
            # -------------------------------------------------------
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Load target stats once
                if not hasattr(self, 'y_stats') or self.y_stats is None:
                    try:
                        y_mean, y_std = self.datamodule.target_stats
                        self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
                    except:
                        self.y_stats = (None, None)

                # Apply augmentations
                if eff_edge_p:
                    x = apply_edge_dropout(x, eff_edge_p)
                if eff_feat_p:
                    x = apply_feature_mask(x, eff_feat_p)

                # Standardize targets if dataset provides stats
                y_mean, y_std = self.y_stats
                if y_mean is not None:
                    targets_std = (targets - y_mean) / y_std
                    supervised_losses = [self.supervised_criterion(m(x), targets_std) for m in self.models]
                else:
                    supervised_losses = [self.supervised_criterion(m(x), targets) for m in self.models]

                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.item() / len(self.models))

                # -------------------------------------------------------
                # UNSUPERVISED LOSS: PSEUDO-LABELING
                # -------------------------------------------------------
                pseudo_loss = torch.tensor(0.0, device=self.device)

                if self.use_pseudo_labels and epoch > self.unsup_rampup_epochs:
                    x_u = self._get_unlabeled_batch()

                    if x_u is not None:
                        # Generate predictions for unlabeled graphs
                        preds_u = [m(x_u) for m in self.models]
                        preds_u = torch.stack(preds_u)      # shape: [K, B, 1]

                        # Compute model disagreement (variance across ensemble)
                        variance = preds_u.var(dim=0)       # [B, 1]

                        # Keep only predictions where ensemble agrees
                        mask = (variance < self.variance_threshold).float()

                        # Compute pseudo-labels (ensemble mean)
                        pseudo_labels = preds_u.mean(0).detach()

                        # Compute unsupervised loss
                        # MSE between student predictions and pseudo-labels
                        student_preds = torch.stack([m(x_u) for m in self.models]).mean(0)

                        pseudo_loss = torch.nn.functional.mse_loss(
                            student_preds * mask,
                            pseudo_labels * mask
                        )

                # Ramp-up unsup coefficient
                ramp = self._rampup(epoch)

                # Total loss = supervised + weighted pseudo loss
                total_loss = supervised_loss + ramp * self.unsup_weight * pseudo_loss

                total_loss.backward()

                # Gradient clipping
                if eff_clip_n:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.optimizer.param_groups[0]["params"]], eff_clip_n
                    )

                self.optimizer.step()

            # -------------------------------------------------------
            # END OF EPOCH: VALIDATION + SCHEDULER + LOGGING
            # -------------------------------------------------------
            supervised_losses_logged = float(np.mean(supervised_losses_logged))

            summary_dict = {"supervised_loss": supervised_losses_logged}

            if epoch % validation_interval == 0:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)

                if self.scheduler_step_on == "val_MSE":
                    self.scheduler.step(val_metrics["val_MSE"])
            else:
                if self.scheduler_step_on == "epoch":
                    self.scheduler.step()

            # Print minimal status for HPC
            try:
                lr = float(np.mean([pg["lr"] for pg in self.optimizer.param_groups]))
            except:
                lr = -1

            if "val_MSE" in summary_dict:
                print(f"Epoch {epoch}/{total_epochs} | SupLoss={summary_dict['supervised_loss']:.6f} | ValMSE={summary_dict['val_MSE']:.6f} | LR={lr:.2e}")
            else:
                print(f"Epoch {epoch}/{total_epochs} | SupLoss={summary_dict['supervised_loss']:.6f} | LR={lr:.2e}")

            summary_dict["lr"] = lr
            self.logger.log_dict(summary_dict, step=epoch)
