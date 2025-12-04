from functools import partial
from copy import deepcopy
import os

import numpy as np
import torch
from tqdm import tqdm

from regularizers import apply_edge_dropout, apply_feature_mask


class SemiSupervisedEnsemble:
    """
    Supervised ensemble trainer + Optional VAT-style consistency on unlabeled graphs.

    - Supports:
        * multiple student models (ensemble)
        * edge dropout / feature masking regularization
        * LR scheduler (epoch-based or val_MSE-based)
        * optional VAT unsupervised loss on unlabeled data

    We assume the datamodule exposes:
        - train_dataloader()
        - val_dataloader()
        - test_dataloader()
        - optionally unlabeled_dataloader()
        - optionally target_stats = (y_mean, y_std) for de-standardizing targets
    """

    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        method: str | None = None,
        scheduler_step_on: str = "epoch",  # 'epoch' or 'val_MSE' (plateau)

        # ----- Regularization knobs -----
        edge_drop_prob: float = 0.0,
        feature_mask_prob: float = 0.0,
        grad_clip_norm: float = 0.0,

        # ----- VAT semi-supervised knobs -----
        use_vat: bool = False,
        unsup_weight: float = 1.0,
        unsup_rampup_epochs: int = 50,
        vat_start_epoch: int = 0,
        vat_eps: float = 2e-2,
        vat_xi: float = 1e-3,
        unsup_max_ratio: float = 0.2,
        vat_iters: int = 1,
        unsup_every_k_steps: int = 1,

        # ----- Cross Pseudo Supervision (N-CPS) knobs -----
        use_ncps: bool = False,
        ncps_weight: float = 1.0,

        # ----- Mean-teacher knobs -----
        use_mean_teacher: bool = False,
        mean_teacher_weight: float = 1.0,
        mean_teacher_ema_decay: float = 0.99,
    ):
        self.device = device
        self.models = models

        # ----- Optimizer + scheduler -----
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        # `optimizer` is expected to be a partial (Hydra _partial_: true)
        self.optimizer = optimizer(params=all_params)
        # `scheduler` is expected to be a callable taking the optimizer
        self.scheduler = scheduler(optimizer=self.optimizer)
        self.scheduler_step_on = scheduler_step_on

        # ----- Store regularization defaults -----
        self.edge_drop_prob_default = edge_drop_prob
        self.feature_mask_prob_default = feature_mask_prob
        self.grad_clip_norm_default = grad_clip_norm
        self.method = method

        # Resolve checkpoint directory: checkpoints/<method>/ for semi-supervised variants,
        # and checkpoints/ root for supervised.
        project_root = os.path.dirname(os.path.dirname(__file__))

        # Infer a method name from flags if none was provided explicitly.
        inferred_method = None
        if method:
            inferred_method = method
        else:
            if self.use_vat:
                inferred_method = "VAT"
            elif self.use_ncps:
                inferred_method = "NCPS"
            elif self.use_mean_teacher:
                inferred_method = "MT"

        method_dir = inferred_method
        if method_dir and method_dir not in {"supervised"}:
            self.checkpoint_dir = os.path.join(project_root, "checkpoints", method_dir)
        else:
            self.checkpoint_dir = os.path.join(project_root, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # ----- VAT config -----
        self.use_vat = use_vat
        self.unsup_weight = unsup_weight
        self.unsup_rampup_epochs = unsup_rampup_epochs
        self.vat_start_epoch = vat_start_epoch
        self.vat_eps = vat_eps
        self.vat_xi = vat_xi
        self.unsup_max_ratio = unsup_max_ratio
        self.vat_iters = vat_iters
        self.unsup_every_k_steps = unsup_every_k_steps

        # ----- N-CPS config -----
        self.use_ncps = use_ncps
        self.ncps_weight = ncps_weight

        # ----- Mean-teacher config -----
        self.use_mean_teacher = use_mean_teacher
        self.mean_teacher_weight = mean_teacher_weight
        self.mean_teacher_ema_decay = mean_teacher_ema_decay

        # ----- Dataloaders -----
        self.datamodule = datamodule
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Optional unlabeled dataloader for VAT / N-CPL / mean-teacher
        self.unlabeled_dataloader = (
            datamodule.unsupervised_train_dataloader()
            if hasattr(datamodule, "unsupervised_train_dataloader")
            else None
        )
        self._unlabeled_iter = None

        # Logging
        self.logger = logger

        # Cache for target mean/std
        self.y_stats = None

        # Teacher models (for mean-teacher)
        if self.use_mean_teacher:
            self.teacher_models = [deepcopy(m).to(self.device) for m in self.models]
            for tm in self.teacher_models:
                tm.eval()
        else:
            self.teacher_models = None

        # Checkpoint tracking
        self.best_val = float("inf")
        self.best_ckpt_path = os.path.join(self.checkpoint_dir, "best.ckpt")

    # ------------------------------------------------------------------
    # Helper: save checkpoints (full state)
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str, epoch: int | None = None, extra: dict | None = None):
        """
        Save a training checkpoint containing ensemble model weights, optimizer
        and scheduler states, and optional metadata.

        - `path`: destination file (e.g. `outputs/best.ckpt`)
        - `epoch`: optional current epoch number
        - `extra`: optional dict for arbitrary metadata (cfg, best_metric...)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Move y_stats to cpu if present so checkpoint is portable
        y_stats_cpu = (None, None)
        if self.y_stats is not None:
            ym, ys = self.y_stats
            y_stats_cpu = (
                ym.detach().cpu() if (ym is not None and isinstance(ym, torch.Tensor)) else ym,
                ys.detach().cpu() if (ys is not None and isinstance(ys, torch.Tensor)) else ys,
            )

        ckpt = {
            "epoch": epoch,
            "models": [m.state_dict() for m in self.models],
            "optimizer": self.optimizer.state_dict() if hasattr(self.optimizer, "state_dict") else None,
            "scheduler": self.scheduler.state_dict() if hasattr(self.scheduler, "state_dict") else None,
            "y_stats": y_stats_cpu,
            "meta": extra,
        }

        torch.save(ckpt, path)

    def load_checkpoint(self, path: str, map_location: str | None = None, load_optimizer: bool = True, load_scheduler: bool = True):
        """
        Load checkpoint and restore model weights plus optional optimizer/scheduler states.

        - `map_location`: torch.load map_location (defaults to self.device)
        - `load_optimizer`, `load_scheduler`: whether to restore optimizer/scheduler

        Returns the loaded checkpoint dict.
        """
        map_location = map_location or self.device
        ckpt = torch.load(path, map_location=map_location)

        # Load models (assumes same number/order of models)
        model_states = ckpt.get("models", [])
        if len(model_states) != len(self.models):
            # best-effort: load what matches, warn user
            for i, st in enumerate(model_states):
                if i < len(self.models):
                    try:
                        self.models[i].load_state_dict(st)
                    except Exception:
                        pass
        else:
            for m, st in zip(self.models, model_states):
                m.load_state_dict(st)

        # Restore optimizer state if available and requested
        if load_optimizer and ckpt.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                # incompatible optimizer state (e.g., different param groups)
                pass

        # Restore scheduler state if available and requested
        if load_scheduler and ckpt.get("scheduler") is not None and hasattr(self.scheduler, "load_state_dict"):
            try:
                self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception:
                pass

        # Restore y_stats if present
        y_stats = ckpt.get("y_stats")
        if y_stats is not None:
            self.y_stats = y_stats

        return ckpt

    # ------------------------------------------------------------------
    # Helper: get (and cache) target mean/std from datamodule
    # ------------------------------------------------------------------
    def _get_y_stats(self):
        if self.y_stats is None:
            try:
                y_mean, y_std = self.datamodule.target_stats  # type: ignore
                self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
            except Exception:
                self.y_stats = (None, None)
        return self.y_stats

    # ------------------------------------------------------------------
    # Helper: iterator over unlabeled dataloader (for VAT)
    # ------------------------------------------------------------------
    def _get_unlabeled_batch(self):
        """Return a batch from unlabeled_dataloader, cycling if needed."""
        if self.unlabeled_dataloader is None:
            return None
        if self._unlabeled_iter is None:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
        try:
            batch = next(self._unlabeled_iter)
        except StopIteration:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
            batch = next(self._unlabeled_iter)

        # Assume first element is the input graph(s)
        if isinstance(batch, (tuple, list)):
            x_u = batch[0]
        else:
            x_u = batch
        return x_u.to(self.device)

    # ------------------------------------------------------------------
    # Helper: ramp-up for unsupervised loss weight
    # ------------------------------------------------------------------
    def _unsup_rampup(self, epoch: int) -> float:
        """
        Smoothly increase the weight of unsupervised VAT loss at early epochs.
        Typical exponential ramp-up used in semi-supervised papers.
        """
        if self.unsup_rampup_epochs <= 0:
            return 1.0
        t = min(epoch / float(self.unsup_rampup_epochs), 1.0)
        return float(np.exp(-5.0 * (1.0 - t) ** 2))

    # ------------------------------------------------------------------
    # Helper: L2-normalization over batch
    # ------------------------------------------------------------------
    @staticmethod
    def _l2_normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Normalize tensor over the batch dimension to unit L2 norm.

        We flatten all feature dimensions per example (e.g. per node)
        and normalize each row, then reshape back.
        """
        if t is None:
            return t
        # t: [N, F] or [N, ...]. We treat the first dim as "batch-like".
        flat = t.view(t.size(0), -1)
        norm = flat.norm(p=2, dim=1, keepdim=True) + eps
        flat_normed = flat / norm
        return flat_normed.view_as(t)

    # ------------------------------------------------------------------
    # Helper: update teacher models with EMA
    # ------------------------------------------------------------------
    def _update_teacher_models(self):
        if not self.use_mean_teacher or self.teacher_models is None:
            return
        decay = self.mean_teacher_ema_decay
        for student, teacher in zip(self.models, self.teacher_models):
            with torch.no_grad():
                for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                    p_t.data.mul_(decay).add_(p_s.data, alpha=1.0 - decay)

    # ------------------------------------------------------------------
    # N-CPS loss (Cross Pseudo Supervision) on unlabeled batch
    # ------------------------------------------------------------------
    def _compute_ncps_loss(self, preds_list: list[torch.Tensor]) -> torch.Tensor:
        """Cross Pseudo Supervision between models on unlabeled data.

        preds_list: list of [B, 1] tensors, one per model.
        For two models f1, f2 we use:
            ||f1(x_u) - sg(f2(x_u))||^2 + ||f2(x_u) - sg(f1(x_u))||^2.
        For >2 models, we sum CPS over all ordered pairs.
        """
        if not self.use_ncps:
            return torch.tensor(0.0, device=self.device)

        if len(preds_list) < 2:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        n = len(preds_list)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # student i matches teacher j (stop-grad on teacher)
                loss = loss + ((preds_list[i] - preds_list[j].detach()) ** 2).mean()
        # average over number of ordered pairs
        loss = loss / float(n * (n - 1))
        return loss

    # ------------------------------------------------------------------
    # Mean-teacher consistency loss on an unlabeled batch
    # ------------------------------------------------------------------
    def _compute_mean_teacher_loss(self, x_u) -> torch.Tensor:
        """Consistency between student ensemble and EMA teacher on unlabeled data.

        x_u: unlabeled batch from the unlabeled_dataloader.
        """
        if not self.use_mean_teacher or self.teacher_models is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            teacher_preds = [tm(x_u) for tm in self.teacher_models]
            teacher_mean = torch.stack(teacher_preds).mean(0)

        student_preds = [m(x_u) for m in self.models]
        student_mean = torch.stack(student_preds).mean(0)

        return ((student_mean - teacher_mean) ** 2).mean()

    # ------------------------------------------------------------------
    # VAT loss on a single unlabeled batch
    # ------------------------------------------------------------------
    def _compute_vat_loss(self, x_u) -> torch.Tensor:
        """
        VAT computed on node features x_u.x.

        Important:
        - Models run in eval mode during VAT to avoid corrupting BN / Dropout stats.
        - Gradients are taken only w.r.t. r (perturbation), not params.
        """
        if not self.use_vat:
            return torch.tensor(0.0, device=self.device)
        if not hasattr(x_u, "x") or x_u.x is None:
            return torch.tensor(0.0, device=self.device)

        # Save training/eval state
        prev_train_states = [m.training for m in self.models]
        for m in self.models:
            m.eval()

        try:
            with torch.no_grad():
                base_pred = torch.stack([m(x_u) for m in self.models]).mean(0)

            r = torch.randn_like(x_u.x)

            for _ in range(self.vat_iters):
                r = self.vat_xi * self._l2_normalize(r)
                r.requires_grad_()

                pert_data = x_u.clone()
                pert_data.x = x_u.x + r

                pert_pred = torch.stack([m(pert_data) for m in self.models]).mean(0)
                consistency = ((pert_pred - base_pred.detach()) ** 2).mean()

                grad_r = torch.autograd.grad(consistency, r, retain_graph=False, create_graph=False)[0]
                r = grad_r.detach()

            r_adv = self.vat_eps * self._l2_normalize(r)

            pert_data_final = x_u.clone()
            pert_data_final.x = x_u.x + r_adv

            adv_pred = torch.stack([m(pert_data_final) for m in self.models]).mean(0)
            vat_loss = ((adv_pred - base_pred.detach()) ** 2).mean()

        finally:
            # Restore original training/eval states
            for m, st in zip(self.models, prev_train_states):
                m.train(st)

        return vat_loss

    # ------------------------------------------------------------------
    # Validation loop (ensemble averaged prediction)
    # ------------------------------------------------------------------
    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        y_mean, y_std = self._get_y_stats()

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble predictions
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                # If targets are standardized, unscale for MSE reporting
                if y_mean is not None and y_std is not None:
                    avg_preds_unscaled = avg_preds * y_std + y_mean
                    val_loss = torch.nn.functional.mse_loss(avg_preds_unscaled, targets)
                else:
                    val_loss = torch.nn.functional.mse_loss(avg_preds, targets)

                val_losses.append(val_loss.item())

        val_loss = float(np.mean(val_losses))
        return {"val_MSE": val_loss}

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        total_epochs,
        validation_interval,
        edge_drop_prob: float | None = None,
        feature_mask_prob: float | None = None,
        grad_clip_norm: float | None = None,
    ):
        global_step = 0    
    
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            # Put models in training mode
            for model in self.models:
                model.train()

            supervised_losses_logged = []

            # Resolve regularization knobs (override defaults if provided)
            eff_edge_p = self.edge_drop_prob_default if edge_drop_prob is None else edge_drop_prob
            eff_feat_p = self.feature_mask_prob_default if feature_mask_prob is None else feature_mask_prob
            eff_clip_n = self.grad_clip_norm_default if grad_clip_norm is None else grad_clip_norm

            # Ensure target stats are loaded once
            y_mean, y_std = self._get_y_stats()

            # ------------------- Labeled training loop -------------------
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Data augmentation on labeled graphs
                if (eff_edge_p or 0) > 0:
                    x = apply_edge_dropout(x, float(eff_edge_p))
                if (eff_feat_p or 0) > 0:
                    x = apply_feature_mask(x, float(eff_feat_p))

                # Supervised loss (standardized targets if stats are available)
                if y_mean is not None and y_std is not None:
                    targets_std = (targets - y_mean) / y_std
                    sup_losses = [self.supervised_criterion(model(x), targets_std) for model in self.models]
                else:
                    sup_losses = [self.supervised_criterion(model(x), targets) for model in self.models]

                supervised_loss = torch.stack(sup_losses).mean()
                supervised_losses_logged.append(supervised_loss.detach().item())

                # (Optional) CPS on labeled data is not used; supervised loss
                # already enforces correctness on labeled targets.

                # ------------------- VAT unsupervised loss -------------------
                vat_loss = torch.tensor(0.0, device=self.device)
                mt_loss = torch.tensor(0.0, device=self.device)
                ncps_loss_unlabeled = torch.tensor(0.0, device=self.device)

                apply_unsup_this_step = (self.unlabeled_dataloader is not None) and (global_step % self.unsup_every_k_steps == 0)

                if self.unlabeled_dataloader is not None:
                    x_u = self._get_unlabeled_batch()
                    if x_u is not None:
                        # N-CPS loss on unlabeled data between models
                        if self.use_ncps:
                            preds_u_list = [m(x_u) for m in self.models]
                            ncps_loss_unlabeled = self._compute_ncps_loss(preds_u_list)

                        if self.use_vat and apply_unsup_this_step:
                            vat_loss = self._compute_vat_loss(x_u)
                            
                        if self.use_mean_teacher:
                            mt_loss = self._compute_mean_teacher_loss(x_u)
                
                vat_unsup = torch.tensor(0.0, device=self.device)
                
                if self.use_vat and epoch >= self.vat_start_epoch:
                    epoch_from_start = epoch - self.vat_start_epoch
                    ramp = self._unsup_rampup(epoch_from_start)
                    vat_term = ramp * self.unsup_weight * vat_loss
                    vat_unsup = vat_unsup + vat_term
                else:
                    ramp = 0.0
                    vat_term = torch.tensor(0.0, device=self.device)

                if self.unsup_max_ratio > 0 and vat_unsup.item() > 0 and supervised_loss.item() > 0:
                    max_unsup = self.unsup_max_ratio * supervised_loss.item()
                    if vat_unsup.item() > max_unsup:
                        scale = max_unsup / (vat_unsup.item() + 1e-8)
                        vat_unsup = vat_unsup * scale

                mt_ramp = self._unsup_rampup(epoch)  # reuse same schedule for mean-teacher
                total_loss = (
                    supervised_loss
                    + self.ncps_weight * ncps_loss_unlabeled
                    + mt_ramp * self.mean_teacher_weight * mt_loss
                    + vat_unsup
                )
                print(
                    f"Epoch {epoch} | SupLoss: {supervised_loss.item():.4f} | "
                    f"VAT: {vat_unsup.item():.5f} | MT: {mt_loss.item():.5f} | "
                    f"NCPS: {ncps_loss_unlabeled.item():.5f}"
                )

                # Backprop on combined loss
                total_loss.backward()

                # Gradient clipping if enabled
                if eff_clip_n is not None and eff_clip_n > 0:
                    params = []
                    for g in self.optimizer.param_groups:
                        params += list(g.get("params", []))
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, float(eff_clip_n))

                self.optimizer.step()

                # Update mean-teacher models
                self._update_teacher_models()

            # ------------------- End epoch: logging + scheduler -------------------
            supervised_losses_logged = float(np.mean(supervised_losses_logged))
            summary_dict = {"supervised_loss": supervised_losses_logged}
            # Optional: log individual unsupervised components for debugging
            try:
                summary_dict["vat_loss"] = float(vat_unsup.item())
                summary_dict["mt_loss"] = float(mt_loss.item())
                summary_dict["ncps_unlabeled"] = float(ncps_loss_unlabeled.item())
            except Exception:
                pass

            # Validation + scheduler step
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)

                if self.scheduler_step_on == "val_MSE":
                    if hasattr(self.scheduler, "step"):
                        self.scheduler.step(val_metrics.get("val_MSE"))

                # Checkpointing + track best val_MSE
                current_val = val_metrics.get("val_MSE", None)
                if current_val is not None:
                    if current_val < self.best_val:
                        self.best_val = current_val
                        extra = {"best_val_MSE": self.best_val, "method": self.method}
                        self.save_checkpoint(path=self.best_ckpt_path, epoch=epoch, extra=extra)

                # Expose best val to logging
                if self.best_val < float("inf"):
                    summary_dict["best_val_MSE"] = self.best_val
                pbar.set_postfix(summary_dict)
            else:
                if self.scheduler_step_on == "epoch":
                    self.scheduler.step()

            # Log average LR
            try:
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                summary_dict["lr"] = float(np.mean(lrs))
            except Exception:
                pass

            # Simple console print for HPC (no nice tqdm sometimes)
            try:
                lr_print = summary_dict.get("lr", -1.0)
                if "val_MSE" in summary_dict:
                    best_val_str = (
                        f" | BestValMSE={summary_dict['best_val_MSE']:.6f}"
                        if "best_val_MSE" in summary_dict
                        else ""
                    )
                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        f"SupLoss={summary_dict['supervised_loss']:.6f} | "
                        f"ValMSE={summary_dict['val_MSE']:.6f}"  # current
                        f"{best_val_str} | "
                        f"LR={lr_print:.2e}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{total_epochs} | "
                        f"SupLoss={summary_dict['supervised_loss']:.6f} | "
                        f"LR={lr_print:.2e}"
                    )
            except Exception:
                pass

            # Log to WandB or custom logger
            self.logger.log_dict(summary_dict, step=epoch)
