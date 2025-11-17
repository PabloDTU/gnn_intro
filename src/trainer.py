from copy import deepcopy
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

         # ---- Mean Teacher knobs ----
        use_mean_teacher: bool = True,
        ema_decay: float = 0.99,
        unsup_weight: float = 1.0,
        unsup_rampup_epochs: int = 5,
    ):
        self.device = device
        self.models = models # student models

        # --- Mean Teacher setup ---
        self.use_mean_teacher = use_mean_teacher
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight
        self.unsup_rampup_epochs = unsup_rampup_epochs

        if self.use_mean_teacher:
            # One teacher per student model, same architecture, no grads
            self.teacher_models = []
            for m in self.models:
                tm = deepcopy(m).to(self.device)
                for p in tm.parameters():
                    p.requires_grad_(False)
                self.teacher_models.append(tm)
        else:
            self.teacher_models = None

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

        # Unlabeled loader for Mean Teacher
        self.unlabeled_dataloader = (
            datamodule.unlabeled_dataloader()
            if hasattr(datamodule, "unlabeled_dataloader")
            else None
        )
        self._unlabeled_iter = None
        
        # Logging
        self.logger = logger

        # Target stats cache
        self.y_stats = None
    

    # -------------- helper methods --------------
    # Get (and cache) target stats from datamodule
    def _get_y_stats(self):
        if self.y_stats is None:
            try:
                y_mean, y_std = self.datamodule.target_stats  # type: ignore
                self.y_stats = (y_mean.to(self.device), y_std.to(self.device))
            except Exception:
                self.y_stats = (None, None)
        return self.y_stats
    
    # Get a batch from the unlabeled dataloader (for Mean Teacher)
    def _get_unlabeled_batch(self):
        """Cycle through unlabeled dataloader; return None if not available."""
        if self.unlabeled_dataloader is None:
            return None
        if self._unlabeled_iter is None:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
        try:
            batch = next(self._unlabeled_iter)
        except StopIteration:
            self._unlabeled_iter = iter(self.unlabeled_dataloader)
            batch = next(self._unlabeled_iter)
        # assume first element is x_u; ignore any dummy target
        if isinstance(batch, (list, tuple)):
            x_u = batch[0]
        else:
            x_u = batch
        return x_u.to(self.device)
    
    # Ramp-up function for unsupervised loss weight
    def _mean_teacher_rampup(self, epoch: int) -> float:
        if self.unsup_rampup_epochs <= 0:
            return 1.0
        t = min(epoch / float(self.unsup_rampup_epochs), 1.0)
        # smooth ramp-up (same form as in many semi-supervised papers)
        return float(np.exp(-5.0 * (1.0 - t) ** 2))
    
    # Update teacher model parameters using EMA of student parameters
    def _update_teacher(self):
        if not self.use_mean_teacher:
            return
        with torch.no_grad():
            for t_model, s_model in zip(self.teacher_models, self.models):
                for t_param, s_param in zip(t_model.parameters(), s_model.parameters()):
                    t_param.data.mul_(self.ema_decay).add_(
                        s_param.data, alpha=1.0 - self.ema_decay
                    )

    # -------------- validation --------------
    def validate(self):
        # For Mean Teacher, it is common to evaluate the teacher.
        eval_models = self.teacher_models if self.use_mean_teacher else self.models
        for model in eval_models:
            model.eval()

        val_losses = []
        y_mean, y_std = self._get_y_stats()
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                # Ensemble predictions
                preds = [model(x) for model in eval_models]
                avg_preds = torch.stack(preds).mean(0)
                # Unscale targets and predictions if stats are available
                if y_mean is not None and y_std is not None:
                    avg_preds_unscaled = avg_preds * y_std + y_mean
                    val_loss = torch.nn.functional.mse_loss(avg_preds_unscaled, targets)
                else:
                    val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                # Accumulate
                val_losses.append(val_loss.item())
        # Finalize validation metrics
        val_loss = float(np.mean(val_losses))
        return {"val_MSE": val_loss}

    def train(
        self,
        total_epochs,
        validation_interval,
        edge_drop_prob: float | None = None,
        feature_mask_prob: float | None = None,
        grad_clip_norm: float | None = None,
    ):
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            if self.teacher_models is not None:
                for tm in self.teacher_models:
                    tm.eval()  # teacher is not trained by backprop

            supervised_losses_logged = []

            eff_edge_p = self.edge_drop_prob_default if edge_drop_prob is None else edge_drop_prob
            eff_feat_p = self.feature_mask_prob_default if feature_mask_prob is None else feature_mask_prob
            eff_clip_n = self.grad_clip_norm_default if grad_clip_norm is None else grad_clip_norm

            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # augment labelled batch (student path)
                x_lab = x
                if (eff_edge_p or 0) > 0:
                    x_lab = apply_edge_dropout(x_lab, float(eff_edge_p))
                if (eff_feat_p or 0) > 0:
                    x_lab = apply_feature_mask(x_lab, float(eff_feat_p))

                # ---- supervised loss (same as before, possibly with standardised targets) ----
                supervised_losses = [
                    self.supervised_criterion(model(x_lab), targets)
                    for model in self.models
                ]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)
                )

                # ---- unsupervised Mean Teacher loss on unlabeled data ----
                unsup_loss = torch.tensor(0.0, device=self.device)
                if self.use_mean_teacher and self.unlabeled_dataloader is not None:
                    x_u = self._get_unlabeled_batch()
                    if x_u is not None:
                        # two noisy views: one for student, one for teacher
                        x_u_student = x_u
                        x_u_teacher = x_u

                        if (eff_edge_p or 0) > 0:
                            x_u_student = apply_edge_dropout(x_u_student, float(eff_edge_p))
                        if (eff_feat_p or 0) > 0:
                            x_u_student = apply_feature_mask(x_u_student, float(eff_feat_p))

                        student_preds = [m(x_u_student) for m in self.models]
                        with torch.no_grad():
                            teacher_preds = [tm(x_u_teacher) for tm in self.teacher_models]

                        # consistency loss per model, summed
                        for sp, tp in zip(student_preds, teacher_preds):
                            unsup_loss = unsup_loss + torch.mean((sp - tp.detach()) ** 2)

                # ramped unsupervised weight
                ramp = self._mean_teacher_rampup(epoch)
                total_loss = supervised_loss + ramp * self.unsup_weight * unsup_loss

                total_loss.backward()

                if eff_clip_n is not None and eff_clip_n > 0:
                    params = []
                    for g in self.optimizer.param_groups:
                        params += list(g.get("params", []))
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, float(eff_clip_n))

                self.optimizer.step()
                # EMA update for teacher after student update
                self._update_teacher()

            supervised_losses_logged = float(np.mean(supervised_losses_logged))

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
            }

            # validation + scheduler logic 
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                if self.scheduler_step_on == "val_MSE":
                    if hasattr(self.scheduler, "step"):
                        self.scheduler.step(val_metrics.get("val_MSE"))
                pbar.set_postfix(summary_dict)
            else:
                if self.scheduler_step_on == "epoch":
                    self.scheduler.step()

            # log LR
            try:
                lrs = [pg["lr"] for pg in self.optimizer.param_groups]
                summary_dict["lr"] = float(np.mean(lrs))
            except Exception:
                pass

            self.logger.log_dict(summary_dict, step=epoch)