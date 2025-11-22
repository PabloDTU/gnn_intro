import torch


def build_warmup_cosine(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    warmup_start_factor: float = 0.1,
    eta_min: float = 1e-6,
):
    """
    Linear warmup followed by cosine annealing.

    - warmup_epochs: number of epochs to linearly scale LR from
      warmup_start_factor * base_lr to base_lr
    - total_epochs: total number of training epochs
    - eta_min: minimum LR in cosine phase
    """
    warmup_epochs = max(0, int(warmup_epochs))
    t_max = max(1, int(total_epochs) - warmup_epochs)

    schedulers = []
    milestones = []

    if warmup_epochs > 0:
        schedulers.append(
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                total_iters=warmup_epochs,
            )
        )
        milestones.append(warmup_epochs)

    schedulers.append(
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    )

    if len(milestones) == 0:
        # No warmup: just return cosine scheduler
        return schedulers[0]

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=milestones
    )


def build_reduce_on_plateau(
    optimizer: torch.optim.Optimizer,
    mode: str = "min",
    factor: float = 0.5,
    patience: int = 10,
    min_lr: float = 1e-6,
    threshold: float = 1e-4,
):
    """
    ReduceLROnPlateau factory configured for validation MSE minimization.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        min_lr=min_lr,
        threshold=threshold,
        verbose=True,
    )