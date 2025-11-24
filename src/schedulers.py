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


def build_onecycle(
    optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    three_phase: bool = False,
):
    """
    Build a OneCycleLR scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to apply the schedule to.
    max_lr : float
        The maximum learning rate the scheduler will reach.
    total_steps : int
        The total number of steps (epochs * steps_per_epoch or directly epochs).
    pct_start : float, optional
        Percentage of total steps spent increasing the learning rate.
    div_factor : float, optional
        Determines initial learning rate = max_lr / div_factor.
    final_div_factor : float, optional
        Determines minimum learning rate = initial_lr / final_div_factor.
    three_phase : bool, optional
        If True, use 3-phase cycle.

    Returns
    -------
    torch.optim.lr_scheduler.OneCycleLR
    """

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
        three_phase=three_phase,
        anneal_strategy="cos",  # smoother decay
    )

    return scheduler