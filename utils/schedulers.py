from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR, CosineAnnealingLR, SequentialLR


def build_scheduler(config, optimizer, num_steps):
  
    warmup_steps = int(config.warmup_epochs * num_steps)

    def warmup_lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # no change after warmup

    if config.scheduler_name == 'cosine':
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs * num_steps,  # Total number of steps
            eta_min=config.min_lr  # Minimum learning rate after annealing
        )
    elif config.scheduler_name == 'setplr':
        base_scheduler = StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.step_gamma
        )
    elif config.scheduler_name == 'multisetplr':
        base_scheduler = MultiStepLR(
            optimizer, 
            milestones=config.msteplr_milestones, 
            gamma=float(config.msteplr_gamma)
        )
    else:
        raise NotImplementedError(f'LR scheduler {config.scheduler_name} not implemented')

    if config.warmup_epochs > 0:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
        return SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, base_scheduler],
            milestones=[warmup_steps]
        )
    else:
        return base_scheduler