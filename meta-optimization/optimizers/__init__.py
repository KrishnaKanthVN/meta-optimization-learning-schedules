import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, 
    CosineAnnealingLR, 
    OneCycleLR, 
    CyclicLR
)


def get_optimizer(optimizer_name, model_parameters, **kwargs):
    """Factory function to get the specified optimizer."""
    if optimizer_name == 'sgd':
        return optim.SGD(
            model_parameters, 
            lr=kwargs.get('lr', 0.1),
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
    elif optimizer_name == 'adam':
        return optim.Adam(
            model_parameters,
            lr=kwargs.get('lr', 0.001),
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(
            model_parameters,
            lr=kwargs.get('lr', 0.001),
            alpha=kwargs.get('alpha', 0.99),
            weight_decay=kwargs.get('weight_decay', 5e-4)
        )
    elif optimizer_name == 'adamw':
        return optim.AdamW(
            model_parameters,
            lr=kwargs.get('lr', 0.001),
            betas=kwargs.get('betas', (0.9, 0.999)),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(scheduler_name, optimizer, **kwargs):
    """Factory function to get the specified learning rate scheduler."""
    if scheduler_name == 'constant':
        return None  # No scheduler means constant learning rate
    elif scheduler_name == 'step':
        return StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 200),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.1),
            steps_per_epoch=kwargs.get('steps_per_epoch', 391),  # For CIFAR10 with batch_size=128
            epochs=kwargs.get('epochs', 200),
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25),
            final_div_factor=kwargs.get('final_div_factor', 10000)
        )
    elif scheduler_name == 'cyclical':
        return CyclicLR(
            optimizer,
            base_lr=kwargs.get('base_lr', 0.001),
            max_lr=kwargs.get('max_lr', 0.1),
            step_size_up=kwargs.get('step_size_up', 2000),
            mode=kwargs.get('mode', 'triangular2')
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")