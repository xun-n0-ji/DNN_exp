import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional

class SchedulerFactory:
    """
    A factory class for creating learning rate schedulers.
    """
    
    @staticmethod
    def create(optimizer, config: Dict[str, Any]) -> Optional[_LRScheduler]:
        """
        Creates a learning rate scheduler based on the configuration.

        Args:
            optimizer: Optimizer
            config: Dictionary containing scheduler configuration

        Returns:
            Initialized scheduler, or None if no configuration.
        """
        if not config:
            return None
        
        name = config.get('name')
        if not name:
            return None
        
        params = config.get('params', {})
        
        # Standard Scheduler
        if hasattr(optim.lr_scheduler, name):
            scheduler_class = getattr(optim.lr_scheduler, name)
            return scheduler_class(optimizer, **params)
        
        # Custom Scheduler
        if name == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
        elif name == 'GradualWarmupScheduler':
            return SchedulerFactory.create_warmup_scheduler(optimizer, **params)
        elif name == 'CyclicLR':
            return optim.lr_scheduler.CyclicLR(optimizer, **params)
        else:
            raise ValueError(f"Scheduler {name} is unavailable")
    
    @staticmethod
    def create_warmup_scheduler(optimizer, warmup_epochs: int = 5, after_scheduler_config: Dict[str, Any] = None, **kwargs):
        """
        Creates a warmup scheduler.

        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            after_scheduler_config: Scheduler configuration to use after warmup
            **kwargs: Additional parameters

        Returns:
            Warmup scheduler

        Note:
            An implementation of GradualWarmupScheduler is required. It is shown here as a placeholder.
                """
        try:
            from warmup_scheduler import GradualWarmupScheduler
        except ImportError:
            raise ImportError("To use the warmup scheduler, you need to install the warmup_scheduler package.")
        
        after_scheduler = None
        if after_scheduler_config:
            after_scheduler_name = after_scheduler_config.get('name')
            after_scheduler_params = after_scheduler_config.get('params', {})
            
            if hasattr(optim.lr_scheduler, after_scheduler_name):
                scheduler_class = getattr(optim.lr_scheduler, after_scheduler_name)
                after_scheduler = scheduler_class(optimizer, **after_scheduler_params)
            else:
                raise ValueError(f"Scheduler {after_scheduler_name} is unavailable")
        
        return GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_epochs, after_scheduler=after_scheduler)

class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """
    CosineAnnealingWarmRestarts scheduler with warmup
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5, T_0: int = 10, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        """
        Initialization

        Args:
            optimizer: Optimizer
            warmup_epochs: Number of warmup epochs
            T_0: Number of iterations before the first restart
            T_mult: Multiplier of T_i after restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch
        """
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = None
        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Gets the current learning rate.

        Returns:
            List of current learning rates
        """
        if self.base_lrs is None:
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch < self.warmup_epochs:
            # During the warm-up period
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # After warm-up
            epoch = self.last_epoch - self.warmup_epochs
            T_cur = epoch % self.T_0
            return [eta_min + (base_lr - eta_min) * (1 + torch.cos(torch.tensor(T_cur * torch.pi / self.T_0))) / 2
                    for base_lr, eta_min in zip(self.base_lrs, [self.eta_min] * len(self.base_lrs))] 