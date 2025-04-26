import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional

class OptimizerFactory:
    """
    A factory class for creating optimizers
    """
    
    @staticmethod
    def create(params, config: Dict[str, Any]) -> optim.Optimizer:
        """
        Create an optimizer based on your settings.
        
        Args:
            params: parameters to pass to the optimizer
            config: dictionary containing optimizer configuration

        Returns:
            initialized optimizer
        """
        name = config.get('name', 'Adam')
        params_config = config.get('params', {})
        
        # Standard Optimizer
        if hasattr(optim, name):
            optimizer_class = getattr(optim, name)
            return optimizer_class(params, **params_config)
        
        # Custom Optimizer
        if name == 'SAM':  # Sharpness-Aware Minimization
            return OptimizerFactory.create_sam(params, **params_config)
        elif name == 'Lookahead':
            base_optimizer_config = params_config.pop('base_optimizer', {'name': 'Adam', 'params': {}})
            base_optimizer = OptimizerFactory.create(params, base_optimizer_config)
            return OptimizerFactory.create_lookahead(base_optimizer, **params_config)
        else:
            raise ValueError(f"Optimizer {name} is not available")
    
    @staticmethod
    def create_sam(params, base_optimizer: str = 'Adam', rho: float = 0.05, **kwargs):
        """
        Creates a Sharpness-Aware Minimization (SAM) optimizer.

        Args:
            params: Parameters to pass to the optimizer
            base_optimizer: Name of base optimizer
            rho: SAM rho parameters
            **kwargs: Extra parameters to pass to the base optimizer

        Returns:
            SAM optimizer

        Note:
            A PyTorch implementation of SAM is required. It is shown here as a placeholder.
            A real implementation may require importing external packages.
        """
        try:
            from sam import SAM
        except ImportError:
            raise ImportError("To use the SAM Optimizer, you must install the SAM package.")
        
        base_optimizer_class = getattr(optim, base_optimizer)
        return SAM(params, base_optimizer_class, rho=rho, **kwargs)
    
    @staticmethod
    def create_lookahead(base_optimizer, k: int = 5, alpha: float = 0.5):
        """
        Creates a lookahead optimizer.

        Args:
            base_optimizer: Base optimizer
            k: Update interval
            alpha: Blend factor between old and new parameters

        Returns:
            Lookahead optimizer

        Note:
            Lookahead implementation is required. Shown here as placeholder.
        """
        try:
            from lookahead import Lookahead
        except ImportError:
            raise ImportError("To use the Lookahead optimizer, you need to install the Lookahead package.")
        
        return Lookahead(base_optimizer, k=k, alpha=alpha) 