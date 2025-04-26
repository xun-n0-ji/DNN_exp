import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable

class CriterionFactory:
    """
    Factory class for creating loss functions
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        """
        Creates a loss function based on the configuration.

        Args:
            config: A dictionary containing the loss function configuration.

        Returns:
            The initialized loss function.
        """
        name = config.get('name', 'CrossEntropyLoss')
        params = config.get('params', {})
        
        # Standard loss function
        if hasattr(nn, name):
            criterion_class = getattr(nn, name)
            return criterion_class(**params)
        
        # Custom loss functions
        if name == 'FocalLoss':
            return CriterionFactory.create_focal_loss(**params)
        elif name == 'DiceLoss':
            return CriterionFactory.create_dice_loss(**params)
        elif name == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss(**params)
        else:
            raise ValueError(f"Loss function {name} is not available")
    
    @staticmethod
    def create_focal_loss(gamma: float = 2.0, alpha: Optional[float] = None, 
                          reduction: str = 'mean') -> Callable:
        """
        Creates a Focal Loss.

        Args:
            gamma: Focal parameter
            alpha: Balance parameter
            reduction: Reduction method

        Returns:
            Focal Loss function
        """
        def focal_loss(inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            loss = (1 - pt) ** gamma * ce_loss
            
            if alpha is not None:
                alpha_tensor = torch.tensor([1.0 - alpha, alpha], device=inputs.device)
                alpha_weight = alpha_tensor.gather(0, targets)
                loss = alpha_weight * loss
            
            if reduction == 'mean':
                return loss.mean()
            elif reduction == 'sum':
                return loss.sum()
            else:
                return loss
        
        return focal_loss
    
    @staticmethod
    def create_dice_loss(smooth: float = 1.0, reduction: str = 'mean') -> Callable:
        """
        Creates a Dice Loss.

        Args:
            smooth: Smoothing parameters
            reduction: Reduction method

        Returns:
            Dice Loss function
        """
        def dice_loss(inputs, targets):
            # Sigmoid applied (if input is logit)
            inputs = torch.sigmoid(inputs)
            
            # Smoothing
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()
            dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
            
            return 1.0 - dice
        
        return dice_loss 