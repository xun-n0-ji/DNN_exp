import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Callable

class CriterionFactory:
    """
    損失関数を作成するファクトリクラス
    """
    
    @staticmethod
    def create(config: Dict[str, Any]) -> nn.Module:
        """
        設定に基づいて損失関数を作成します。
        
        Args:
            config: 損失関数の設定を含む辞書
            
        Returns:
            初期化された損失関数
        """
        name = config.get('name', 'CrossEntropyLoss')
        params = config.get('params', {})
        
        # 標準の損失関数
        if hasattr(nn, name):
            criterion_class = getattr(nn, name)
            return criterion_class(**params)
        
        # カスタム損失関数
        if name == 'FocalLoss':
            return CriterionFactory.create_focal_loss(**params)
        elif name == 'DiceLoss':
            return CriterionFactory.create_dice_loss(**params)
        elif name == 'BCEWithLogitsLoss':
            return nn.BCEWithLogitsLoss(**params)
        else:
            raise ValueError(f"損失関数 {name} は利用できません")
    
    @staticmethod
    def create_focal_loss(gamma: float = 2.0, alpha: Optional[float] = None, 
                          reduction: str = 'mean') -> Callable:
        """
        Focal Lossを作成します。
        
        Args:
            gamma: 焦点パラメータ
            alpha: バランスパラメータ
            reduction: 削減方法
            
        Returns:
            Focal Loss関数
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
        Dice Lossを作成します。
        
        Args:
            smooth: スムージングパラメータ
            reduction: 削減方法
            
        Returns:
            Dice Loss関数
        """
        def dice_loss(inputs, targets):
            # シグモイド適用（入力がロジットの場合）
            inputs = torch.sigmoid(inputs)
            
            # 平滑化
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()
            dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
            
            return 1.0 - dice
        
        return dice_loss 