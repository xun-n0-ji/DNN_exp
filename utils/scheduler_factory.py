import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Any, Optional

class SchedulerFactory:
    """
    学習率スケジューラを作成するファクトリクラス
    """
    
    @staticmethod
    def create(optimizer, config: Dict[str, Any]) -> Optional[_LRScheduler]:
        """
        設定に基づいて学習率スケジューラを作成します。
        
        Args:
            optimizer: オプティマイザ
            config: スケジューラの設定を含む辞書
            
        Returns:
            初期化されたスケジューラ、または設定がない場合はNone
        """
        if not config:
            return None
        
        name = config.get('name')
        if not name:
            return None
        
        params = config.get('params', {})
        
        # 標準のスケジューラ
        if hasattr(optim.lr_scheduler, name):
            scheduler_class = getattr(optim.lr_scheduler, name)
            return scheduler_class(optimizer, **params)
        
        # カスタムスケジューラ
        if name == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
        elif name == 'GradualWarmupScheduler':
            return SchedulerFactory.create_warmup_scheduler(optimizer, **params)
        elif name == 'CyclicLR':
            return optim.lr_scheduler.CyclicLR(optimizer, **params)
        else:
            raise ValueError(f"スケジューラ {name} は利用できません")
    
    @staticmethod
    def create_warmup_scheduler(optimizer, warmup_epochs: int = 5, after_scheduler_config: Dict[str, Any] = None, **kwargs):
        """
        Warmupスケジューラを作成します。
        
        Args:
            optimizer: オプティマイザ
            warmup_epochs: ウォームアップのエポック数
            after_scheduler_config: ウォームアップ後に使用するスケジューラの設定
            **kwargs: 追加のパラメータ
            
        Returns:
            Warmupスケジューラ
            
        Note: 
            GradualWarmupSchedulerの実装が必要です。ここではプレースホルダとして示しています。
        """
        try:
            from warmup_scheduler import GradualWarmupScheduler
        except ImportError:
            raise ImportError("Warmupスケジューラを使用するには、warmup_schedulerパッケージのインストールが必要です。")
        
        after_scheduler = None
        if after_scheduler_config:
            after_scheduler_name = after_scheduler_config.get('name')
            after_scheduler_params = after_scheduler_config.get('params', {})
            
            if hasattr(optim.lr_scheduler, after_scheduler_name):
                scheduler_class = getattr(optim.lr_scheduler, after_scheduler_name)
                after_scheduler = scheduler_class(optimizer, **after_scheduler_params)
            else:
                raise ValueError(f"スケジューラ {after_scheduler_name} は利用できません")
        
        return GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=warmup_epochs, after_scheduler=after_scheduler)

class CosineAnnealingWarmRestartsWithWarmup(_LRScheduler):
    """
    ウォームアップ付きのCosineAnnealingWarmRestartsスケジューラ
    """
    
    def __init__(self, optimizer, warmup_epochs: int = 5, T_0: int = 10, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1):
        """
        初期化
        
        Args:
            optimizer: オプティマイザ
            warmup_epochs: ウォームアップのエポック数
            T_0: 最初のリスタートまでの反復回数
            T_mult: リスタート後のT_iの乗数
            eta_min: 最小学習率
            last_epoch: 最後のエポック
        """
        self.warmup_epochs = warmup_epochs
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = None
        super(CosineAnnealingWarmRestartsWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        現在の学習率を取得します。
        
        Returns:
            現在の学習率のリスト
        """
        if self.base_lrs is None:
            self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        
        if self.last_epoch < self.warmup_epochs:
            # ウォームアップ期間中
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # ウォームアップ後
            epoch = self.last_epoch - self.warmup_epochs
            T_cur = epoch % self.T_0
            return [eta_min + (base_lr - eta_min) * (1 + torch.cos(torch.tensor(T_cur * torch.pi / self.T_0))) / 2
                    for base_lr, eta_min in zip(self.base_lrs, [self.eta_min] * len(self.base_lrs))] 