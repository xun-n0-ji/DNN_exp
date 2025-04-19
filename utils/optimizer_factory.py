import torch
import torch.optim as optim
from typing import Dict, Any, List, Optional

class OptimizerFactory:
    """
    オプティマイザを作成するファクトリクラス
    """
    
    @staticmethod
    def create(params, config: Dict[str, Any]) -> optim.Optimizer:
        """
        設定に基づいてオプティマイザを作成します。
        
        Args:
            params: オプティマイザに渡すパラメータ
            config: オプティマイザの設定を含む辞書
            
        Returns:
            初期化されたオプティマイザ
        """
        name = config.get('name', 'Adam')
        params_config = config.get('params', {})
        
        # 標準のオプティマイザ
        if hasattr(optim, name):
            optimizer_class = getattr(optim, name)
            return optimizer_class(params, **params_config)
        
        # カスタムオプティマイザ
        if name == 'SAM':  # Sharpness-Aware Minimization
            return OptimizerFactory.create_sam(params, **params_config)
        elif name == 'Lookahead':
            base_optimizer_config = params_config.pop('base_optimizer', {'name': 'Adam', 'params': {}})
            base_optimizer = OptimizerFactory.create(params, base_optimizer_config)
            return OptimizerFactory.create_lookahead(base_optimizer, **params_config)
        else:
            raise ValueError(f"オプティマイザ {name} は利用できません")
    
    @staticmethod
    def create_sam(params, base_optimizer: str = 'Adam', rho: float = 0.05, **kwargs):
        """
        Sharpness-Aware Minimization (SAM) オプティマイザを作成します。
        
        Args:
            params: オプティマイザに渡すパラメータ
            base_optimizer: ベースとなるオプティマイザの名前
            rho: SAMのrhoパラメータ
            **kwargs: ベースオプティマイザに渡す追加のパラメータ
            
        Returns:
            SAMオプティマイザ
            
        Note: 
            SAMのPyTorch実装が必要です。ここではプレースホルダとして示しています。
            実際の実装には外部パッケージのインポートが必要かもしれません。
        """
        try:
            from sam import SAM
        except ImportError:
            raise ImportError("SAMオプティマイザを使用するには、SAMパッケージのインストールが必要です。")
        
        base_optimizer_class = getattr(optim, base_optimizer)
        return SAM(params, base_optimizer_class, rho=rho, **kwargs)
    
    @staticmethod
    def create_lookahead(base_optimizer, k: int = 5, alpha: float = 0.5):
        """
        Lookaheadオプティマイザを作成します。
        
        Args:
            base_optimizer: ベースとなるオプティマイザ
            k: 更新間隔
            alpha: 古いパラメータと新しいパラメータの混合係数
            
        Returns:
            Lookaheadオプティマイザ
            
        Note: 
            Lookaheadの実装が必要です。ここではプレースホルダとして示しています。
        """
        try:
            from lookahead import Lookahead
        except ImportError:
            raise ImportError("Lookaheadオプティマイザを使用するには、Lookaheadパッケージのインストールが必要です。")
        
        return Lookahead(base_optimizer, k=k, alpha=alpha) 