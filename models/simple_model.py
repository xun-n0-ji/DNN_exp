import torch
import torch.nn as nn
from typing import Dict, Any

class SimpleModel(nn.Module):
    """
    シンプルなニューラルネットワークモデル
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 2):
        """
        モデルの初期化
        
        Args:
            input_dim: 入力次元
            hidden_dim: 隠れ層の次元
            output_dim: 出力次元
        """
        super(SimpleModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        """
        順伝播
        
        Args:
            x: 入力テンソル
            
        Returns:
            出力テンソル
        """
        return self.layers(x)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SimpleModel':
        """
        設定から直接モデルを構築するメソッド
        
        Args:
            config: モデル設定を含む辞書
            
        Returns:
            構築されたモデルインスタンス
        """
        model_params = config.get('params', {})
        
        return cls(
            input_dim=model_params.get('input_dim', 10),
            hidden_dim=model_params.get('hidden_dim', 20),
            output_dim=model_params.get('output_dim', 2)
        ) 