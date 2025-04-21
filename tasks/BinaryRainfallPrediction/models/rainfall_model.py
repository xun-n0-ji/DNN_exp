import torch
import torch.nn as nn
from typing import Dict, Any

class RainfallModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(RainfallModel, self).__init__()
        
        layers = []
        
        # 入力層から最初の隠れ層
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # 隠れ層
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        # 出力層
        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        layers.append(torch.nn.Sigmoid())
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RainfallModel':
        """
        設定から直接モデルを構築するメソッド
        
        Args:
            config: モデル設定を含む辞書
            
        Returns:
            構築されたモデルインスタンス
        """

        model_params = config['model']['params']
        
        return cls(
            input_dim=model_params.get('input_dim', 4),
            hidden_dims=model_params.get('hidden_dims', [32, 16]),
            output_dim=model_params.get('output_dim', 1),
            dropout_rate=model_params.get('dropout_rate', 0.2)
        ) 