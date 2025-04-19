import torch
import torch.nn as nn
from typing import Dict, Any

class RainfallPredictionModel(nn.Module):
    """
    降雨予測のためのシンプルなニューラルネットワークモデル
    """
    
    def __init__(self, input_dim: int = 4, hidden_dims: list = [32, 16], output_dim: int = 1, dropout_rate: float = 0.2):
        """
        モデルの初期化
        
        Args:
            input_dim: 入力特徴量の次元
            hidden_dims: 隠れ層の次元のリスト
            output_dim: 出力次元 (バイナリ分類のため1)
            dropout_rate: ドロップアウト率
        """
        super(RainfallPredictionModel, self).__init__()
        
        # レイヤーのリストを構築
        layers = []
        
        # 入力層から最初の隠れ層
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隠れ層
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # 出力層
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())  # バイナリ分類なのでシグモイド関数を使用
        
        # Sequential にレイヤーを結合
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        順伝播
        
        Args:
            x: 入力テンソル
            
        Returns:
            出力テンソル
        """
        return self.model(x)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RainfallPredictionModel':
        """
        設定から直接モデルを構築するメソッド
        
        Args:
            config: モデル設定を含む辞書
            
        Returns:
            構築されたモデルインスタンス
        """
        model_params = config.get('params', {})
        
        return cls(
            input_dim=model_params.get('input_dim', 4),
            hidden_dims=model_params.get('hidden_dims', [32, 16]),
            output_dim=model_params.get('output_dim', 1),
            dropout_rate=model_params.get('dropout_rate', 0.2)
        ) 