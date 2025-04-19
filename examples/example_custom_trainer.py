import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, Tuple

from trainer import Trainer

class CustomDataset(Dataset):
    """
    カスタムデータセットの例
    """
    def __init__(self, size: int = 1000, input_dim: int = 10, noise: float = 0.1):
        """
        初期化
        
        Args:
            size: データセットのサイズ
            input_dim: 入力次元
            noise: ノイズの標準偏差
        """
        self.size = size
        self.input_dim = input_dim
        
        # ランダムな重みを生成
        self.weights = torch.randn(input_dim)
        
        # 入力特徴量を生成
        self.data = torch.randn(size, input_dim)
        
        # 線形モデル + ノイズでターゲットを生成
        targets = torch.matmul(self.data, self.weights)
        targets += torch.randn(size) * noise
        
        # 二値分類問題に変換
        self.targets = (targets > 0).float()
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class CustomTrainer(Trainer):
    """
    カスタムトレーナーの例
    """
    def _init_model(self) -> nn.Module:
        """
        モデルを初期化します。
        
        Returns:
            初期化されたモデル
        """
        model_config = self.config.get('model', {})
        model_params = model_config.get('params', {})
        
        input_dim = model_params.get('input_dim', 10)
        hidden_dim = model_params.get('hidden_dim', 20)
        
        # 二値分類のための単純な線形モデル
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        return model
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        データローダーを作成します。
        
        Returns:
            学習データローダーと検証データローダーのタプル
        """
        dataset_config = self.config.get('dataset', {})
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        input_dim = self.config.get('model', {}).get('params', {}).get('input_dim', 10)
        
        # カスタムデータセットの作成
        train_dataset = CustomDataset(size=1000, input_dim=input_dim)
        val_dataset = CustomDataset(size=200, input_dim=input_dim)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """
        1エポックの学習を行います。
        
        Args:
            dataloader: 学習データのデータローダー
            
        Returns:
            エポックの学習結果を含む辞書
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.unsqueeze(1).to(self.device)  # (B, 1)の形状にする
            
            # 勾配をゼロにリセット
            self.optimizer.zero_grad()
            
            # 順伝播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 逆伝播と最適化
            loss.backward()
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        # エポックの平均損失と精度を計算
        epoch_loss = total_loss / total
        epoch_acc = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
        
        return metrics
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        検証を行います。
        
        Args:
            dataloader: 検証データのデータローダー
            
        Returns:
            検証結果を含む辞書
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.unsqueeze(1).to(self.device)  # (B, 1)の形状にする
                
                # 順伝播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 統計情報の更新
                total_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
        
        # 平均損失と精度を計算
        avg_loss = total_loss / total
        avg_acc = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
        
        return metrics

# 使用例
if __name__ == "__main__":
    # 設定の作成
    config = {
        'model': {
            'params': {
                'input_dim': 10,
                'hidden_dim': 32
            }
        },
        'dataset': {
            'batch_size': 64,
            'num_workers': 2
        },
        'criterion': {
            'name': 'BCELoss',
            'params': {}
        },
        'optimizer': {
            'name': 'Adam',
            'params': {
                'lr': 0.01
            }
        },
        'training': {
            'epochs': 10,
            'eval_interval': 1
        }
    }
    
    # 実験ディレクトリの作成
    exp_dir = "experiments/custom_task/custom_model/exp001"
    os.makedirs(exp_dir, exist_ok=True)
    
    # トレーナーのインスタンス化
    trainer = CustomTrainer(config, exp_dir)
    
    # データローダーの作成
    train_loader, val_loader = trainer.create_dataloaders()
    
    # 学習の実行
    trainer.train(
        train_loader, 
        val_loader, 
        config.get('training', {}).get('epochs', 10),
        config.get('training', {}).get('eval_interval', 1)
    ) 