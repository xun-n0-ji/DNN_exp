import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Tuple, List

from trainer import Trainer

class BaseTrainer(Trainer):
    """
    基本的なトレーナーの実装例。
    他のタスク固有のトレーナーは、この基本トレーナーをベースにすることができます。
    """
    
    def _init_model(self) -> nn.Module:
        """
        モデルを初期化します。
        
        Returns:
            初期化されたモデル
        """
        model_config = self.config.get('model', {})
        model_name = model_config.get('name', 'SimpleModel')
        model_params = model_config.get('params', {})
        
        # モデルの構築（ここでは簡単な例として線形モデルを作成）
        # 実際のタスクでは、適切なモデルを構築する必要があります
        if model_name == 'SimpleModel':
            input_dim = model_params.get('input_dim', 10)
            hidden_dim = model_params.get('hidden_dim', 20)
            output_dim = model_params.get('output_dim', 2)
            
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            
            return model
        else:
            # 他のモデルの場合、適切なモデルクラスをインポートして使用
            raise NotImplementedError(f"モデル {model_name} は実装されていません")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        データローダーを作成します。
        
        Returns:
            学習データローダーと検証データローダーのタプル
        """
        dataset_config = self.config.get('dataset', {})
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        
        # データセットの作成（ここでは簡単な例としてダミーデータを作成）
        # 実際のタスクでは、適切なデータセットを作成する必要があります
        class DummyDataset(Dataset):
            def __init__(self, size=1000, input_dim=10):
                self.size = size
                self.input_dim = input_dim
                self.data = torch.randn(size, input_dim)
                self.targets = torch.randint(0, 2, (size,))
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]
        
        input_dim = self.config.get('model', {}).get('params', {}).get('input_dim', 10)
        train_dataset = DummyDataset(size=1000, input_dim=input_dim)
        val_dataset = DummyDataset(size=200, input_dim=input_dim)
        
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
            targets = targets.to(self.device)
            
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
            
            # 分類タスクの場合の精度計算
            if outputs.shape[1] > 1:  # マルチクラス分類の場合
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
            else:  # 二値分類の場合
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
                targets = targets.to(self.device)
                
                # 順伝播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 統計情報の更新
                total_loss += loss.item() * inputs.size(0)
                
                # 分類タスクの場合の精度計算
                if outputs.shape[1] > 1:  # マルチクラス分類の場合
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                else:  # 二値分類の場合
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