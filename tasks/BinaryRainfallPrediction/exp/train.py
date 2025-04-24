import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import yaml
from sklearn.preprocessing import StandardScaler

from base import Trainer

exp_no = '00001'
run_no = '00001'
exp_dir = Path(__file__).parent / f'exp{exp_no}' / f'run{run_no}'
config_filepath = exp_dir / 'config.yaml'

class RainfallDataset(Dataset):
    """
    降雨予測のためのデータセット
    """
    def __init__(self, csv_file, transform=None):
        self.df = pl.read_csv(csv_file)
        self.transform = transform
        
        # 日付を除外して特徴量とターゲットを抽出
        # カラム名を確認して、正しく特徴量とターゲットを分離
        self.features = self.df.drop(['id', 'day', 'rainfall']).to_numpy()
        self.targets = self.df['rainfall'].to_numpy()
        
        # 特徴量の次元数を記録（モデル初期化時に使用）
        self.input_dim = self.features.shape[1]
        
        # 特徴量のスケーリング
        if self.transform:
            # transform が fit_transform されているか確認
            if hasattr(self.transform, 'mean_'):
                self.features = self.transform.transform(self.features)
            else:
                self.features = self.transform.fit_transform(self.features)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 特徴量とターゲットをテンソルに変換
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        
        return features, target

class RainfallTrainer(Trainer):
    """
    降雨予測のためのトレーナー
    """
    
    def _init_model(self) -> nn.Module:
        """
        モデルを初期化します。
        
        Returns:
            初期化されたモデル
        """
        model_config = self.config
        module = import_module(model_config['model']['module'])
        model_name = model_config['model']['name']
        model_class = getattr(module, model_name)
        
        # データセットから入力次元を自動取得
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('data_path', 'tasks/BinaryRainfallPrediction/data/train.csv')
        temp_dataset = RainfallDataset(data_path)
        input_dim = temp_dataset.input_dim
        
        # 設定ファイルの入力次元を更新
        self.config['model']['params']['input_dim'] = input_dim
        
        # 更新した設定でモデルを初期化
        return model_class.from_config(model_config)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        データローダーを作成します。
        
        Returns:
            学習データローダーと検証データローダーのタプル
        """
        dataset_config = self.config.get('dataset', {})
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        train_ratio = dataset_config.get('train_ratio', 0.8)
        
        # データファイルのパス
        data_path = dataset_config.get('data_path', 'tasks/BinaryRainfallPrediction/data/train.csv')
        
        # スケーラーの作成
        scaler = StandardScaler()
        
        # フルデータセットの作成
        full_dataset = RainfallDataset(data_path, transform=scaler)
        
        # 学習データと検証データに分割
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # データローダーの作成
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
            
            # 予測と正解の比較
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

    def get_metrics_info(self) -> List[Dict[str, Any]]:
        """
        トラッキングする評価指標の情報を返す。
        """
        return [
            {'name': 'loss', 'modes': ['train', 'val'], 'display_name': 'Loss'},
            {'name': 'accuracy', 'modes': ['train', 'val'], 'display_name': 'Accuracy'},
            {'name': 'precision', 'modes': ['val'], 'display_name': 'Precision'},
            {'name': 'recall', 'modes': ['val'], 'display_name': 'Recall'},
            {'name': 'f1', 'modes': ['val'], 'display_name': 'F1 Score'}
        ]
    
    def validate(self, dataloader) -> Dict[str, float]:
        """
        検証を行う。
        
        Args:
            dataloader: 検証データのデータローダー
            
        Returns:
            検証結果を含む辞書
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 順伝播
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # 統計情報の更新
                total_loss += loss.item() * inputs.size(0)
                
                # 予測と正解の比較
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # 予測結果と正解を保存
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 精度、再現率、F1スコアの計算
        tp = np.sum((np.array(all_targets) == 1) & (np.array(all_preds) == 1))
        fp = np.sum((np.array(all_targets) == 0) & (np.array(all_preds) == 1))
        fn = np.sum((np.array(all_targets) == 1) & (np.array(all_preds) == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 平均損失と精度を計算
        avg_loss = total_loss / total
        avg_acc = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def save_predictions(self, dataloader, output_path: str):
        """
        予測結果をCSVファイルに保存する。
        
        Args:
            dataloader: データローダー
            output_path: 出力ファイルパス
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                
                # 順伝播
                outputs = self.model(inputs)
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        # 予測結果をDataFrameに変換
        df = pl.DataFrame({
            'predicted_probability': np.array(all_probs).flatten(),
            'predicted_class': np.array(all_preds).flatten()
        })
        
        # 出力ディレクトリがなければ作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # CSVファイルに保存
        df.to_csv(output_path, index=False) 

def main():
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    trainer = RainfallTrainer(config, exp_dir)
    train_dataloader, val_dataloader = trainer.create_dataloaders()
    trainer.train(train_dataloader, val_dataloader, config['training']['epochs'], config['training']['eval_interval'], config['training']['checkpoint_interval'])

if __name__ == '__main__':
    main()