from abc import ABC, abstractmethod
import os
import yaml
import torch
import logging
from typing import Dict, Any, Optional

class Trainer(ABC):
    """
    学習プロセスの基本クラス。
    全てのタスク固有のトレーナーはこのクラスを継承する必要があります。
    """
    def __init__(self, config: Dict[str, Any], exp_dir: str):
        """
        トレーナーの初期化
        
        Args:
            config: 設定パラメータを含む辞書
            exp_dir: 実験結果を保存するディレクトリパス
        """
        self.config = config
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ロギングの設定
        self.logger = self._setup_logger()
        
        # モデル、損失関数、オプティマイザの初期化
        self.model = self._init_model()
        self.model.to(self.device)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラ
        os.makedirs(self.exp_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.exp_dir, 'train.log'))
        file_handler.setLevel(logging.INFO)
        
        # コンソールハンドラ
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # フォーマッタ
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # ハンドラをロガーに追加
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    def _init_model(self) -> torch.nn.Module:
        """
        モデルを初期化するメソッド。
        継承先で実装する必要があります。
        
        Returns:
            初期化されたモデル
        """
        raise NotImplementedError("サブクラスで_init_modelを実装する必要があります")
    
    def _init_criterion(self):
        """
        損失関数を初期化するメソッド。
        configに基づいて適切な損失関数を返します。
        
        Returns:
            初期化された損失関数
        """
        criterion_name = self.config.get('criterion', {}).get('name', 'CrossEntropyLoss')
        criterion_params = self.config.get('criterion', {}).get('params', {})
        
        if not hasattr(torch.nn, criterion_name):
            self.logger.error(f"損失関数 {criterion_name} は PyTorch に存在しません")
            raise ValueError(f"損失関数 {criterion_name} は PyTorch に存在しません")
        
        criterion_class = getattr(torch.nn, criterion_name)
        return criterion_class(**criterion_params)
    
    def _init_optimizer(self):
        """
        オプティマイザを初期化するメソッド。
        configに基づいて適切なオプティマイザを返します。
        
        Returns:
            初期化されたオプティマイザ
        """
        optimizer_name = self.config.get('optimizer', {}).get('name', 'Adam')
        optimizer_params = self.config.get('optimizer', {}).get('params', {'lr': 0.001})
        
        if not hasattr(torch.optim, optimizer_name):
            self.logger.error(f"オプティマイザ {optimizer_name} は PyTorch に存在しません")
            raise ValueError(f"オプティマイザ {optimizer_name} は PyTorch に存在しません")
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        return optimizer_class(self.model.parameters(), **optimizer_params)
    
    def _init_scheduler(self):
        """
        学習率スケジューラを初期化するメソッド。
        configに基づいて適切なスケジューラを返します。
        
        Returns:
            初期化されたスケジューラ、または設定されていない場合はNone
        """
        scheduler_config = self.config.get('scheduler', {})
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name')
        scheduler_params = scheduler_config.get('params', {})
        
        if not scheduler_name:
            return None
        
        if not hasattr(torch.optim.lr_scheduler, scheduler_name):
            self.logger.error(f"スケジューラ {scheduler_name} は PyTorch に存在しません")
            raise ValueError(f"スケジューラ {scheduler_name} は PyTorch に存在しません")
        
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
        return scheduler_class(self.optimizer, **scheduler_params)
    
    @abstractmethod
    def train_epoch(self, dataloader):
        """
        1エポックの学習を行うメソッド。
        継承先で実装する必要があります。
        
        Args:
            dataloader: 学習データのデータローダー
            
        Returns:
            エポックの学習結果（損失など）
        """
        raise NotImplementedError("サブクラスでtrain_epochを実装する必要があります")
    
    @abstractmethod
    def validate(self, dataloader):
        """
        検証を行うメソッド。
        継承先で実装する必要があります。
        
        Args:
            dataloader: 検証データのデータローダー
            
        Returns:
            検証結果（損失、精度など）
        """
        raise NotImplementedError("サブクラスでvalidateを実装する必要があります")
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        チェックポイントを保存するメソッド。
        
        Args:
            epoch: 現在のエポック
            metrics: 保存する評価指標
            is_best: 最良のモデルかどうか
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_dir = os.path.join(self.exp_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 通常のチェックポイントを保存
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # 最新のチェックポイントとして保存
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        # 最良のモデルとして保存
        if is_best:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_best.pth'))
            self.logger.info(f"Epoch {epoch}: 最良のモデルを保存しました")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        チェックポイントを読み込むメソッド。
        
        Args:
            checkpoint_path: チェックポイントファイルのパス
            
        Returns:
            読み込んだチェックポイント情報
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"チェックポイントファイル {checkpoint_path} が見つかりません")
            raise FileNotFoundError(f"チェックポイントファイル {checkpoint_path} が見つかりません")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def train(self, train_dataloader, val_dataloader, num_epochs: int, eval_interval: int = 1):
        """
        学習の実行メソッド。
        
        Args:
            train_dataloader: 学習データのデータローダー
            val_dataloader: 検証データのデータローダー
            num_epochs: 学習エポック数
            eval_interval: 検証を行う間隔（エポック数）
        """
        self.logger.info("学習を開始します")
        
        best_val_metric = float('inf')  # 最小化する場合
        # best_val_metric = float('-inf')  # 最大化する場合
        
        for epoch in range(1, num_epochs + 1):
            # 学習
            train_metrics = self.train_epoch(train_dataloader)
            train_loss = train_metrics.get('loss', 0.0)
            self.logger.info(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}")
            
            # スケジューラの更新
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 検証
            if epoch % eval_interval == 0:
                val_metrics = self.validate(val_dataloader)
                val_loss = val_metrics.get('loss', 0.0)
                
                # ここではval_lossを最小化する例を示していますが、
                # 他の指標（accuracyなど）を最大化したい場合は条件を変更してください
                is_best = val_loss < best_val_metric
                if is_best:
                    best_val_metric = val_loss
                
                self.logger.info(f"Epoch {epoch}/{num_epochs}, Val Loss: {val_loss:.4f}")
                
                # チェックポイントの保存
                self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
        
        self.logger.info("学習が完了しました")
