from abc import ABC, abstractmethod
import os
import yaml
import polars as pl
import torch
import logging
from typing import Dict, Any, Optional, List

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

        self._init_metrics_history()
        
    def _setup_logger(self) -> logging.Logger:
        """ロガーの設定"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # ファイルハンドラ
        #os.makedirs(self.exp_dir, exist_ok=True)
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
    
    def _init_metrics_history(self):
        """
        評価結果を記録するDataFrameを初期化します。
        """
        num_epochs = self.config['training'].get('epochs', 100)
        
        # 基本のカラム
        columns = {'epoch': list(range(1, num_epochs + 1))}
        
        # 評価指標ごとにカラムを追加
        metrics_info = self.get_metrics_info()
        for metric in metrics_info:
            metric_name = metric['name']
            for mode in metric['modes']:
                column_name = f'{mode}_{metric_name}'
                columns[column_name] = [None] * num_epochs
        
        # DataFrameの作成
        self.metrics_history = pl.DataFrame(columns)

    def _update_metrics_history(self, metrics: Dict[str, float], epoch: int, mode: str):
        """
        メトリクス履歴を更新します。
        
        Args:
            metrics: 更新するメトリクス
            epoch: 現在のエポック
            mode: 'train'または'val'
        """
        # 更新する行のインデックスを計算（0-indexed）
        row_idx = epoch - 1
        
        # 更新するデータの辞書を作成
        update_dict = {}
        
        # 定義された指標情報を使用して適切なカラムを更新
        metrics_info = self.get_metrics_info()
        for metric in metrics_info:
            metric_name = metric['name']
            if mode in metric['modes'] and metric_name in metrics:
                column_name = f'{mode}_{metric_name}'
                # 該当する行のみを更新
                self.metrics_history = self.metrics_history.with_columns(
                    pl.when(pl.col('epoch') == epoch)
                    .then(pl.lit(metrics[metric_name]))
                    .otherwise(pl.col(column_name))
                    .alias(column_name)
                )

    @abstractmethod
    def get_metrics_info(self) -> List[Dict[str, Any]]:
        """
        トラッキングする評価指標の情報を返します。
        
        Returns:
            各評価指標に関する情報のリスト。各要素は辞書で以下のキーを含む：
            - name: 指標の名前（例: 'loss', 'accuracy'）
            - modes: この指標を記録するモード（例: ['train', 'val']）
            - (オプション) display_name: 表示用の名前
        """
        raise NotImplementedError("サブクラスでget_metrics_infoを実装する必要があります")

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
            self.logger.error(f'Scheduler "{scheduler_name}" does not exist in PyTorch.')
            raise ValueError(f'Scheduler "{scheduler_name}" does not exist in PyTorch.')
        
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
        raise NotImplementedError('Need to implement "train_epoch" in subclass.')
    
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
        raise NotImplementedError('Need to implement "validate" in subclass.')
    
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
            self.logger.info(f"Epoch {epoch}: Saved the cirrent best model.")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        チェックポイントを読み込むメソッド。
        
        Args:
            checkpoint_path: チェックポイントファイルのパス
            
        Returns:
            読み込んだチェックポイント情報
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Could not find {checkpoint_path}.")
            raise FileNotFoundError(f"Could not found {checkpoint_path}.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint
    
    def train(self, train_dataloader, val_dataloader, num_epochs: int, eval_interval: int = 1, checkpoint_interval: int = 10):
        """
        学習の実行メソッド。
        
        Args:
            train_dataloader: 学習データのデータローダー
            val_dataloader: 検証データのデータローダー
            num_epochs: 学習エポック数
            eval_interval: 検証を行う間隔（エポック数）
        """
        self.logger.info("Start training...")
        
        mode = self.config['training']['optimization_mode']
        if mode == 'min':
            best_val_metric = float('inf')
        elif mode == 'max':
            best_val_metric = float('-inf')
        else:
            raise ValueError(f"Invalid optimization_mode: {mode}. Expected 'min' or 'max'.")
        
        for epoch in range(1, num_epochs + 1):
            # 学習
            train_metrics = self.train_epoch(train_dataloader)
            self._update_metrics_history(train_metrics, epoch, 'train')
            
            # ログ出力
            log_message = f"Epoch {epoch}/{num_epochs}, Train Loss: {train_metrics['loss']:.4f}"
            self.logger.info(log_message)
            
            # 検証
            if epoch % eval_interval == 0 or epoch % checkpoint_interval == 0:
                val_metrics = self.validate(val_dataloader)
                self._update_metrics_history(val_metrics, epoch, 'val')
                
                # 検証結果のログ出力
                log_message = f"Validation, Val Loss: {val_metrics['loss']:.4f}"
                if 'accuracy' in val_metrics:
                    log_message += f", Val Accuracy: {val_metrics['accuracy']:.4f}"
                self.logger.info(log_message)
                
                # スケジューラの更新
                if self.scheduler is not None:
                    # ReduceLROnPlateauかどうかを確認
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # 監視する指標を渡す（通常は検証ロス）
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # 最良モデルの判断
                metric_name = 'loss'  # 基本は損失で判断
                watch_metric = val_metrics.get(metric_name, val_metrics['loss'])
                
                is_best = False
                if (mode == 'min' and watch_metric < best_val_metric) or \
                   (mode == 'max' and watch_metric > best_val_metric):
                    best_val_metric = watch_metric
                    is_best = True
                
                # 設定に応じてモデルを保存
                save_best = self.config['training'].get('save_best', True)
                if save_best and is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                
                save_last = self.config['training'].get('save_last', True)
                if save_last:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
            else:
                # 検証を行わない場合でもスケジューラを更新（ReduceLROnPlateau以外）
                if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
        
        # 訓練終了時の処理
        self.logger.info(f"Training completed after {num_epochs} epochs.")
