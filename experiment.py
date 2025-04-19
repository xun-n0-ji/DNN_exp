import os
import yaml
import json
import datetime
from typing import Dict, Any, Optional, List, Type

from config_utils import ConfigManager

class Experiment:
    """
    実験を管理するクラス。
    実験ディレクトリの構造を管理し、設定ファイルの読み込みと保存を行います。
    """
    
    def __init__(self, task_name: str, model_name: str, exp_name: str, base_dir: str = "experiments"):
        """
        実験の初期化
        
        Args:
            task_name: タスク名
            model_name: モデル名
            exp_name: 実験名
            base_dir: 実験のベースディレクトリ
        """
        self.task_name = task_name
        self.model_name = model_name
        self.exp_name = exp_name
        self.base_dir = base_dir
        
        # 実験ディレクトリのパス
        self.exp_dir = os.path.join(self.base_dir, self.task_name, self.model_name, self.exp_name)
        
        # 実験ディレクトリを作成
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 設定ファイルのパス
        self.config_path = os.path.join(self.exp_dir, "config.yaml")
        
        # メタデータのパス
        self.metadata_path = os.path.join(self.exp_dir, "metadata.json")
    
    def initialize(self, base_config_path: Optional[str] = None, update_config: Optional[Dict[str, Any]] = None):
        """
        実験を初期化します。
        
        Args:
            base_config_path: 基本設定ファイルのパス。指定がない場合はデフォルト設定を使用します。
            update_config: 更新設定。指定がある場合は基本設定をこの設定で更新します。
        """
        # 基本設定の読み込み
        if base_config_path is not None and os.path.exists(base_config_path):
            base_config = ConfigManager.load_config(base_config_path)
        else:
            base_config = ConfigManager.get_default_config()
        
        # 更新設定の適用
        if update_config is not None:
            config = ConfigManager.update_config(base_config, update_config)
        else:
            config = base_config
        
        # 設定ファイルの保存
        ConfigManager.save_config(config, self.config_path)
        
        # メタデータの保存
        metadata = {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'exp_name': self.exp_name,
            'created_at': datetime.datetime.now().isoformat(),
            'base_config_path': base_config_path
        }
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
    
    def get_config(self) -> Dict[str, Any]:
        """
        実験の設定を取得します。
        
        Returns:
            設定情報を含む辞書
        """
        return ConfigManager.load_config(self.config_path)
    
    def update_config(self, update_config: Dict[str, Any]):
        """
        設定を更新します。
        
        Args:
            update_config: 更新設定を含む辞書
        """
        current_config = self.get_config()
        updated_config = ConfigManager.update_config(current_config, update_config)
        ConfigManager.save_config(updated_config, self.config_path)
    
    @staticmethod
    def list_tasks(base_dir: str = "experiments") -> List[str]:
        """
        タスクの一覧を取得します。
        
        Args:
            base_dir: 実験のベースディレクトリ
            
        Returns:
            タスク名のリスト
        """
        if not os.path.exists(base_dir):
            return []
        
        return [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
    
    @staticmethod
    def list_models(task_name: str, base_dir: str = "experiments") -> List[str]:
        """
        指定されたタスクのモデル一覧を取得します。
        
        Args:
            task_name: タスク名
            base_dir: 実験のベースディレクトリ
            
        Returns:
            モデル名のリスト
        """
        task_dir = os.path.join(base_dir, task_name)
        if not os.path.exists(task_dir):
            return []
        
        return [d for d in os.listdir(task_dir) 
                if os.path.isdir(os.path.join(task_dir, d))]
    
    @staticmethod
    def list_experiments(task_name: str, model_name: str, base_dir: str = "experiments") -> List[str]:
        """
        指定されたタスクとモデルの実験一覧を取得します。
        
        Args:
            task_name: タスク名
            model_name: モデル名
            base_dir: 実験のベースディレクトリ
            
        Returns:
            実験名のリスト
        """
        model_dir = os.path.join(base_dir, task_name, model_name)
        if not os.path.exists(model_dir):
            return []
        
        return [d for d in os.listdir(model_dir) 
                if os.path.isdir(os.path.join(model_dir, d))]
    
    @staticmethod
    def find_best_experiment(task_name: str, model_name: str, metric_key: str, 
                            base_dir: str = "experiments", is_max: bool = False) -> Optional[str]:
        """
        指定されたタスクとモデルの中で最良の実験を探します。
        
        Args:
            task_name: タスク名
            model_name: モデル名
            metric_key: 評価指標のキー
            base_dir: 実験のベースディレクトリ
            is_max: Trueの場合は最大値、Falseの場合は最小値を最良とする
            
        Returns:
            最良の実験名、見つからない場合はNone
        """
        experiments = Experiment.list_experiments(task_name, model_name, base_dir)
        if not experiments:
            return None
        
        best_exp = None
        best_value = float("-inf") if is_max else float("inf")
        
        for exp_name in experiments:
            exp = Experiment(task_name, model_name, exp_name, base_dir)
            checkpoint_path = os.path.join(exp.exp_dir, "checkpoints", "checkpoint_best.pth")
            
            if not os.path.exists(checkpoint_path):
                continue
            
            try:
                import torch
                checkpoint = torch.load(checkpoint_path, map_location="cpu")
                
                if "metrics" in checkpoint and metric_key in checkpoint["metrics"]:
                    value = checkpoint["metrics"][metric_key]
                    
                    if (is_max and value > best_value) or (not is_max and value < best_value):
                        best_value = value
                        best_exp = exp_name
            except Exception as e:
                print(f"実験 {exp_name} のチェックポイントを読み込めませんでした: {e}")
        
        return best_exp 