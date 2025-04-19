import os
import yaml
from typing import Dict, Any, Optional
import copy

class ConfigManager:
    """
    設定ファイルの管理クラス。
    基本設定ファイルと更新設定ファイルを読み込み、マージした設定を提供します。
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        YAMLファイルから設定を読み込みます。
        
        Args:
            config_path: 設定ファイルのパス
            
        Returns:
            設定情報を含む辞書
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイル {config_path} が見つかりません")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config if config is not None else {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        設定をYAMLファイルに保存します。
        
        Args:
            config: 保存する設定情報を含む辞書
            config_path: 保存先のファイルパス
        """
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    @staticmethod
    def update_config(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        基本設定を更新設定でマージします。
        
        Args:
            base_config: 基本設定を含む辞書
            update_config: 更新設定を含む辞書
            
        Returns:
            マージされた設定を含む辞書
        """
        config = copy.deepcopy(base_config)
        
        def _update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = _update_dict(d[k], v)
                else:
                    d[k] = v
            return d
        
        return _update_dict(config, update_config)
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        デフォルト設定を返します。
        
        Returns:
            デフォルト設定を含む辞書
        """
        return {
            'model': {
                'name': 'DefaultModel',
                'params': {}
            },
            'dataset': {
                'batch_size': 32,
                'num_workers': 4
            },
            'training': {
                'epochs': 10,
                'eval_interval': 1
            },
            'criterion': {
                'name': 'CrossEntropyLoss',
                'params': {}
            },
            'optimizer': {
                'name': 'Adam',
                'params': {
                    'lr': 0.001
                }
            },
            'scheduler': {
                'name': 'StepLR',
                'params': {
                    'step_size': 5,
                    'gamma': 0.1
                }
            }
        } 