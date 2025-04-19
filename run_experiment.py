import os
import argparse
import yaml
from typing import Dict, Any, Optional, Type

from experiment import Experiment
from config_utils import ConfigManager
from trainer import Trainer

def parse_args():
    """
    コマンドライン引数を解析します。
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(description='実験実行スクリプト')
    parser.add_argument('--task', type=str, required=True, help='タスク名')
    parser.add_argument('--model', type=str, required=True, help='モデル名')
    parser.add_argument('--exp', type=str, required=True, help='実験名')
    parser.add_argument('--base_dir', type=str, default='experiments', help='実験のベースディレクトリ')
    parser.add_argument('--base_config', type=str, help='基本設定ファイルのパス')
    parser.add_argument('--update_config', type=str, help='更新設定ファイルのパス')
    parser.add_argument('--trainer_class', type=str, required=True, help='トレーナークラス名')
    
    return parser.parse_args()

def get_trainer_class(trainer_class_name: str) -> Type[Trainer]:
    """
    トレーナークラスを取得します。
    
    Args:
        trainer_class_name: トレーナークラス名
        
    Returns:
        トレーナークラス
    """
    try:
        # 'module_name.ClassName' 形式のクラス名を解析
        if '.' in trainer_class_name:
            module_name, class_name = trainer_class_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            trainer_class = getattr(module, class_name)
        else:
            # クラス名のみの場合（同じディレクトリ内のモジュールを想定）
            # この場合は、現在のモジュールの名前空間内からクラスを探す
            globals_copy = globals().copy()
            if trainer_class_name in globals_copy:
                trainer_class = globals_copy[trainer_class_name]
            else:
                # トレーナーモジュールから直接インポート
                module = __import__('trainers', fromlist=[trainer_class_name])
                trainer_class = getattr(module, trainer_class_name)
        
        # クラスがTrainerのサブクラスであることを確認
        if not issubclass(trainer_class, Trainer):
            raise TypeError(f"{trainer_class_name}はTrainerのサブクラスではありません")
        
        return trainer_class
    except (ImportError, AttributeError, TypeError) as e:
        raise ValueError(f"トレーナークラス {trainer_class_name} を取得できませんでした: {e}")

def main():
    """
    メイン関数
    """
    args = parse_args()
    
    # 実験インスタンスの作成
    experiment = Experiment(args.task, args.model, args.exp, args.base_dir)
    
    # 更新設定の読み込み
    update_config = None
    if args.update_config and os.path.exists(args.update_config):
        update_config = ConfigManager.load_config(args.update_config)
    
    # 実験の初期化
    experiment.initialize(args.base_config, update_config)
    
    # 設定の取得
    config = experiment.get_config()
    
    # トレーナークラスの取得
    trainer_class = get_trainer_class(args.trainer_class)
    
    # トレーナーインスタンスの作成
    trainer = trainer_class(config, experiment.exp_dir)
    
    # データローダーの作成（トレーナー固有の処理）
    train_dataloader, val_dataloader = trainer.create_dataloaders()
    
    # 学習の実行
    trainer.train(
        train_dataloader, 
        val_dataloader, 
        config.get('training', {}).get('epochs', 10),
        config.get('training', {}).get('eval_interval', 1)
    )

if __name__ == '__main__':
    main() 