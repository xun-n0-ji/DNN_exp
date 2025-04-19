import os
import json
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

def find_latest_experiment(base_dir: str = "experiments", task_name: Optional[str] = None, 
                         model_name: Optional[str] = None) -> Tuple[str, str, str]:
    """
    最新の実験を見つけます。
    
    Args:
        base_dir: 実験のベースディレクトリ
        task_name: タスク名 (指定されていない場合は最新のものを使用)
        model_name: モデル名 (指定されていない場合は最新のものを使用)
        
    Returns:
        タスク名、モデル名、実験名のタプル
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"実験ディレクトリ {base_dir} が見つかりません")
    
    # タスクを探す
    if task_name is None:
        tasks = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
        if not tasks:
            raise FileNotFoundError(f"ディレクトリ {base_dir} にタスクが見つかりません")
        
        # 最新のタスクを選択（アルファベット順）
        task_name = sorted(tasks)[-1]
    
    task_dir = os.path.join(base_dir, task_name)
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"タスクディレクトリ {task_dir} が見つかりません")
    
    # モデルを探す
    if model_name is None:
        models = [d for d in os.listdir(task_dir) 
                 if os.path.isdir(os.path.join(task_dir, d))]
        if not models:
            raise FileNotFoundError(f"ディレクトリ {task_dir} にモデルが見つかりません")
        
        # 最新のモデルを選択（アルファベット順）
        model_name = sorted(models)[-1]
    
    model_dir = os.path.join(task_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"モデルディレクトリ {model_dir} が見つかりません")
    
    # 実験を探す
    experiments = [d for d in os.listdir(model_dir) 
                  if os.path.isdir(os.path.join(model_dir, d)) 
                  and re.match(r"exp\d+", d)]
    if not experiments:
        raise FileNotFoundError(f"ディレクトリ {model_dir} に実験が見つかりません")
    
    # 番号が最大の実験を見つける
    def extract_number(exp_name):
        match = re.search(r"exp(\d+)", exp_name)
        return int(match.group(1)) if match else 0
    
    exp_name = sorted(experiments, key=extract_number)[-1]
    
    return task_name, model_name, exp_name

def save_results(results: Dict[str, Any], output_path: str):
    """
    実験結果をJSONファイルに保存します。
    
    Args:
        results: 保存する結果の辞書
        output_path: 出力ファイルのパス
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def plot_metrics(log_file: str, output_dir: str, metrics: List[str] = ['loss', 'accuracy']):
    """
    学習ログからメトリクスをプロットします。
    
    Args:
        log_file: ログファイルのパス
        output_dir: 出力ディレクトリ
        metrics: プロットするメトリクスのリスト
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"ログファイル {log_file} が見つかりません")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ログファイルの解析
    train_data = {'epoch': []}
    val_data = {'epoch': []}
    
    for metric in metrics:
        train_data[metric] = []
        val_data[metric] = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # トレーニングログの解析
            train_match = re.search(r"Epoch (\d+)/\d+, Train Loss: ([\d\.]+)", line)
            if train_match:
                epoch = int(train_match.group(1))
                loss = float(train_match.group(2))
                
                if epoch not in train_data['epoch']:
                    train_data['epoch'].append(epoch)
                    train_data['loss'].append(loss)
                else:
                    idx = train_data['epoch'].index(epoch)
                    train_data['loss'][idx] = loss
            
            # 検証ログの解析
            val_match = re.search(r"Epoch (\d+)/\d+, Val Loss: ([\d\.]+)", line)
            if val_match:
                epoch = int(val_match.group(1))
                loss = float(val_match.group(2))
                
                if epoch not in val_data['epoch']:
                    val_data['epoch'].append(epoch)
                    val_data['loss'].append(loss)
                else:
                    idx = val_data['epoch'].index(epoch)
                    val_data['loss'][idx] = loss
    
    # データフレームの作成
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # プロット
    for metric in metrics:
        if metric in train_df.columns and metric in val_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(train_df['epoch'], train_df[metric], label=f'Train {metric}')
            plt.plot(val_df['epoch'], val_df[metric], label=f'Val {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} vs. Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
            plt.close()

def compare_experiments(base_dir: str, task_name: str, model_name: str, 
                       exp_names: List[str], metric: str = 'loss'):
    """
    複数の実験を比較します。
    
    Args:
        base_dir: 実験のベースディレクトリ
        task_name: タスク名
        model_name: モデル名
        exp_names: 比較する実験名のリスト
        metric: 比較するメトリック
    """
    plt.figure(figsize=(12, 8))
    
    for exp_name in exp_names:
        log_file = os.path.join(base_dir, task_name, model_name, exp_name, 'train.log')
        if not os.path.exists(log_file):
            print(f"警告: ログファイル {log_file} が見つかりません")
            continue
        
        # 実験の設定を読み込み
        config_file = os.path.join(base_dir, task_name, model_name, exp_name, 'config.yaml')
        config_label = exp_name
        
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # 設定から適切なラベルを作成
            # 例: バッチサイズと学習率を表示
            batch_size = config.get('dataset', {}).get('batch_size', 'N/A')
            lr = config.get('optimizer', {}).get('params', {}).get('lr', 'N/A')
            config_label = f"{exp_name} (bs={batch_size}, lr={lr})"
        
        # ログデータの解析
        val_pattern = f"Val {metric.capitalize()}: ([\d\.]+)"
        epochs = []
        values = []
        
        with open(log_file, 'r') as f:
            for line in f:
                val_match = re.search(rf"Epoch (\d+)/\d+, {val_pattern}", line)
                if val_match:
                    epochs.append(int(val_match.group(1)))
                    values.append(float(val_match.group(2)))
        
        plt.plot(epochs, values, marker='o', label=config_label)
    
    plt.xlabel('Epoch')
    plt.ylabel(f'Validation {metric.capitalize()}')
    plt.title(f'Comparison of Experiments - {metric.capitalize()}')
    plt.legend()
    plt.grid(True)
    
    output_dir = os.path.join(base_dir, task_name, model_name, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
    plt.close() 