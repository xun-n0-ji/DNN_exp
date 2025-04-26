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
    Finds the latest experiment.

    Args:
        base_dir: Base directory of experiments
        task_name: Task name (if not specified, latest one is used)
        model_name: Model name (if not specified, latest one is used)

    Returns:
        Tuple of task name, model name, experiment name
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Experiment directory {base_dir} not found")
    
    # Find a task
    if task_name is None:
        tasks = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d))]
        if not tasks:
            raise FileNotFoundError(f"No tasks found in directory {base_dir}")
        
        # Select the most recent task (alphabetical order)
        task_name = sorted(tasks)[-1]
    
    task_dir = os.path.join(base_dir, task_name)
    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task directory {task_dir} not found")
    
    # Find a model
    if model_name is None:
        models = [d for d in os.listdir(task_dir) 
                 if os.path.isdir(os.path.join(task_dir, d))]
        if not models:
            raise FileNotFoundError(f"No models found in directory {task_dir}")
        
        # Select the latest models (alphabetical order)
        model_name = sorted(models)[-1]
    
    model_dir = os.path.join(task_dir, model_name)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} not found")
    
    # 実験を探す
    experiments = [d for d in os.listdir(model_dir) 
                  if os.path.isdir(os.path.join(model_dir, d)) 
                  and re.match(r"exp\d+", d)]
    if not experiments:
        raise FileNotFoundError(f"No experiments found in directory {model_dir}")
    
    # 番号が最大の実験を見つける
    def extract_number(exp_name):
        match = re.search(r"exp(\d+)", exp_name)
        return int(match.group(1)) if match else 0
    
    exp_name = sorted(experiments, key=extract_number)[-1]
    
    return task_name, model_name, exp_name

def save_results(results: Dict[str, Any], output_path: str):
    """
    Save the experiment results to a JSON file.

    Args:
        results: A dictionary of results to save
        output_path: Path to the output file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

def plot_metrics(log_file: str, output_dir: str, metrics: List[str] = ['loss', 'accuracy']):
    """
    Plot metrics from the training log.

    Args:
        log_file: Path to log file
        output_dir: Output directory
        metrics: List of metrics to plot
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file {log_file} not found")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyzing log files
    train_data = {'epoch': []}
    val_data = {'epoch': []}
    
    for metric in metrics:
        train_data[metric] = []
        val_data[metric] = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Analyzing training logs
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
            
            # Analysis of validation logs
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
    
    # Creating a DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Plot
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
    Compare multiple experiments.

    Args:
        base_dir: Base directory of experiments
        task_name: Task name
        model_name: Model name
        exp_names: List of experiment names to compare
        metric: Metric to compare
    """
    plt.figure(figsize=(12, 8))
    
    for exp_name in exp_names:
        log_file = os.path.join(base_dir, task_name, model_name, exp_name, 'train.log')
        if not os.path.exists(log_file):
            print(f"WARNING: Log file {log_file} not found")
            continue
        
        # Load the experiment settings
        config_file = os.path.join(base_dir, task_name, model_name, exp_name, 'config.yaml')
        config_label = exp_name
        
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
            # Create appropriate labels from the configuration
            # Example: Show batch size and learning rate
            batch_size = config.get('dataset', {}).get('batch_size', 'N/A')
            lr = config.get('optimizer', {}).get('params', {}).get('lr', 'N/A')
            config_label = f"{exp_name} (bs={batch_size}, lr={lr})"
        
        # Analyzing log data
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