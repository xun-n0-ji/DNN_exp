import os
from pathlib import Path
from typing import Dict, Any, Tuple, List
from importlib import import_module

import torch
import torch.nn as nn
import polars as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import yaml
from sklearn.preprocessing import StandardScaler

from core import Trainer

exp_no = '00001'
run_no = '00001'

RUN_NAME = f'run{run_no}'
EXP_DIR = Path(__file__).parent / f'exp{exp_no}' / RUN_NAME
config_filepath = EXP_DIR / 'config.yaml'

class RainfallDataset(Dataset):
    """
    Rainfall prediction dataset
    """
    def __init__(self, csv_file, transform=None):
        self.df = pl.read_csv(csv_file)
        self.transform = transform
        
        # Extract features and targets by excluding dates
        # Check column names to correctly separate features and targets
        self.features = self.df.drop(['id', 'day', 'rainfall']).to_numpy()
        self.targets = self.df['rainfall'].to_numpy()
        
        # Record the number of feature dimensions (used when initializing the model)
        self.input_dim = self.features.shape[1]
        
        # Feature Scaling
        if self.transform:
            # Check if transform is fit_transform
            if hasattr(self.transform, 'mean_'):
                self.features = self.transform.transform(self.features)
            else:
                self.features = self.transform.fit_transform(self.features)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Convert features and targets to tensors
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        
        return features, target

class RainfallTrainer(Trainer):
    """
    Rainfall forecast trainer
    """
    
    def _init_model(self) -> nn.Module:
        """
        Initializes the model.

        Returns:
            The initialized model.
        """
        model_config = self.config
        module = import_module(model_config['model']['module'])
        model_name = model_config['model']['name']
        model_class = getattr(module, model_name)
        
        # Automatically obtain input dimensions from the dataset
        dataset_config = self.config.get('dataset', {})
        data_path = dataset_config.get('data_path', 'tasks/BinaryRainfallPrediction/data/train.csv')
        temp_dataset = RainfallDataset(data_path)
        input_dim = temp_dataset.input_dim
        
        # Updated input dimensions in config file
        self.config['model']['params']['input_dim'] = input_dim
        
        # Initialize the model with the updated settings
        return model_class.from_config(model_config)
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates a DataLoader.

        Returns:
            A tuple of training data loader and validation data loader.
        """
        dataset_config = self.config.get('dataset', {})
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        train_ratio = dataset_config.get('train_ratio', 0.8)
        
        # Data file Path
        data_path = dataset_config.get('data_path', 'tasks/BinaryRainfallPrediction/data/train.csv')
        
        # Creating a scaler
        scaler = StandardScaler()
        
        # Creating a full dataset
        full_dataset = RainfallDataset(data_path, transform=scaler)
        
        # Split into training and validation data
        train_size = int(len(full_dataset) * train_ratio)
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Creating a Data Loader
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
        Train for one epoch.
        
        Args:
            dataloader: Data loader for training data
            
        Returns:
            A dictionary containing the training results for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Reset gradient to zero
            self.optimizer.zero_grad()
            
            # Forward propagation
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backpropagation and Optimization
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            
            # Comparison of predictions and correct answers
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
        
        # Calculate the average loss and accuracy for an epoch
        epoch_loss = total_loss / total
        epoch_acc = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }
        
        return metrics

    def get_metrics_info(self) -> List[Dict[str, Any]]:
        """
        Returns information about the metrics you are tracking.
        """
        return [
            {'name': 'loss', 'modes': ['train', 'val'], 'display_name': 'Loss'},
            {'name': 'accuracy', 'modes': ['train', 'val'], 'display_name': 'Accuracy'},
            {'name': 'precision', 'modes': ['val'], 'display_name': 'Precision'},
            {'name': 'recall', 'modes': ['val'], 'display_name': 'Recall'},
            {'name': 'f1', 'modes': ['val'], 'display_name': 'F1 Score'}
        ]
    
    def validate(self, dataloader) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Performs validation.

        Args:
            dataloader: Data loader for validation data

        Returns:
            Dictionary containing validation results
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pred_batches = []
        target_batches = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward propagation
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Update statistics
                total_loss += loss.item() * inputs.size(0)
                
                # Comparison of predictions and correct answers
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                
                # Save prediction results and correct answers
                pred_batches.append(predicted.cpu().numpy())
                target_batches.append(targets.cpu().numpy())
        
        all_preds = np.concatenate(pred_batches).ravel()
        all_targets = np.concatenate(target_batches).ravel()

        # Calculate precision, recall and F1 score
        tp = np.sum((np.array(all_targets) == 1) & (np.array(all_preds) == 1))
        fp = np.sum((np.array(all_targets) == 0) & (np.array(all_preds) == 1))
        fn = np.sum((np.array(all_targets) == 1) & (np.array(all_preds) == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total
        avg_acc = correct / total if total > 0 else 0.0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        results = {
            'preds': all_preds,
            'targets': all_targets
        }
        
        return metrics, results
    
    def save_predictions(self, dataloader, output_path: str):
        """
        Save the prediction results to a CSV file.

        Args:
            dataloader: Data loader
            output_path: Output file path
        """
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                
                # Forward propagation
                outputs = self.model(inputs)
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
        
        # Convert the prediction results to a DataFrame
        df = pl.DataFrame({
            'predicted_probability': np.array(all_probs).flatten(),
            'predicted_class': np.array(all_preds).flatten()
        })
        
        # Create the output directory if it does not exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV file
        df.to_csv(output_path, index=False) 

def main():
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    trainer = RainfallTrainer(config, EXP_DIR, RUN_NAME)
    train_dataloader, val_dataloader = trainer.create_dataloaders()
    trainer.train(train_dataloader, val_dataloader, config['training']['epochs'], config['training']['eval_interval'], config['training']['checkpoint_interval'])

if __name__ == '__main__':
    main()