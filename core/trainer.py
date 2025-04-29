from abc import ABC, abstractmethod
import os
from pathlib import Path
import polars as pl
import torch
import logging
from typing import Dict, Any, Optional, List

from .exp_manager import ExpManager

class Trainer(ABC):
    """
    Base class for the learning process.
    All task-specific trainers must inherit from this class.
    """
    def __init__(self, config: Dict[str, Any], exp_dir: str, run_name:str):
        """
        Initializing the trainer
        
        Args:
            config: A dictionary containing configuration parameters
            exp_dir: The directory path to store the experiment results
        """
        self.config = config
        self.exp_dir = exp_dir
        self.checkpoint_dir = Path(self.exp_dir) / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self._setup_exp_manager(run_name)

        # Configuring Logging
        self.logger = self._setup_logger()
        
        # Initializing the model, loss function, and optimizer
        self.model = self._init_model()
        self.model.to(self.device)
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        self._init_metrics_history()
        self.last_val_results = {}

    def _setup_exp_manager(self, run_name):
        task_name = self.config.get('task_name')
        metrics_logger_name = self.config.get('metrics_logger')
        self.exp_manager = ExpManager(task_name, run_name, metrics_logger_name)

    # NOTE: Have to change the name
    def _setup_logger(self) -> logging.Logger:
        """Configuring the Logger"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # File Handler
        #os.makedirs(self.exp_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(self.exp_dir, 'train.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add a handler to a logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _set_last_val_results(self, **kwargs):
        self.last_val_results.update(**kwargs)
    
    @abstractmethod
    def _init_model(self) -> torch.nn.Module:
        """
        A method to initialize the model.
        This must be implemented by the inheritor.

        Returns:
            The initialized model.
        """
        raise NotImplementedError("Subclasses must implement _init_model")
    
    def _init_metrics_history(self):
        """
        Initialize a DataFrame to record the evaluation results.
        """
        num_epochs = self.config['training'].get('epochs', 100)
        
        # Basic Columns
        columns = {'epoch': list(range(1, num_epochs + 1))}
        
        # Add a column for each evaluation metric
        metrics_info = self.get_metrics_info()
        for metric in metrics_info:
            metric_name = metric['name']
            for mode in metric['modes']:
                column_name = f'{mode}_{metric_name}'
                columns[column_name] = [None] * num_epochs
        
        # Creating a DataFrame
        self.metrics_history = pl.DataFrame(columns)

    def _update_metrics_history(self, metrics: Dict[str, float], epoch: int, mode: str):
        """
        Updates metric history.

        Args:
            metrics: Metrics to update
            epoch: Current epoch
            mode: 'train' or 'val'
        """
        # Calculate the index of the row to update (0-indexed)
        row_idx = epoch - 1
        
        # Create a dictionary of the data to be updated
        update_dict = {}
        
        # Updates the appropriate columns with the defined metric information
        metrics_info = self.get_metrics_info()
        for metric in metrics_info:
            metric_name = metric['name']
            if mode in metric['modes'] and metric_name in metrics:
                column_name = f'{mode}_{metric_name}'
                # Update only applicable rows
                self.metrics_history = self.metrics_history.with_columns(
                    pl.when(pl.col('epoch') == epoch)
                    .then(pl.lit(metrics[metric_name]))
                    .otherwise(pl.col(column_name))
                    .alias(column_name)
                )

    @abstractmethod
    def get_metrics_info(self) -> List[Dict[str, Any]]:
        """
        Returns information about the metrics being tracked.

        Returns:
            A list of information about each metric. Each element is a dictionary with the following keys:
            - name: Name of the metric (e.g. 'loss', 'accuracy')
            - modes: Modes to record this metric (e.g. ['train', 'val'])
            - (Optional) display_name: Name for display
        """
        raise NotImplementedError("Subclasses must implement get_metrics_info")

    def _init_criterion(self):
        """
        A method to initialize a loss function.
        Returns an appropriate loss function based on the config.

        Returns:
            The initialized loss function.
        """
        criterion_name = self.config.get('criterion', {}).get('name', 'CrossEntropyLoss')
        criterion_params = self.config.get('criterion', {}).get('params', {})
        
        if not hasattr(torch.nn, criterion_name):
            self.logger.error(f"Loss function {criterion_name} does not exist in PyTorch")
            raise ValueError(f"Loss function {criterion_name} does not exist in PyTorch")
        
        criterion_class = getattr(torch.nn, criterion_name)
        return criterion_class(**criterion_params)
    
    def _init_optimizer(self):
        """
        A method to initialize the optimizer.
        Returns an appropriate optimizer based on the "config".

        Returns:
            The initialized optimizer.
        """
        optimizer_name = self.config.get('optimizer', {}).get('name', 'Adam')
        optimizer_params = self.config.get('optimizer', {}).get('params', {'lr': 0.001})
        
        if not hasattr(torch.optim, optimizer_name):
            self.logger.error(f"Optimizer {optimizer_name} does not exist in PyTorch")
            raise ValueError(f"Optimizer {optimizer_name} does not exist in PyTorch")
        
        optimizer_class = getattr(torch.optim, optimizer_name)
        return optimizer_class(self.model.parameters(), **optimizer_params)
    
    def _init_scheduler(self):
        """
        A method to initialize the learning rate scheduler.
        Returns an appropriate scheduler based on the config.

        Returns:
            The initialized scheduler, or None if not set.
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
        A method to train for one epoch.
        This must be implemented in the inheritor.

        Args:
            dataloader: Data loader for training data

        Returns:
            The training result of the epoch (loss, etc.)
        """
        raise NotImplementedError('Subclasses must implement train_epoch.')
    
    @abstractmethod
    def validate(self, dataloader):
        """
        The method to perform validation.
        Must be implemented in inheritors.

        Args:
            dataloader: Data loader for validation data

        Returns:
            Validation result (loss, accuracy, etc.)
        """
        raise NotImplementedError('Subclasses must implement validate.')
    
    def save_checkpoint(self, filename:str, epoch: int, metrics: Dict[str, float]):
        """
        Method to save checkpoint.

        Args:
            epoch: Current epoch
            metrics: Evaluation metrics to save
            is_best: Is this the best model?
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        
        # Save regular checkpoints
        torch.save(checkpoint, str(self.checkpoint_dir / filename))
        
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Method to read checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Read checkpoint information
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
        Training execution method.

        Args:
            train_dataloader: Data loader for training data
            val_dataloader: Data loader for validation data
            num_epochs: Number of training epochs
            eval_interval: Interval between validations (number of epochs)
        """
        self.logger.info("Start training...")
        
        mode = self.config['training']['optimization_mode']
        if mode == 'min':
            best_val_metric = float('inf')
        elif mode == 'max':
            best_val_metric = float('-inf')
        else:
            raise ValueError(f"Invalid optimization_mode: {mode}. Expected 'min' or 'max'.")
        
        with self.exp_manager:
            for epoch in range(1, num_epochs + 1):
                # Training
                train_metrics = self.train_epoch(train_dataloader)
                self._update_metrics_history(train_metrics, epoch, 'train')
                self.exp_manager.log_metrics(train_metrics, epoch, 'train')
                
                # Log output
                log_message = f"Epoch {epoch}/{num_epochs}, Train Loss: {train_metrics['loss']:.4f}"
                self.logger.info(log_message)
                
                # Validation
                if epoch % eval_interval == 0 \
                    or epoch % checkpoint_interval == 0 \
                    or epoch == num_epochs - 1:
                    val_metrics, val_results = self.validate(val_dataloader)
                    self._update_metrics_history(val_metrics, epoch, 'val')
                    self.exp_manager.log_metrics(val_metrics, epoch, 'val')
                    
                    # Log output of verification results
                    log_message = f"Validation, Val Loss: {val_metrics['loss']:.4f}"
                    if 'accuracy' in val_metrics:
                        log_message += f", Val Accuracy: {val_metrics['accuracy']:.4f}"
                    self.logger.info(log_message)
                    
                    # Scheduler Updates
                    if self.scheduler is not None:
                        # Check if ReduceLROnPlateau
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            # Pass in the metric you want to monitor (usually validation loss)
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()
                    
                    # Determining the best model
                    metric_name = 'loss'  # Basically, judge based on losses
                    watch_metric = val_metrics.get(metric_name, val_metrics['loss'])
                    
                    is_best = False
                    if (mode == 'min' and watch_metric < best_val_metric) or \
                    (mode == 'max' and watch_metric > best_val_metric):
                        best_val_metric = watch_metric
                        is_best = True
                    
                    if epoch % checkpoint_interval == 0:
                        self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch, val_metrics)

                    # Save the model according to your settings
                    save_best = self.config['training'].get('save_best', True)
                    if save_best and is_best:
                        self.save_checkpoint('checkpoint_best.pth', epoch, val_metrics)
                    
                    save_latest = self.config['training'].get('save_latest', True)
                    if save_latest:
                        self.save_checkpoint('checkpoint_latest.pth', epoch, val_metrics)
                else:
                    # Update scheduler even if validation is not performed (except ReduceLROnPlateau)
                    if self.scheduler is not None and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

            """if epoch == num_epochs - 1:
                self._set_last_val_results(**val_results)"""
            #self.exp_manager.log_results(**self.last_val_results)
            self.exp_manager.log_results(**val_results)
        
        # Processing at the end of training
        self.logger.info(f"Training completed after {num_epochs} epochs.")
