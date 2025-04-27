from pathlib import Path
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any

import torch
from torch.serialization import add_safe_globals

# Add NumPy's scalar into the safe list (for Pytorch > 2.6)
import numpy as np
import numpy._core.multiarray as multiarray
add_safe_globals([multiarray.scalar, np.dtype, np.dtypes.Float64DType])

class Predictor(ABC):
    """
    The base class for the evaluation process.
    All task-specific evaluators inherit from this class.
    """
    def __init__(self, config: Dict[str, Any], exp_dir: str, checkpoint_path: str = None):
        """
        Initializing the evaluator
        
        Args:
            config: A dictionary containing configuration parameters
            exp_dir: Directory path to save the experiment results
            checkpoint_path: Checkpoint pass of the model to evaluate (if None, use best model)
        """
        self.config = config
        self.exp_dir = exp_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configuring Logging
        self.logger = self._setup_logger()
        
        # Initializing the model
        self.model = self._init_model()
        self.model.to(self.device)
        
        # Loading a checkpoint
        if checkpoint_path is None:
            # Default is to use the best model
            checkpoint_path = Path(self.exp_dir) / 'checkpoints' / 'checkpoint_best.pth'
        
        if Path(checkpoint_path).exists:
            self.load_checkpoint(checkpoint_path)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            self.logger.warning(f"Checkpoint {checkpoint_path} not found. Using initialized model.")
    
    def _setup_logger(self) -> logging.Logger:
        """Logger configuration"""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path(self.exp_dir) / 'prediction'
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'eval.log')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    @abstractmethod
    def _init_model(self) -> torch.nn.Module:
        """
        Method to initialize a model.

        Returns:
            The initialized model.
        """
        raise NotImplementedError('Need to implement "_init_model" in subclass')
    
    def load_checkpoint(self, checkpoint_path: str, resume_training: bool = False):
        """
        The method to read the checkpoint.
        
        Args:
            checkpoint_path: Checkpoint file path
            
        Returns:
            Loaded checkpoint information
        """
        if not Path(checkpoint_path).exists:
            self.logger.error(f"Could not find {checkpoint_path}.")
            raise FileNotFoundError(f"Could not find {checkpoint_path}.")
        
        # If training is resumed, read all information, otherwise only weights
        checkpoint = torch.load(
            checkpoint_path, 
            map_location=self.device,
            weights_only=not resume_training
        )
        
        # Model weights are always loaded
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # If training is resumed, also read the optimizer and scheduler states.
        if resume_training and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint
    
    @abstractmethod
    def predict(self, dataloader):
        """
        The method for evaluating the model.
        
        Args:
            dataloader: Dataloader for evaluation data
            
        Returns:
            Evaluation results dictionary
        """
        raise NotImplementedError('Need to implement "predict" in subclass')