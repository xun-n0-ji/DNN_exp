from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any, Tuple
import functools

import mlflow

class MetricsLogger(ABC):
    def __init__(self):
        pass

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log the model parameters.
        
        Args:
            params: A dictionary of parameters {parameter name: value}
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, mode:str) -> None:
        """
        Logs evaluation metrics.
        
        Args:
            metrics: A dictionary of metrics {metric name: value}
            step: Number of steps (optional)
        """
        for key, value in metrics.items():
            mlflow.log_metric(f'{mode}_{key}', value, step=step)

    
    def log_model(self, model, model_name: str = "model", 
                  signature=None, input_example=None) -> None:
        """
        Save the model to MLflow.

        Args:
            model: The model to save
            model_name: Name of the model
            signature: Signature of the model (input/output schema)
            input_example: Example input
        """
        # Select the logging method according to the model framework
        if hasattr(model, "predict") and hasattr(model, "fit"):
            # In the case of a scikit-learn model like
            mlflow.sklearn.log_model(
                model, model_name, signature=signature, input_example=input_example
            )
        else:
            # Save your model in a generic way
            mlflow.pyfunc.log_model(
                model_name, python_model=model, 
                signature=signature, input_example=input_example
            )
        
        print(f"Logged model as: {model_name}")

    @abstractmethod
    def log_results(self, **kwargs):
        raise NotImplementedError('Subclasses must implement "log_results".')
    
    def loggable(func):
        """A decorator that indicates which functions should be logged."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        
        # Marking functions
        wrapper._loggable = True
        return wrapper