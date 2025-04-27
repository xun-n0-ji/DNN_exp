import mlflow
from typing import Dict, List, Union, Optional, Any, Tuple

from utils.metrics_logger_factory import MetricsLoggerFactory

class LoggerFuncCaller:
    def log_params(self, params: Dict[str, Any]) -> None:
        self.metrics_logger.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int, mode:str) -> None:
        self.metrics_logger.log_metrics(metrics, step, mode)

    def log_model(self, model, model_name: str = "model", 
                  signature=None, input_example=None) -> None:
        self.metrics_logger.log_model(model, model_name, signature, input_example)

    def log_results(self, **kwargs):
        self.metrics_logger.log_results(**kwargs)

class ExpManager(LoggerFuncCaller):
    """
    A utility class for managing MLflow experiments.
    It manages experiment names, parameters, metrics, etc., and also visualizes them.
    """
    
    def __init__(self, task_name: str, run_name: str, metrics_logger_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize MLflowManager.

        Args:
            task_name: MLflow experiment name
            tracking_uri: MLflow tracking server URI (default is local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.task_name = task_name
        self.run_name = run_name
        mlflow.set_experiment(task_name)
        self._setup_logger(metrics_logger_name)
        self.active_run = None
        self.run_id = None
        
    def _setup_logger(self, metrics_logger_name):
        MetricsLoggerClass = MetricsLoggerFactory.get_metrics_logger_class(metrics_logger_name)
        self.metrics_logger = MetricsLoggerClass()

    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Starts an MLflow run.

        Args:
            run_name: Name of the run (optional)
        """
        self.active_run = mlflow.start_run(run_name=run_name)
        self.run_id = self.active_run.info.run_id
    
    def end_run(self) -> None:
        """
        Terminates the current execution.
        """
        if self.active_run:
            mlflow.end_run()
            self.active_run = None
            self.run_id = None
    
    def __enter__(self):
        """
        Entry point for use as a context manager
        """
        self.start_run(self.run_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Finalization for use as a context manager
        """
        self.end_run()
