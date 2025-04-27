from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
import inspect

import mlflow
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from core import MetricsLogger

class ClassificationMetricsLogger(MetricsLogger):
    def __init__(self):
        pass

    def log_results(self, **kwargs) -> None:
        """Executes a function in a class that has a loggable decorator and logs the results"""
        # Get all methods in a class
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for method_name, method in methods:

            if method_name in ['__init__', 'log_results']:
                continue
                
            # Process only functions marked with the decorator
            if hasattr(method, '_loggable') and method._loggable:
                # Get the parameter names of a function
                sig = inspect.signature(method)
                params = sig.parameters
                
                param_names = [p for p in params.keys() if p != 'self']
                # Extract only the parameters required for a function
                filtered_args = {k: kwargs[k] for k in param_names if k in kwargs}
                
                # Run the function and save the result
                try:
                    method(**filtered_args)
                    print(f"Successfully executed {method_name} with args: {list(filtered_args.keys())}")
                except Exception as e:
                    print(f"Failed to execute {method_name}: {e}")

    @MetricsLogger.loggable
    def _log_confusion_matrix(self, targets: np.ndarray, preds: np.ndarray, 
                            labels: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (10, 8)):
        """
        Generate and save the confusion matrix.

        Args:
            targets: actual labels
            preds: predicted labels
            labels: class labels (optional)
            figsize: figure size
        """
        # Calculate the confusion matrix
        cm = confusion_matrix(targets, preds)
        
        # Create a plot
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        if labels:
            plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45)
            plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
        
        # Saving an image
        cm_path = Path('confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(str(cm_path))
        plt.close()
        
        # Register as an artifact in MLflow
        mlflow.log_artifact(str(cm_path))
        cm_path.unlink()

    @MetricsLogger.loggable
    def _log_roc_curve(self, targets: np.ndarray, preds: np.ndarray,
                     figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Generates and saves the ROC curve (for binary classification).

        Args:
            targets: actual label
            preds: predicted score (probability)
            figsize: figure size
        """
        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(targets, preds)
        roc_auc = auc(fpr, tpr)
        
        # Create a plot
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        
        # Save an image
        roc_path = Path('roc_curve.png')
        plt.tight_layout()
        plt.savefig(str(roc_path))
        plt.close()
        
        # Register as an artifact in MLflow
        mlflow.log_artifact(str(roc_path))
        roc_path.unlink()

    # Needs to be implemented
    #@MetricsLogger.loggable
    def log_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str], 
                              figsize: Tuple[int, int] = (12, 10),
                              top_n: Optional[int] = None) -> None:
        """
        Visualize and save feature importance.

        Args:
            feature_importance: List of feature importances
            feature_names: List of feature names
            figsize: Figure size
            top_n: Number of top features to display (optional)
        """
        # Sort feature importance
        indices = np.argsort(feature_importance)
        
        if top_n is not None and top_n < len(feature_names):
            indices = indices[-top_n:]
            
        # Create a plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Ranking')
        
        # Save an image
        fi_path = Path('feature_importance.png')
        plt.tight_layout()
        plt.savefig(str(fi_path))
        plt.close()
        
        # Register as an artifact in MLflow
        mlflow.log_artifact(str(fi_path))
        fi_path.unlink()
    
    @MetricsLogger.loggable
    def log_as_dataframes(self, targets: np.ndarray, preds: np.ndarray) -> None:
        """
        Save the DataFrame.

        Args:
            targets: target data to save
            preds: target data to save
        """
        # Convert ndarray to polars.DataFrame
        df = pl.DataFrame({
            "target": targets.flatten(),
            "prediction": preds.flatten()
        })
        
        file_name = 'pred_and_targets.csv'
        df.write_csv(file_name)
        
        # Upload files to mlflow
        mlflow.log_artifact(file_name)

        Path(file_name).unlink()
