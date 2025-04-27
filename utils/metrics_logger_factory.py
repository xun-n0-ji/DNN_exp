import importlib
from pathlib import Path
import os

class MetricsLoggerFactory:
    @staticmethod
    def get_metrics_logger_class(name: str):
        """
        Returns the corresponding MetricsLogger class according to the name.
        e.g., "ClassificationMetricsLogger" -> utils.metric_logger.classification_metrics_logger.ClassificationMetricsLogger
        """
        # Explore the script (.py file)
        base_path = Path(__file__).parent / 'metrics_logger'
        files = [f for f in os.listdir(base_path) if f.endswith(".py") and not f.startswith("__")]

        root_path = Path(__file__).parents[1]
        relative_base = base_path.relative_to(root_path)
        base_module_path = str(relative_base).replace(os.sep, ".")

        # Find classes in all .py files
        for f in files:
            module_name = f"{base_module_path}.{f[:-3]}"  # Remove .py from filename
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, name, None)
                if cls is not None:
                    return cls
            except Exception as e:
                print(f"Warning: could not import {module_name}: {e}")
                continue

        raise ValueError(f"Unknown MetricsLogger: {name}")