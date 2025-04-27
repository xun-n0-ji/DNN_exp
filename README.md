# AI learning process management framework

A framework for efficiently managing the AI ​​learning process. It organizes multiple tasks, models, and experiments hierarchically to ensure reproducibility and flexibility.

## Features

- Experiment management with a hierarchical directory structure
- Highly extensible design using abstract base classes
- Management of experiment parameters with configuration files
- Function for saving and comparing experiment results

## Directory structure

```
experiments/
├── task1/
│   ├── model1/
│   │   ├── exp001/
│   │   │   ├── config.yaml
│   │   │   ├── checkpoints/
│   │   │   ├── logs/
│   │   │   └── ...
│   │   ├── exp002/
│   │   │   └── ...
│   │   └── ...
│   ├── model2/
│   │   └── ...
│   └── ...
├── task2/
│   └── ...
└── ...
```

## Main components
- `Trainer`: Base class for the training process
- `Experiment`: Class for managing experiments
- `ConfigManager`: Class for managing configuration files
- Factory classes: Classes for creating loss functions, optimizers, and schedulers

## Usage

### Basic usage

1. Inherit the Trainer class to create a task-specific trainer
2. Create a configuration file
3. Run the experiment using run_experiment.py

```bash
python run_experiment.py --task task1 --model model1 --exp exp001 --trainer_class CustomTrainer --base_config examples/default_config.yaml
```

### Create a custom trainer

```python
from trainer import Trainer
import torch.nn as nn

class CustomTrainer(Trainer):
    def _init_model(self) -> nn.Module:
        # Implement model initialization
        ...

    def train_epoch(self, dataloader):
        # Implement training for one epoch
        ...

    def validate(self, dataloader):
        # Implement validation
        ...

    def create_dataloaders(self):
        # Implement data loader creation
        ...
```

### Example of config file

```yaml
# Model config
model:
  name: SimpleModel
  params:
    input_dim: 10
    hidden_dim: 128
    output_dim: 2

# Optimizer
optimizer:
  name: Adam
  params:
    lr: 0.001
    weight_decay: 0.0001

# Learning rate scheduler
scheduler:
  name: ReduceLROnPlateau
  params:
    mode: min
    factor: 0.1
    patience: 5
```

## Project Configuration

```
.
├── config_utils.py   # Config management utilities
├── experiment.py     # Experiment management classes
├── models/           # Model definitions
├── run_experiment.py # Experiment execution script
├── trainer.py        # Training process base class
├── trainers/         # Task-specific trainers
└── utils/            # Utility functions
```

## 拡張方法

1. If you are adding a new trainer:
   - Create a new trainer class in the trainers directory
   - Inherit the Trainer class and implement the required methods

2. If you are adding a new model:
   - Create a new model class in the models directory
   - It is useful to implement the from_config static method

3. If you are adding a new loss function or optimizer:
   - Add to utils/criterion_factory.py or utils/optimizer_factory.py

## Requirements

- Python 3.6+
- PyTorch 1.0+
- NumPy
- Matplotlib
- PyYAML 