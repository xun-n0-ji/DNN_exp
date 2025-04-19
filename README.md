# AI学習プロセス管理フレームワーク

AI学習プロセスを効率的に管理するためのフレームワークです。複数のタスク、モデル、実験を階層的に整理し、再現性と柔軟性を確保します。

## 特徴

- 階層的なディレクトリ構造による実験管理
- 抽象基底クラスを使用した拡張性の高い設計
- 設定ファイルによる実験パラメータの管理
- 実験結果の保存と比較機能

## ディレクトリ構造

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

## 主要コンポーネント

- `Trainer`: 学習プロセスの基本クラス
- `Experiment`: 実験を管理するクラス
- `ConfigManager`: 設定ファイルを管理するクラス
- ファクトリクラス群: 損失関数、オプティマイザ、スケジューラを作成するクラス

## 使用方法

### 基本的な使い方

1. Trainerクラスを継承して、タスク固有のトレーナーを作成
2. 設定ファイルを作成
3. run_experiment.pyを使用して実験を実行

```bash
python run_experiment.py --task task1 --model model1 --exp exp001 --trainer_class CustomTrainer --base_config examples/default_config.yaml
```

### カスタムトレーナーの作成

```python
from trainer import Trainer
import torch.nn as nn

class CustomTrainer(Trainer):
    def _init_model(self) -> nn.Module:
        # モデルの初期化処理を実装
        ...
    
    def train_epoch(self, dataloader):
        # 1エポックの学習処理を実装
        ...
    
    def validate(self, dataloader):
        # 検証処理を実装
        ...
    
    def create_dataloaders(self):
        # データローダーの作成処理を実装
        ...
```

### 設定ファイルの例

```yaml
# モデル設定
model:
  name: SimpleModel
  params:
    input_dim: 10
    hidden_dim: 128
    output_dim: 2

# オプティマイザ
optimizer:
  name: Adam
  params:
    lr: 0.001
    weight_decay: 0.0001

# 学習率スケジューラ
scheduler:
  name: ReduceLROnPlateau
  params:
    mode: min
    factor: 0.1
    patience: 5
```

## プロジェクト構成

```
.
├── config_utils.py    # 設定管理ユーティリティ
├── experiment.py      # 実験管理クラス
├── models/            # モデル定義
├── run_experiment.py  # 実験実行スクリプト
├── trainer.py         # 学習プロセス基底クラス
├── trainers/          # タスク固有のトレーナー
└── utils/             # ユーティリティ関数
```

## 拡張方法

1. 新しいトレーナーを追加する場合:
   - trainersディレクトリに新しいトレーナークラスを作成
   - Trainerクラスを継承して必要なメソッドを実装

2. 新しいモデルを追加する場合:
   - modelsディレクトリに新しいモデルクラスを作成
   - from_config静的メソッドを実装すると便利

3. 新しい損失関数やオプティマイザを追加する場合:
   - utils/criterion_factory.py または utils/optimizer_factory.py に追加

## 要件

- Python 3.6+
- PyTorch 1.0+
- NumPy
- Matplotlib
- PyYAML 