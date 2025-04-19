# 降雨予測（Binary Rainfall Prediction）

降雨の有無を予測する二値分類モデルの実装です。

## セットアップ

### 依存ライブラリのインストール

uvを使用して依存ライブラリをインストールします：

```bash
uv pip install -r tasks/BinaryRainfallPrediction/requirements.txt
```

## 使用方法

### 基本的な実行

以下のコマンドで実験を実行できます：

```bash
python tasks/BinaryRainfallPrediction/run_rainfall.py
```

### パラメータの調整

コマンドライン引数を使用して、様々な設定を調整できます：

```bash
python tasks/BinaryRainfallPrediction/run_rainfall.py --epochs 20 --batch_size 32 --learning_rate 0.001
```

### 主なオプション

| オプション | 説明 | デフォルト値 |
|------------|------|------------|
| `--exp_id` | 実験ID | 自動生成（日時） |
| `--model_name` | モデル名 | model1 |
| `--epochs` | エポック数 | 50 |
| `--batch_size` | バッチサイズ | 16 |
| `--learning_rate` | 学習率 | 0.001 |
| `--hidden_dims` | 隠れ層の次元数（カンマ区切り） | "32,16" |
| `--dropout_rate` | ドロップアウト率 | 0.2 |
| `--seed` | 乱数シード | 42 |
| `--no_scaling` | 特徴量のスケーリングを無効化 | False |
| `--gpu` | GPUを使用（利用可能な場合） | False |

## 出力

実験結果は以下のディレクトリに保存されます：
`tasks/BinaryRainfallPrediction/output/{model_name}/{exp_id}/`

出力ファイル：
- `config.yaml`: 実験の設定
- `best_model.pth`: 検証ロスが最小のモデル
- `final_model.pth`: 最終エポックのモデル
- `model_checkpoint.pth`: 全体の状態を含むチェックポイント
- `training_history.csv`: トレーニング履歴
- `evaluation_results.json`: 評価指標
- `learning_curves.png`: 学習曲線のグラフ
- `probability_distribution.png`: 予測確率の分布

## データセット

データセットは `tasks/BinaryRainfallPrediction/data/train.csv` にあります。

特徴量：
- pressure: 気圧
- maxtemp: 最高気温
- temparature: 気温
- mintemp: 最低気温
- dewpoint: 露点温度
- humidity: 湿度
- cloud: 雲量
- sunshine: 日照時間
- winddirection: 風向
- windspeed: 風速

ターゲット：
- rainfall: 降雨の有無（1: あり、0: なし） 