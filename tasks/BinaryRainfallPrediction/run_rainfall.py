import os
import argparse
import yaml
import torch
import polars as pl
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler

# デフォルト設定
DEFAULT_CONFIG = {
    "model": {
        "input_dim": 13,
        "hidden_dims": [32, 16],
        "output_dim": 1,
        "dropout_rate": 0.2
    },
    "training": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001
    },
    "data": {
        "path": "tasks/BinaryRainfallPrediction/data/train.csv",
        "train_ratio": 0.8
    },
    "output": {
        "dir": "tasks/BinaryRainfallPrediction/output"
    }
}

class RainfallDataset(Dataset):
    """
    降雨予測のためのデータセット
    """
    def __init__(self, csv_file, transform=None):
        self.df = pl.read_csv(csv_file)
        self.transform = transform
        
        # 日付を除外して特徴量とターゲットを抽出
        # カラム名を確認して、正しく特徴量とターゲットを分離
        self.features = self.df.drop(['id', 'day', 'rainfall']).to_numpy()
        self.targets = self.df['rainfall'].to_numpy()
        
        # 特徴量のスケーリング
        if self.transform:
            # transform が fit_transform されているか確認
            if hasattr(self.transform, 'mean_'):
                self.features = self.transform.transform(self.features)
            else:
                self.features = self.transform.fit_transform(self.features)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # 特徴量とターゲットをテンソルに変換
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32).view(1)
        
        return features, target

class RainfallModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate):
        super(RainfallModel, self).__init__()
        
        layers = []
        
        # 入力層から最初の隠れ層
        layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # 隠れ層
        for i in range(len(hidden_dims) - 1):
            layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
        
        # 出力層
        layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        layers.append(torch.nn.Sigmoid())
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train(model, train_loader, val_loader, config, device, output_dir):
    """
    モデルの学習を行います
    """
    # ロス関数とオプティマイザの設定
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # 学習履歴の記録用
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # 最良モデルの保存用
    best_val_loss = float('inf')
    
    print(f"学習を開始します...")
    for epoch in range(config["training"]["epochs"]):
        # 学習モード
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 勾配をゼロにリセット
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 逆伝播と最適化
            loss.backward()
            optimizer.step()
            
            # 統計情報の更新
            train_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)
        
        # 評価モード
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 順伝播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 統計情報の更新
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        # エポックごとの統計
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # 履歴の更新
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"Epoch {epoch+1}: 最良のモデルを保存しました")
    
    # 最終モデルの保存
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # 学習履歴をCSVファイルに保存
    history_df = pl.DataFrame(history)
    history_df.write_csv(os.path.join(output_dir, 'training_history.csv'))
    
    return history

def evaluate(model, test_loader, device):
    """
    モデルの評価を行う関数
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 予測を実行
            outputs = model(inputs)
            probs = outputs.squeeze().cpu().numpy()  # スクイーズして1次元に
            preds = (outputs > 0.5).float()

            # 正解数をカウント
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            # 結果を保存する
            all_preds.extend(preds.cpu().numpy().flatten())  # フラット化して追加
            all_probs.extend(probs.flatten() if isinstance(probs, np.ndarray) else [probs])  # 単一値の場合を処理
            all_targets.extend(targets.cpu().numpy().flatten())  # フラット化して追加

    # 評価指標を計算
    accuracy = correct / total
    
    # 混同行列からの指標計算
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    
    # 精度、再現率、F1スコア、特異度を計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity
    }
    
    return results, all_probs

def parse_args():
    """
    コマンドライン引数の解析
    """
    parser = argparse.ArgumentParser(description='雨量予測実験を実行する')
    
    # 実験管理の引数
    parser.add_argument('--exp_id', type=str, default=None, help='実験ID')
    parser.add_argument('--model_name', type=str, default='model1', help='モデル名')
    parser.add_argument('--output_dir', type=str, default=None, help='出力ディレクトリのパス')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    
    # データ関連の引数
    parser.add_argument('--data_path', type=str, default=None, help='データセットのパス')
    parser.add_argument('--train_ratio', type=float, default=None, help='訓練データの割合')
    parser.add_argument('--no_scaling', action='store_true', help='特徴量のスケーリングを無効化')
    
    # モデル構造の引数
    parser.add_argument('--input_dim', type=int, default=None, help='入力次元数')
    parser.add_argument('--hidden_dims', type=str, default=None, help='隠れ層の次元数（カンマ区切り、例: "32,16"）')
    parser.add_argument('--dropout_rate', type=float, default=None, help='ドロップアウト率')
    
    # 訓練関連の引数
    parser.add_argument('--epochs', type=int, default=None, help='エポック数')
    parser.add_argument('--batch_size', type=int, default=None, help='バッチサイズ')
    parser.add_argument('--learning_rate', type=float, default=None, help='学習率')
    parser.add_argument('--weight_decay', type=float, default=None, help='重み減衰パラメータ')
    parser.add_argument('--no_early_stopping', action='store_true', help='早期停止を無効化')
    
    # GPUの設定
    parser.add_argument('--gpu', action='store_true', help='GPUを使用する（利用可能な場合）')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用するGPUのID')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # タスク名（固定）
    task_name = "BinaryRainfallPrediction"
    
    # 実験IDの生成（指定がない場合）
    if args.exp_id is None:
        exp_id = f"exp{datetime.now().strftime('%Y%m%d%H%M%S')}"
    else:
        exp_id = args.exp_id
    
    # 設定の読み込み
    config = DEFAULT_CONFIG.copy()
    
    # コマンドライン引数で設定を上書き
    # 訓練関連の設定
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.weight_decay:
        config["training"]["weight_decay"] = args.weight_decay
    
    # モデル関連の設定
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.input_dim:
        config["model"]["input_dim"] = args.input_dim
    if args.hidden_dims:
        config["model"]["hidden_dims"] = [int(dim) for dim in args.hidden_dims.split(',')]
    if args.dropout_rate:
        config["model"]["dropout_rate"] = args.dropout_rate
    
    # データ関連の設定
    if args.data_path:
        config["data"]["path"] = args.data_path
    if args.train_ratio:
        config["data"]["train_ratio"] = args.train_ratio
    
    # 出力ディレクトリの設定
    if args.output_dir:
        config["output"]["dir"] = args.output_dir
    output_dir = os.path.join(config["output"]["dir"], config["model"]["name"], exp_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # ランダムシードの設定
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"実験を開始します: {task_name}/{config['model']['name']}/{exp_id}")
    print(f"ランダムシード: {seed}")
    
    # 設定ファイルの保存
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # デバイスの設定
    if args.gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用デバイス: {device}")
    
    # データの準備
    print("データの読み込みと前処理を行っています...")
    scaler = None if args.no_scaling else StandardScaler()
    full_dataset = RainfallDataset(config["data"]["path"], transform=scaler)
    
    # 入力次元を自動検出
    if config["model"]["input_dim"] is None or config["model"]["input_dim"] != full_dataset.features.shape[1]:
        config["model"]["input_dim"] = full_dataset.features.shape[1]
        print(f"入力次元数を自動検出しました: {config['model']['input_dim']}")
    
    # データセットの分割
    # 再現性のためにシードを設定
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 訓練:検証:テストの比率を設定 (例: 70:15:15)
    train_ratio = config["data"]["train_ratio"]
    val_ratio = (1 - train_ratio) / 2
    test_ratio = (1 - train_ratio) / 2
    
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    print(f"データセットの分割: 訓練={train_size}, 検証={val_size}, テスト={test_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=2
    )
    
    # モデルの初期化
    print("モデルを初期化しています...")
    input_dim = config["model"]["input_dim"]
    print(f"使用する入力次元数: {input_dim}")
    
    model = RainfallModel(
        input_dim=config["model"]["input_dim"],
        hidden_dims=config["model"]["hidden_dims"],
        output_dim=config["model"]["output_dim"],
        dropout_rate=config["model"]["dropout_rate"]
    ).to(device)
    
    # モデルの学習
    print("モデルの学習を行います...")
    history = train(model, train_loader, val_loader, config, device, output_dir)
    
    # 学習曲線の描画と保存を行う
    try:
        import matplotlib.pyplot as plt
        
        # 損失のプロット
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 精度のプロット
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
        print("学習曲線を保存しました")
    except Exception as e:
        print(f"学習曲線の作成中にエラーが発生しました: {e}")
    
    # 最良モデルを読み込む
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    
    # モデルの評価
    print("モデルの評価を行います...")
    results, all_probs = evaluate(model, test_loader, device)
    
    # 評価結果をJSONファイルに保存
    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # 予測確率分布のヒストグラムを作成
    try:
        plt.figure(figsize=(8, 6))
        plt.hist(all_probs, bins=20, alpha=0.7)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
        print("予測確率分布を保存しました")
    except Exception as e:
        print(f"確率分布の作成中にエラーが発生しました: {e}")
    
    # モデルの状態を保存
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'evaluation_results': results
    }, os.path.join(output_dir, 'model_checkpoint.pth'))
    
    print(f"実験が終了しました。結果は {output_dir} に保存されています。")
    return results

if __name__ == '__main__':
    main() 