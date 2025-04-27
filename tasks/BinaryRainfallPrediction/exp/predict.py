from pathlib import Path
from typing import Dict, List, Any, Tuple
from importlib import import_module
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
import polars as pl
from core import Predictor

# Experiment config
exp_no = '00001'
run_no = '00001'
exp_dir = Path(__file__).parent / f'exp{exp_no}' / f'run{run_no}'
config_filepath = exp_dir / 'config.yaml'
checkpoint_path = exp_dir / 'checkpoints' / 'checkpoint_best.pth'
predict_data_path = Path(__file__).parents[1] / 'data' / 'test.csv'

class RainfallPredDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if not Path(csv_file).exists:
            raise FileNotFoundError(f"{csv_file} does not exist.")
            
        self.df = pl.read_csv(csv_file)
        self.transform = transform
        
        self.ids = self.df['id'].to_numpy()
        
        # Extract features
        self.features = self.df.drop(['id', 'day']).to_numpy()
        
        self.input_dim = self.features.shape[1]
        
        # Scaling features
        if self.transform:
            # Confirm if "transform" is "fit_transform"
            if hasattr(self.transform, 'mean_'):
                self.features = self.transform.transform(self.features)
            else:
                self.features = self.transform.fit_transform(self.features)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        id_value = int(self.ids[idx])
        
        return features, id_value

class RainfallPredictor(Predictor):
    def _init_model(self) -> torch.nn.Module:
        model_config = self.config
        module = import_module(model_config['model']['module'])
        model_name = model_config['model']['name']
        model_class = getattr(module, model_name)
        
        # Call the appropriate method depending on how you created the model
        if hasattr(model_class, 'from_config'):
            return model_class.from_config(model_config)
        else:
            # Obtaining parameters manually
            model_params = model_config['model'].get('params', {})
            return model_class(**model_params)
    
    def predict(self, dataloader):
        self.model.eval()
        all_outputs = []
        all_ids = []
        
        self.logger.info("Starting prediction...")
        
        with torch.no_grad():
            for inputs, ids in dataloader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                
                all_outputs.append(outputs.cpu())
                all_ids.extend(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        
        all_outputs = torch.cat(all_outputs, dim=0)
        
        predicted = all_outputs.float().cpu().numpy().flatten()
        
        self.logger.info(f"Prediction complete. Generated {len(predicted)} predictions.")
        
        results = {
            'id': all_ids,
            'rainfall': predicted
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], model_name = ''):
        # Convert predictions to DataFrame
        submission = pl.DataFrame({
            k :v for k, v in results.items()
        })
        
        # Create output directory
        output_dir = exp_dir / 'prediction'
        output_dir.mkdir(exist_ok=True)
        
        if not model_name:
            model_name = self.config['model'].get('name', 'model')
        
        # Save
        submission_path = output_dir / f'prediction_exp{exp_no}_run{run_no}_model-{model_name}.csv'
        submission.write_csv(submission_path, include_header=True)
        self.logger.info(f"Submission file saved to {submission_path}")
        
        return submission_path

def create_dataset_and_loader(file_path: str, config: Dict[str, Any]):
    # Create Scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Create dataset
    dataset = RainfallPredDataset(file_path, transform=scaler)
    
    # Get configs of dataloader
    batch_size = config.get('dataset', {}).get('batch_size', 16)
    num_workers = config.get('dataset', {}).get('num_workers', 2)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return dataset, dataloader

if __name__ == '__main__':
    with open(config_filepath, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    if not predict_data_path.exists:
        raise FileNotFoundError(f"{predict_data_path} does not exist.")
    
    print(f"Configuration: {config_filepath}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Predictions: {predict_data_path}")
    
    # Initialize predictor
    predictor = RainfallPredictor(config, str(exp_dir), str(checkpoint_path))
    
    # Create dataset & dataloader
    _, predict_loader = create_dataset_and_loader(predict_data_path, config)
    
    results = predictor.predict(predict_loader)
    predictor.save_results(results)
    print(f"Predicted: {len(results[list(results.keys())[0]])} predictions were done.")
    