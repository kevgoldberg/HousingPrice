"""
Configuration management for Housing Price Prediction V4
"""

import yaml
from pathlib import Path
import torch

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.setup_device()
    
    def load_config(self):
        """Load configuration from YAML file"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self.get_default_config()
            self.save_default_config()
        
        # Set attributes
        for key, value in config.items():
            setattr(self, key, value)
    
    def setup_device(self):
        """Setup computing device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def get_default_config(self):
        """Default configuration"""
        return {
            'SEED': 42,
            'DATA_PATHS': {
                'train': '../house-prices-advanced-regression-techniques/train.csv',
                'test': '../house-prices-advanced-regression-techniques/test.csv'
            },
            'MODEL': {
                'hidden_sizes': [128, 64],
                'dropout_rate': 0.2,
                'learning_rate': 1e-2,
                'weight_decay': 1e-4,
                'batch_size': 128,
                'epochs': 200,
                'patience': 10
            },
            'PREPROCESSING': {
                'val_ratio': 0.20,
                'fill_numeric_with': 'mean',
                'fill_categorical_with': 'Missing',
                'scaling_method': 'standard'
            },
            'CV': {
                'k_folds': 5,
                'cv_epochs': 100,
                'cv_batch_size': 64
            }
        }
    
    def save_default_config(self):
        """Save default configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.get_default_config(), f, default_flow_style=False)
