"""
Data loading utilities for Housing Price Prediction V4
"""

import pandas as pd
import numpy as np
from pathlib import Path

class HousingDataLoader:
    def __init__(self, config):
        self.config = config
        self.train_path = config.DATA_PATHS['train']
        self.test_path = config.DATA_PATHS['test']
    
    def load_data(self):
        """Load training and test datasets"""
        print("Loading data...")
        
        # Check if files exist
        if not Path(self.train_path).exists():
            raise FileNotFoundError(f"Training data not found: {self.train_path}")
        if not Path(self.test_path).exists():
            raise FileNotFoundError(f"Test data not found: {self.test_path}")
        
        # Load data
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)
        
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        
        return train_data, test_data
    
    def get_feature_types(self, df):
        """Categorize features by data type"""
        # Drop ID and target if present
        features_df = df.drop(columns=['Id'] + (['SalePrice'] if 'SalePrice' in df.columns else []))
        
        numeric_features = features_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = features_df.select_dtypes(include=['object']).columns.tolist()
        bool_features = features_df.select_dtypes(include=['bool']).columns.tolist()
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'boolean': bool_features
        }
