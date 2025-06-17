"""
Utility functions for Housing Price Prediction V4
"""

import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def save_predictions(predictions, ids, filepath):
    """Save predictions to CSV file"""
    # Convert log predictions back to original scale
    predictions_original = np.expm1(predictions.flatten())
    
    submission = pd.DataFrame({
        'Id': ids,
        'SalePrice': predictions_original
    })
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    submission.to_csv(filepath, index=False)
    print(f"Predictions saved to {filepath}")

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'plots', 'submissions', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
