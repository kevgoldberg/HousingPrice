"""
Model evaluation utilities for Housing Price Prediction V4
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from .model import HousingPriceModel

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.device = config.device
    
    def evaluate_model(self, model, X_train, y_train):
        """Evaluate model performance"""
        print("Evaluating model...")
        
        model.eval()
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            log_predictions = model(X_tensor).cpu().numpy().flatten()
            log_targets = y_tensor.cpu().numpy().flatten()
            
            # Convert back to original scale
            predictions = np.expm1(log_predictions)
            targets = np.expm1(log_targets)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        log_rmse = np.sqrt(mean_squared_error(log_targets, log_predictions))
        
        print(f"Training Performance:")
        print(f"  RMSE (original scale): ${rmse:,.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Log-RMSE (Kaggle metric): {log_rmse:.5f}")
        
        # Plot predictions vs actual
        self._plot_predictions(targets, predictions)
        
        return {
            'rmse': rmse,
            'r2': r2,
            'log_rmse': log_rmse
        }
    
    def cross_validate(self, X_train, y_train):
        """Perform k-fold cross-validation"""
        print("Performing cross-validation...")
        
        kf = KFold(
            n_splits=self.config.CV['k_folds'],
            shuffle=True,
            random_state=self.config.SEED
        )
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"→ Fold {fold + 1}/{self.config.CV['k_folds']}")
            
            # Split data
            X_tr = torch.tensor(X_train[train_idx], dtype=torch.float32).to(self.device)
            y_tr = torch.tensor(y_train[train_idx], dtype=torch.float32).to(self.device)
            X_val = torch.tensor(X_train[val_idx], dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_train[val_idx], dtype=torch.float32).to(self.device)
            
            # Create model and trainer for this fold
            fold_model = HousingPriceModel(X_train.shape[1], self.config).to(self.device)
            fold_optimizer = torch.optim.AdamW(
                fold_model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
            criterion = nn.MSELoss()
            
            # Create dataloader
            train_dataset = TensorDataset(X_tr, y_tr)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.CV['cv_batch_size'],
                shuffle=True
            )
            
            # Train for specified epochs
            for epoch in range(self.config.CV['cv_epochs']):
                fold_model.train()
                for X_batch, y_batch in train_loader:
                    fold_optimizer.zero_grad()
                    predictions = fold_model(X_batch)
                    loss = criterion(predictions, y_batch)
                    loss.backward()
                    fold_optimizer.step()
            
            # Evaluate on validation set
            fold_model.eval()
            with torch.no_grad():
                val_predictions = fold_model(X_val).cpu().numpy().flatten()
                val_targets = y_val.cpu().numpy().flatten()
            
            # Calculate log-RMSE
            fold_score = np.sqrt(mean_squared_error(val_targets, val_predictions))
            cv_scores.append(fold_score)
            print(f"Fold {fold + 1} log-RMSE: {fold_score:.5f}")
        
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        
        print(f"\nCV Results:")
        print(f"  Mean log-RMSE: {mean_score:.5f}")
        print(f"  Std log-RMSE: {std_score:.5f}")
        
        return cv_scores
    
    def _plot_predictions(self, targets, predictions):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(targets, predictions, alpha=0.5)
        
        # Perfect prediction line
        min_val, max_val = targets.min(), targets.max()
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Actual SalePrice')
        plt.ylabel('Predicted SalePrice')
        plt.title('Actual vs Predicted SalePrice')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.show()
