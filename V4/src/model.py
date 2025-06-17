"""
Neural network model and training utilities for Housing Price Prediction V4
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from pathlib import Path

class HousingPriceModel(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.config = config
        
        hidden_sizes = config.MODEL['hidden_sizes']
        dropout_rate = config.MODEL['dropout_rate']
        
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Remove last dropout and add output layer
        layers = layers[:-1]  # Remove last dropout
        layers.append(nn.Linear(prev_size, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ModelTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.MODEL['learning_rate'],
            weight_decay=config.MODEL['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        self.criterion = nn.MSELoss()
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("Starting training...")
        
        # Convert to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        # Create dataset and split
        full_dataset = TensorDataset(X_tensor, y_tensor)
        val_ratio = self.config.PREPROCESSING['val_ratio']
        n_val = int(len(full_dataset) * val_ratio)
        n_train = len(full_dataset) - n_val
        
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.config.SEED)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.MODEL['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.MODEL['batch_size'],
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(1, self.config.MODEL['epochs'] + 1):
            # Train
            train_loss = self._train_epoch(train_loader)
            
            # Validate
            val_loss = self._validate_epoch(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.MODEL['patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch == 1 or epoch % 20 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Load best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        print(f"Training completed. Best validation loss: {best_val_loss:.5f}")
        return self.model
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, X_test):
        """Make predictions on test data"""
        self.model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions
    
    def save_model(self, path):
        """Save model state"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        return self.model
