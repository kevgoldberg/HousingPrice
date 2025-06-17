"""
Data preprocessing utilities for Housing Price Prediction V4
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

class HousingPreprocessor:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        self.feature_columns = None
    
    def preprocess(self, train_data, test_data):
        """Complete preprocessing pipeline"""
        print("Starting preprocessing...")
        
        # Separate features and target
        X_train_raw = train_data.drop(columns=['Id', 'SalePrice'])
        X_test_raw = test_data.drop(columns=['Id'])
        y_train = train_data['SalePrice']
        
        # Get feature types
        feature_types = self._get_feature_types(X_train_raw)
        
        # Handle missing values
        X_train_clean, X_test_clean = self._handle_missing_values(
            X_train_raw, X_test_raw, feature_types
        )
        
        # Encode categorical variables
        X_train_encoded, X_test_encoded = self._encode_categorical(
            X_train_clean, X_test_clean, feature_types['categorical']
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self._scale_features(
            X_train_encoded, X_test_encoded
        )
        
        # Transform target (log1p)
        y_train_transformed = np.log1p(y_train.values).reshape(-1, 1)
        
        print(f"Final feature shape: {X_train_scaled.shape}")
        print("Preprocessing completed.")
        
        return X_train_scaled, X_test_scaled, y_train_transformed
    
    def _get_feature_types(self, df):
        """Categorize features by data type"""
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        bool_features = df.select_dtypes(include=['bool']).columns.tolist()
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'boolean': bool_features
        }
    
    def _handle_missing_values(self, X_train, X_test, feature_types):
        """Handle missing values for different feature types"""
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        
        # Handle numeric features
        if feature_types['numeric']:
            if self.config.PREPROCESSING['fill_numeric_with'] == 'mean':
                numeric_means = X_train_clean[feature_types['numeric']].mean()
                X_train_clean[feature_types['numeric']] = X_train_clean[feature_types['numeric']].fillna(numeric_means)
                X_test_clean[feature_types['numeric']] = X_test_clean[feature_types['numeric']].fillna(numeric_means)
        
        # Handle categorical features
        if feature_types['categorical']:
            fill_value = self.config.PREPROCESSING['fill_categorical_with']
            X_train_clean[feature_types['categorical']] = X_train_clean[feature_types['categorical']].fillna(fill_value)
            X_test_clean[feature_types['categorical']] = X_test_clean[feature_types['categorical']].fillna(fill_value)
        
        return X_train_clean, X_test_clean
    
    def _encode_categorical(self, X_train, X_test, categorical_features):
        """One-hot encode categorical variables"""
        if not categorical_features:
            return X_train, X_test
        
        # Combine for consistent encoding
        X_combined = pd.concat([X_train, X_test], axis=0)
        X_combined_encoded = pd.get_dummies(X_combined, columns=categorical_features, drop_first=True)
        
        # Split back
        n_train = X_train.shape[0]
        X_train_encoded = X_combined_encoded.iloc[:n_train, :].copy()
        X_test_encoded = X_combined_encoded.iloc[n_train:, :].copy()
        
        # Align test columns with train
        X_test_aligned = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        
        self.feature_columns = X_train_encoded.columns
        return X_train_encoded, X_test_aligned
    
    def _scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        # Convert to float32 for memory efficiency
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        return X_train_scaled, X_test_scaled
