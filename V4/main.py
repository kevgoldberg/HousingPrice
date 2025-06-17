"""
Housing Price Prediction V4 - Main Script
Converts V3 Jupyter notebook into a structured Python application
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.config import Config
from src.data_loader import HousingDataLoader
from src.preprocessor import HousingPreprocessor
from src.model import HousingPriceModel, ModelTrainer
from src.evaluator import ModelEvaluator
from src.utils import set_random_seed, save_predictions, create_directories

def main():
    parser = argparse.ArgumentParser(description='Housing Price Prediction V4')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Config file path')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'evaluate'], 
                       default='train', help='Mode to run')
    parser.add_argument('--model-path', type=str, help='Path to saved model')
    
    args = parser.parse_args()
    
    # Create necessary directories
    create_directories()
    
    # Load configuration
    config = Config(args.config)
    
    # Set random seed
    set_random_seed(config.SEED)
    
    # Initialize components
    data_loader = HousingDataLoader(config)
    preprocessor = HousingPreprocessor(config)
    
    if args.mode == 'train':
        print("=== TRAINING MODE ===")
        # Load and preprocess data
        train_data, test_data = data_loader.load_data()
        X_train, X_test, y_train = preprocessor.preprocess(train_data, test_data)
        
        # Train model
        model = HousingPriceModel(X_train.shape[1], config)
        trainer = ModelTrainer(model, config)
        trained_model = trainer.train(X_train, y_train)
        
        # Evaluate
        evaluator = ModelEvaluator(config)
        metrics = evaluator.evaluate_model(trained_model, X_train, y_train)
        
        # Cross-validation
        print("\n=== CROSS-VALIDATION ===")
        cv_scores = evaluator.cross_validate(X_train, y_train)
        
        # Save model
        trainer.save_model('models/best_model.pth')
        
        print(f"\n=== TRAINING COMPLETED ===")
        print(f"Final Metrics: {metrics}")
        
    elif args.mode == 'predict':
        print("=== PREDICTION MODE ===")
        # Load model and make predictions
        if not args.model_path:
            args.model_path = 'models/best_model.pth'
            
        train_data, test_data = data_loader.load_data()
        X_train, X_test, y_train = preprocessor.preprocess(train_data, test_data)
        
        model = HousingPriceModel(X_test.shape[1], config)
        trainer = ModelTrainer(model, config)
        model = trainer.load_model(args.model_path)
        
        predictions = trainer.predict(X_test)
        save_predictions(predictions, test_data['Id'], 'submissions/submission_v4.csv')
        
    elif args.mode == 'evaluate':
        print("=== EVALUATION MODE ===")
        # Cross-validation evaluation only
        train_data, test_data = data_loader.load_data()
        X_train, X_test, y_train = preprocessor.preprocess(train_data, test_data)
        
        evaluator = ModelEvaluator(config)
        cv_scores = evaluator.cross_validate(X_train, y_train)
        print(f"CV Scores: {cv_scores}")

if __name__ == "__main__":
    main()
