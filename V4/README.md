# Housing Price Prediction V4

This project converts the V3 Jupyter notebook into a structured, modular Python application for housing price prediction using PyTorch.

## Project Structure

```
V4/
├── main.py                 # Main script with CLI interface
├── config.yaml            # Configuration file
├── src/
│   ├── config.py          # Configuration management
│   ├── data_loader.py     # Data loading utilities
│   ├── preprocessor.py    # Data preprocessing pipeline
│   ├── model.py           # Neural network model and training
│   ├── evaluator.py       # Model evaluation and cross-validation
│   └── utils.py           # Utility functions
├── models/                # Saved models and scalers
├── plots/                 # Generated visualizations
├── submissions/           # Kaggle submission files
└── logs/                  # Log files
```

## Features

- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: YAML-based configuration for easy experimentation
- **Command Line Interface**: Easy to run different modes (train/predict/evaluate)
- **Model Persistence**: Save and load trained models
- **Cross-Validation**: Built-in k-fold cross-validation
- **Visualization**: Automatic plot generation
- **Reproducibility**: Consistent random seed management

## Requirements

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib seaborn pyyaml joblib
```

## Usage

### Training

```bash
python main.py --mode train
```

### Making Predictions

```bash
python main.py --mode predict --model-path models/best_model.pth
```

### Cross-Validation Only

```bash
python main.py --mode evaluate
```

### Custom Configuration

```bash
python main.py --config custom_config.yaml --mode train
```

## Configuration

Edit `config.yaml` to customize:

- **Data paths**: Location of train/test CSV files
- **Model architecture**: Hidden layer sizes, dropout rate
- **Training parameters**: Learning rate, batch size, epochs
- **Preprocessing**: Validation ratio, missing value handling
- **Cross-validation**: Number of folds, CV epochs

## Key Improvements from V3

1. **Production Ready**: Structured code suitable for deployment
2. **Maintainable**: Clear module boundaries and documentation
3. **Configurable**: Easy to experiment with different parameters
4. **Extensible**: Simple to add new features or models
5. **Error Handling**: Better validation and error messages
6. **Memory Efficient**: Optimized data types and GPU usage

## Model Architecture

- Neural network with configurable hidden layers
- ReLU activation and dropout for regularization
- AdamW optimizer with learning rate scheduling
- Early stopping to prevent overfitting
- Log-transformed target for better predictions

## Evaluation Metrics

- **RMSE**: Root Mean Square Error on original price scale
- **R²**: Coefficient of determination
- **Log-RMSE**: Kaggle competition metric (log-transformed prices)

## Output Files

- `models/best_model.pth`: Trained model weights
- `models/scaler.pkl`: Fitted StandardScaler
- `submissions/submission_v4.csv`: Kaggle submission file
- `plots/predictions_vs_actual.png`: Prediction visualization
