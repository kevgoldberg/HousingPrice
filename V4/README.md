# Housing Price Prediction V4

A machine learning project for predicting housing prices using neural networks.

## Project Structure
```
V4/
├── main.py                 # Main script
├── config.yaml            # Configuration file
├── src/                   # Source code modules
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   ├── evaluator.py
│   └── utils.py
├── models/                # Saved models
├── submissions/           # Prediction outputs
└── data/                  # Dataset files
```

## Usage

### Training
```bash
python main.py --mode train
```

### Prediction
```bash
python main.py --mode predict --model-path models/best_model.pth
```

### Evaluation
```bash
python main.py --mode evaluate
```

## Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- PyYAML

## Setup
```bash
# Clone repository
git clone https://github.com/kevgoldberg/Housing.git
cd Housing/V4

# Install requirements
pip install -r requirements.txt

# Run training
python main.py --mode train
```
