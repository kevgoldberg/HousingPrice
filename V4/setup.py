#!/usr/bin/env python3
"""
Setup script for Housing Price Prediction V4
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    print("Creating project directories...")
    directories = ['models', 'plots', 'submissions', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created: {directory}/")

def verify_data_paths():
    """Verify that data files exist"""
    print("Verifying data paths...")
    
    # Check relative paths from V4 directory
    train_path = Path("../house-prices-advanced-regression-techniques/train.csv")
    test_path = Path("../house-prices-advanced-regression-techniques/test.csv")
    
    if train_path.exists():
        print(f"  ✅ Found training data: {train_path}")
    else:
        print(f"  ⚠️  Training data not found: {train_path}")
        print("     Please ensure the Kaggle dataset is extracted in the parent directory")
    
    if test_path.exists():
        print(f"  ✅ Found test data: {test_path}")
    else:
        print(f"  ⚠️  Test data not found: {test_path}")
        print("     Please ensure the Kaggle dataset is extracted in the parent directory")

def main():
    print("🏠 Housing Price Prediction V4 Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Verify data paths
    verify_data_paths()
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Ensure the Kaggle dataset is in ../house-prices-advanced-regression-techniques/")
    print("2. Run: python main.py --mode train")
    print("3. Or open V4.ipynb in Jupyter/VS Code")

if __name__ == "__main__":
    main()
