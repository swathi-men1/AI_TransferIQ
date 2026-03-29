"""
Setup Script for Transfer IQ Project

This script sets up the environment and verifies all dependencies.
"""

import os
import subprocess
import sys

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed', 
        'models',
        'app',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f" Created directory: {directory}")

def install_dependencies():
    """Install required dependencies"""
    print(" Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f" Error installing dependencies: {e}")

def verify_setup():
    """Verify that all required files are in place"""
    print(" Verifying setup...")
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'README.md',
        'src/preprocessing.py',
        'src/feature_engineering.py',
        'src/xgb_model.py',
        'src/lstm_model.py',
        'src/ensemble.py',
        'src/evaluate.py',
        'app/app.py',
        'train_models.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f" Missing files: {missing_files}")
        return False
    else:
        print(" All required files are present!")
        return True

def main():
    """Run setup"""
    print("=" * 60)
    print("TRANSFER IQ - PROJECT SETUP")
    print("=" * 60)
    
    # Create directories
    print("\n Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n Installing dependencies...")
    install_dependencies()
    
    # Verify setup
    print("\n Verifying setup...")
    if verify_setup():
        print("\n Setup completed successfully!")
        print("\n Next steps:")
        print("   1. Place your data in data/raw/")
        print("   2. Run: python train_models.py")
        print("   3. Run: streamlit run app/app.py")
    else:
        print("\n Setup incomplete. Please check missing files.")

if __name__ == "__main__":
    main()
