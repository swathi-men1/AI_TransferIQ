"""
Create Mock Models for API Testing

This script creates mock trained models so you can test the API
without having to train the actual models first.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import pandas as pd


def create_directories():
    """Create necessary directories for models."""
    directories = [
        "models/lstm",
        "models/ensemble",
        "models/scalers",
        "data/training",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


class MockModel:
    """Mock model that generates random predictions."""
    
    def __init__(self, model_type="ensemble"):
        self.model_type = model_type
        self.feature_names = [
            'age', 'goals', 'assists', 'minutes', 'market_value',
            'injury_risk', 'sentiment_score', 'contract_years'
        ]
    
    def predict(self, X):
        """Generate mock predictions."""
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
        else:
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
        
        # Generate realistic transfer values (in millions)
        base_values = np.random.uniform(5, 100, n_samples)
        
        # Add some noise
        noise = np.random.normal(0, 5, n_samples)
        predictions = base_values + noise
        
        # Ensure positive values
        predictions = np.maximum(predictions, 1.0)
        
        # Convert to actual currency (multiply by 1 million)
        predictions = predictions * 1_000_000
        
        return predictions


class MockScaler:
    """Mock scaler for feature normalization."""
    
    def transform(self, X):
        """Mock transform (returns input as-is)."""
        return X
    
    def inverse_transform(self, X):
        """Mock inverse transform (returns input as-is)."""
        return X


def create_mock_ensemble_models():
    """Create mock XGBoost and LightGBM models."""
    print("\n📦 Creating mock ensemble models...")
    
    # Create mock XGBoost model
    xgb_model = MockModel("xgboost")
    xgb_path = Path("models/ensemble/xgboost_model.pkl")
    with open(xgb_path, 'wb') as f:
        pickle.dump(xgb_model, f)
    print(f"✓ Created mock XGBoost model: {xgb_path}")
    
    # Create mock LightGBM model
    lgb_model = MockModel("lightgbm")
    lgb_path = Path("models/ensemble/lightgbm_model.pkl")
    with open(lgb_path, 'wb') as f:
        pickle.dump(lgb_model, f)
    print(f"✓ Created mock LightGBM model: {lgb_path}")


def create_mock_scaler():
    """Create mock feature scaler."""
    print("\n📏 Creating mock scaler...")
    
    scaler = MockScaler()
    scaler_path = Path("models/scalers/feature_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Created mock scaler: {scaler_path}")


def create_mock_player_data():
    """Create mock player dataset."""
    print("\n👥 Creating mock player data...")
    
    # Generate mock player data
    n_players = 100
    
    player_data = pd.DataFrame({
        'player_id': [f"player_{i:05d}" for i in range(n_players)],
        'player_name': [f"Player {i}" for i in range(n_players)],
        'age': np.random.randint(18, 35, n_players),
        'position': np.random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper'], n_players),
        'goals': np.random.randint(0, 30, n_players),
        'assists': np.random.randint(0, 20, n_players),
        'minutes': np.random.randint(500, 3500, n_players),
        'market_value': np.random.uniform(1, 100, n_players) * 1_000_000,
        'injury_risk': np.random.uniform(0, 1, n_players),
        'sentiment_score': np.random.uniform(-1, 1, n_players),
        'contract_years': np.random.randint(1, 5, n_players),
        'club': np.random.choice(['Club A', 'Club B', 'Club C', 'Club D'], n_players),
    })
    
    # Add some well-known player IDs for testing
    test_players = pd.DataFrame({
        'player_id': ['12345', '67890', '11111', '22222', '33333'],
        'player_name': ['Test Player 1', 'Test Player 2', 'Test Player 3', 'Test Player 4', 'Test Player 5'],
        'age': [25, 28, 22, 30, 26],
        'position': ['Forward', 'Midfielder', 'Defender', 'Forward', 'Midfielder'],
        'goals': [20, 10, 2, 25, 8],
        'assists': [8, 15, 1, 5, 12],
        'minutes': [3000, 2800, 2500, 3200, 2900],
        'market_value': [50_000_000, 40_000_000, 25_000_000, 60_000_000, 35_000_000],
        'injury_risk': [0.2, 0.3, 0.1, 0.4, 0.2],
        'sentiment_score': [0.8, 0.6, 0.5, 0.9, 0.7],
        'contract_years': [3, 2, 4, 2, 3],
        'club': ['Club A', 'Club B', 'Club C', 'Club A', 'Club D'],
    })
    
    # Combine datasets
    player_data = pd.concat([test_players, player_data], ignore_index=True)
    
    # Save to CSV
    data_path = Path("data/training/training_dataset.csv")
    player_data.to_csv(data_path, index=False)
    print(f"✓ Created mock player data: {data_path}")
    print(f"  - Total players: {len(player_data)}")
    print(f"  - Test player IDs: 12345, 67890, 11111, 22222, 33333")


def create_readme():
    """Create README for mock models."""
    readme_content = """# Mock Models for API Testing

These are mock models created for testing the API without training actual models.

## What's Included:

- **XGBoost Model**: `models/ensemble/xgboost_model.pkl`
- **LightGBM Model**: `models/ensemble/lightgbm_model.pkl`
- **Feature Scaler**: `models/scalers/feature_scaler.pkl`
- **Player Data**: `data/training/training_dataset.csv`

## Test Player IDs:

You can use these player IDs for testing:
- `12345` - Test Player 1 (Forward, €50M)
- `67890` - Test Player 2 (Midfielder, €40M)
- `11111` - Test Player 3 (Defender, €25M)
- `22222` - Test Player 4 (Forward, €60M)
- `33333` - Test Player 5 (Midfielder, €35M)

## Note:

These models generate random predictions for testing purposes only.
To get real predictions, train the actual models using:

```bash
python scripts/train_lstm.py
python scripts/run_pipeline.py --train-ensemble
```

## Predictions:

The mock models will return random transfer values between €1M and €100M.
Each prediction will be slightly different due to randomness.
"""
    
    readme_path = Path("models/README_MOCK_MODELS.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"\n✓ Created README: {readme_path}")


def main():
    """Main function to create all mock components."""
    print("=" * 60)
    print("Creating Mock Models for API Testing")
    print("=" * 60)
    
    # Create directories
    create_directories()
    
    # Create mock models
    create_mock_ensemble_models()
    
    # Create mock scaler
    create_mock_scaler()
    
    # Create mock player data
    create_mock_player_data()
    
    # Create README
    create_readme()
    
    print("\n" + "=" * 60)
    print("✅ Mock models created successfully!")
    print("=" * 60)
    print("\nYou can now start the API server:")
    print("  uvicorn src.api.app:app --reload")
    print("\nOr run the test script:")
    print("  python test_api_endpoints.py")
    print("\nTest with these player IDs: 12345, 67890, 11111, 22222, 33333")
    print("=" * 60)


if __name__ == "__main__":
    main()
