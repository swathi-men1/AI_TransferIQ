"""
Fix Mock Models - Create simpler mock models that work with pickle
"""

import os
import pickle
import numpy as np
from pathlib import Path
import pandas as pd


# Define classes at module level so they can be pickled
class SimplePredictor:
    """Simple predictor that generates random predictions."""
    
    def predict(self, X):
        """Generate mock predictions."""
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
        elif hasattr(X, 'shape'):
            n_samples = X.shape[0] if len(X.shape) > 1 else 1
        else:
            n_samples = 1
        
        # Generate realistic transfer values (in millions)
        base_values = np.random.uniform(10, 80, n_samples)
        noise = np.random.normal(0, 5, n_samples)
        predictions = (base_values + noise) * 1_000_000
        predictions = np.maximum(predictions, 1_000_000)
        
        return predictions


class SimpleScaler:
    """Simple scaler that passes through data."""
    
    def transform(self, X):
        return X
    
    def inverse_transform(self, X):
        return X


def main():
    print("Fixing mock models...")
    
    # Create directories
    Path("models/ensemble").mkdir(parents=True, exist_ok=True)
    Path("models/scalers").mkdir(parents=True, exist_ok=True)
    
    # Create and save models
    xgb_model = SimplePredictor()
    lgb_model = SimplePredictor()
    scaler = SimpleScaler()
    
    with open("models/ensemble/xgboost_model.pkl", 'wb') as f:
        pickle.dump(xgb_model, f)
    print("✓ Created XGBoost model")
    
    with open("models/ensemble/lightgbm_model.pkl", 'wb') as f:
        pickle.dump(lgb_model, f)
    print("✓ Created LightGBM model")
    
    with open("models/scalers/feature_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("✓ Created scaler")
    
    print("\n✅ Mock models fixed! Restart the server to load them.")


if __name__ == "__main__":
    main()
