"""
TransferIQ - Model Training & Prediction Utilities
====================================================
Handles XGBoost and LSTM model training, ensemble prediction,
and model persistence for the IPL Player Market Value Prediction system.
"""

import numpy as np
import joblib
import os
from datetime import datetime


def train_xgboost(X_train, y_train, X_test=None, y_test=None, **params):
    """Train an XGBoost regressor for player value prediction."""
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    default_params = {
        'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1
    }
    default_params.update(params)
    
    model = xgb.XGBRegressor(**default_params)
    model.fit(X_train, y_train)
    
    metrics = {}
    y_train_pred = model.predict(X_train)
    metrics['train_mae'] = mean_absolute_error(y_train, y_train_pred)
    metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train, y_train_pred))
    metrics['train_r2'] = r2_score(y_train, y_train_pred)
    
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
        metrics['test_r2'] = r2_score(y_test, y_test_pred)
        metrics['test_predictions'] = y_test_pred
    
    return model, metrics


def build_lstm_model(input_shape, units=64, learning_rate=0.001, **kwargs):
    """Build and compile an LSTM model for time-series value prediction."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import MeanAbsoluteError
    
    dropout_rate = kwargs.get('dropout_rate', 0.2)
    
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1, activation='relu')
    ])
    
    mae = MeanAbsoluteError(name='mae')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=[mae])
    return model


def train_lstm(X_train, y_train, X_test=None, y_test=None,
               epochs=50, batch_size=32, validation_split=0.1, verbose=1, **kwargs):
    """Train an LSTM model for player value prediction."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    X_train_np = np.array(X_train, dtype=float)
    X_seq = X_train_np.reshape((X_train_np.shape[0], 1, X_train_np.shape[1])) if X_train_np.ndim == 2 else X_train_np
    y_train_np = np.array(y_train, dtype=float)
    
    model = build_lstm_model(input_shape=(X_seq.shape[1], X_seq.shape[2]), **kwargs)
    history = model.fit(X_seq, y_train_np, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=verbose)
    
    metrics = {'history': history.history}
    y_train_pred = model.predict(X_seq, verbose=0).flatten()
    metrics['train_mae'] = mean_absolute_error(y_train_np, y_train_pred)
    metrics['train_rmse'] = np.sqrt(mean_squared_error(y_train_np, y_train_pred))
    metrics['train_r2'] = r2_score(y_train_np, y_train_pred)
    
    if X_test is not None and y_test is not None:
        X_test_np = np.array(X_test, dtype=float)
        X_test_seq = X_test_np.reshape((X_test_np.shape[0], 1, X_test_np.shape[1])) if X_test_np.ndim == 2 else X_test_np
        y_test_np = np.array(y_test, dtype=float)
        y_test_pred = model.predict(X_test_seq, verbose=0).flatten()
        metrics['test_mae'] = mean_absolute_error(y_test_np, y_test_pred)
        metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test_np, y_test_pred))
        metrics['test_r2'] = r2_score(y_test_np, y_test_pred)
        metrics['test_predictions'] = y_test_pred
    
    return model, metrics


def ensemble_predict(xgb_model, lstm_model, X, weights=(0.6, 0.4)):
    """Generate ensemble predictions by combining XGBoost and LSTM outputs."""
    xgb_pred = xgb_model.predict(X)
    
    X_np = np.array(X, dtype=float)
    X_seq = X_np.reshape((X_np.shape[0], 1, X_np.shape[1])) if X_np.ndim == 2 else X_np
    lstm_pred = lstm_model.predict(X_seq, verbose=0).flatten()
    
    if hasattr(lstm_model, '_y_scaler'):
        lstm_pred = lstm_model._y_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
    
    ensemble_pred = (weights[0] * xgb_pred + weights[1] * lstm_pred)
    
    return {'xgboost': xgb_pred, 'lstm': lstm_pred, 'ensemble': ensemble_pred, 'weights': weights}


def predict_single(xgb_model, lstm_model, features_array, scaler=None, feature_names=None, weights=(0.6, 0.4)):
    """Generate prediction for a single player input."""
    X = features_array.copy()
    if scaler is not None:
        X = scaler.transform(X)
    
    results = ensemble_predict(xgb_model, lstm_model, X, weights)
    
    return {
        'XGBoost_Prediction': float(results['xgboost'][0]),
        'LSTM_Prediction': float(results['lstm'][0]),
        'Ensemble_Prediction': float(results['ensemble'][0]),
        'XGBoost_Weight': weights[0],
        'LSTM_Weight': weights[1]
    }


def save_models(xgb_model, lstm_model, scaler, save_dir='models'):
    """Save trained models and scaler to disk."""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    joblib.dump(xgb_model, os.path.join(save_dir, f'xgb_model_{timestamp}.pkl'))
    lstm_model.save(os.path.join(save_dir, f'lstm_model_{timestamp}.h5'))
    joblib.dump(scaler, os.path.join(save_dir, f'scaler_{timestamp}.pkl'))
    
    joblib.dump(xgb_model, os.path.join(save_dir, 'xgb_model_latest.pkl'))
    lstm_model.save(os.path.join(save_dir, 'lstm_model_latest.h5'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler_latest.pkl'))
    
    return {'timestamp': timestamp}


def load_models(model_dir='models'):
    """Load trained models and scaler from disk."""
    import tensorflow as tf
    from tensorflow.keras.losses import MeanAbsoluteError
    
    xgb_path = os.path.join(model_dir, 'xgb_model_latest.pkl')
    lstm_path = os.path.join(model_dir, 'lstm_model_latest.h5')
    scaler_path = os.path.join(model_dir, 'scaler_latest.pkl')
    
    if not all(os.path.exists(p) for p in [xgb_path, lstm_path, scaler_path]):
        if not os.path.exists(model_dir):
            return None, None, None
        pkl_files = sorted([f for f in os.listdir(model_dir) if f.startswith('xgb_model_') and f.endswith('.pkl')], reverse=True)
        h5_files = sorted([f for f in os.listdir(model_dir) if f.startswith('lstm_model_') and f.endswith('.h5')], reverse=True)
        scaler_files = sorted([f for f in os.listdir(model_dir) if f.startswith('scaler_') and f.endswith('.pkl')], reverse=True)
        if pkl_files and h5_files and scaler_files:
            xgb_path = os.path.join(model_dir, pkl_files[0])
            lstm_path = os.path.join(model_dir, h5_files[0])
            scaler_path = os.path.join(model_dir, scaler_files[0])
        else:
            return None, None, None
    
    try:
        xgb_model = joblib.load(xgb_path)
        lstm_model = tf.keras.models.load_model(lstm_path, custom_objects={'mae': MeanAbsoluteError()})
        scaler = joblib.load(scaler_path)
        return xgb_model, lstm_model, scaler
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def generate_demo_models(feature_count=5):
    """Generate lightweight demo models for UI demonstration."""
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.losses import MeanAbsoluteError
    
    feature_names = ['Total_Runs', 'Avg_Strike_Rate', 'Total_Wickets', 'Avg_Economy_Rate', 'Severity_Score']
    
    np.random.seed(42)
    n_samples = 500
    X_demo = np.abs(np.random.randn(n_samples, feature_count) * np.array([200, 40, 10, 2, 0.5]))
    X_demo[:, 4] = np.clip(X_demo[:, 4], 0, 2)
    
    y_demo = (
        2_000_000 + X_demo[:, 0] * 20_000 + X_demo[:, 2] * 300_000 +
        X_demo[:, 1] * 5_000 - X_demo[:, 3] * 200_000 - X_demo[:, 4] * 500_000 +
        np.random.randn(n_samples) * 1_000_000
    )
    y_demo = np.clip(y_demo, 200_000, 200_000_000)
    
    xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_demo, y_demo)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_demo)
    X_seq = X_scaled.reshape((n_samples, 1, feature_count))
    
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_demo.reshape(-1, 1)).flatten()
    
    lstm_model = Sequential([
        Input(shape=(1, feature_count)),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse', metrics=[MeanAbsoluteError(name='mae')])
    lstm_model.fit(X_seq, y_scaled, epochs=50, batch_size=16, verbose=0)
    lstm_model._y_scaler = y_scaler
    
    return xgb_model, lstm_model, scaler, feature_names