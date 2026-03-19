import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_ensemble(df):
    print("Training XGBoost Ensemble Model...")
    features = [
        'performance_rating', 'goals_assists', 'minutes_played', 'days_injured',
        'social_sentiment_score', 'contract_duration_months',
        'perf_trend_3m', 'goals_trend_3m', 'cumulative_days_injured'
    ]
    position_cols = [col for col in df.columns if col.startswith('position_')]
    features.extend(position_cols)

    # Chronological Split (No data leakage)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    X = df[features]
    y = df['market_value_scaled']
    
    # 80/20 Time series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]
    num_round = 150
    bst = xgb.train(
        params, dtrain, num_round, evals,
        early_stopping_rounds=15,
        verbose_eval=10
    )

    os.makedirs('models', exist_ok=True)
    bst.save_model("models/xgboost_model.json")

    preds = bst.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\nXGBoost Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    return bst, rmse, mae, r2

if __name__ == "__main__":
    df = pd.read_csv("transferiq_processed.csv")
    train_ensemble(df)
