import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_hybrid_model(df):
    print("Training Hybrid XGBoost Model (Multi-step + % Change)...")

    # Sort to compute valid baselines chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player_id', 'date'])

    # Target: Log-return (1-step ahead)
    # Log-returns are more symmetric and work better for financial forecasting
    df['target'] = np.log1p(df.groupby('player_id')['market_value'].shift(-1)) - np.log1p(df['market_value'])

    # CLIP OUTLIERS in log-space (reasonable bounds)
    df['target'] = df['target'].clip(-0.5, 0.7)

    # CLEAN BAD VALUES
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df['performance_change'] = df.groupby('player_id')['performance_rating'].diff()
    df['sentiment_change'] = df.groupby('player_id')['social_sentiment_score'].diff()
    df['form'] = df['goals_assists'] / (df['minutes_played'] + 1)
    df['injury_ratio'] = df['days_injured'] / (df['minutes_played'] + 1)
    
    df = df.dropna()
    
    features = [
        'market_value_scaled', 
        'lag_1', 'lag_2',
        'performance_rating', 'goals_assists',
        'minutes_played', 'days_injured',
        'social_sentiment_score', 'contract_duration_months',
        'perf_trend_3m', 'goals_trend_3m', 'perf_vol_3m',
        'cumulative_days_injured',
        'performance_change', 'sentiment_change',
        'form', 'injury_ratio',
        'month', 'year'
    ]

    # add position columns
    position_cols = [col for col in df.columns if col.startswith('position_')]
    features.extend(position_cols)

    X = df[features]
    y = df['target']

    # Temporal Split
    df_sorted = df.sort_values(by='date')
    split_idx = int(len(df_sorted) * 0.8)
    
    X_train = X.loc[df_sorted.index[:split_idx]]
    X_test  = X.loc[df_sorted.index[split_idx:]]
    y_train = y.loc[df_sorted.index[:split_idx]]
    y_test  = y.loc[df_sorted.index[split_idx:]]

    # Sample Weighting: Give more weight to non-zero moves to beat naive baseline
    # High weights on large jumps force the model to learn signals that naive misses.
    weights_train = 1.0 + 5.0 * np.abs(y_train)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # HIGH PERFORMANCE PARAMS
    params = {
        'booster': 'dart',        # Dropout for better generalization
        'max_depth': 4,
        'eta': 0.02,
        'objective': 'reg:squarederror',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 5,
        'alpha': 1,
        'rate_drop': 0.1,         # DART drop rate
        'skip_drop': 0.5          # Probability of skipping dropout
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=100,
        verbose_eval=200
    )

    save_path = "D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\XGboost"
    os.makedirs(save_path, exist_ok=True)
    bst.save_model(os.path.join(save_path, "xgboost_model.json"))

    # Reconstruction logic (Exponential for log-returns)
    log_delta_preds = bst.predict(dtest)
    
    # preds = market_value * exp(log_return)
    # We use scaled values for reconstruction consistency
    current_val_scaled = X_test['market_value_scaled']
    
    # Load scalers
    scaler_y = joblib.load('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_y.pkl')
    
    # Convert scaled current value back to real market value for reconstruction
    curr_mval = scaler_y.inverse_transform(current_val_scaled.values.reshape(-1, 1)).flatten()
    
    final_preds = curr_mval * np.exp(log_delta_preds)
    
    # Actuals
    actual_log_next = np.log1p(scaler_y.inverse_transform(y_test.values.reshape(-1, 1))) # This is wrong, y_test is the target log-return
    # Correct way to get actuals:
    # y_test = log1p(next) - log1p(curr) => log1p(next) = log1p(curr) + y_test => next = exp(log1p(curr) + y_test) - 1
    actual = np.expm1(np.log1p(curr_mval) + y_test)

    rmse = np.sqrt(mean_squared_error(actual, final_preds))
    mae = mean_absolute_error(actual, final_preds)
    r2 = r2_score(actual, final_preds)

    print("\nAdvanced XGBoost Results (Real Values):")
    print(f"RMSE: €{rmse:,.2f}")
    print(f"MAE:  €{mae:,.2f}")
    print(f"R2:   {r2:.4f}")

    return bst, rmse, mae, r2


if __name__ == "__main__":
    df = pd.read_csv(
        "D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\transferiq_processed.csv"
    )
    train_hybrid_model(df)