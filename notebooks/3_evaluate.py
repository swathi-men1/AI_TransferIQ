import os
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

def load_data_and_models():
    df = pd.read_csv("D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\transferiq_processed.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    scaler_y = joblib.load('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_y.pkl')
    
    # Load Models
    model_u = load_model('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\univariate_lstm.keras')
    model_m = load_model('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\multivariate_lstm.keras')
        
    bst = xgb.Booster()
    bst.load_model("D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\XGboost\\xgboost_model.json")
        
    # Load Histories
    hist_u = np.load('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\hist_uni.npy', allow_pickle=True).item()
    hist_m = np.load('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\hist_multi.npy', allow_pickle=True).item()
    return df, scaler_y, model_u, model_m, bst, hist_u, hist_m

def plot_loss_curves(hist_u, hist_m):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_u['loss'], label='Train Loss')
    plt.plot(hist_u['val_loss'], label='Val Loss')
    plt.title('Univariate LSTM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
        
    plt.subplot(1, 2, 2)
    plt.plot(hist_m['loss'], label='Train Loss')
    plt.plot(hist_m['val_loss'], label='Val Loss')
    plt.title('Multivariate LSTM Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
        
    plt.tight_layout()
    plt.savefig('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\figures\\loss_curves.png')
    plt.close()

def evaluate_xgboost(df, bst, scaler_y):
    # Sort to compute valid baselines chronologically
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['player_id', 'date'])
    
    # Compute naive forecast (previous value) and rolling average (last 3 intervals)
    df['naive_scaled'] = df.groupby('player_id')['market_value_scaled'].shift(1)
    df['rolling_scaled'] = df.groupby('player_id')['market_value_scaled'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    
    # On-the-fly feature engineering (must match XGboost_train.py)
    df['performance_change'] = df.groupby('player_id')['performance_rating'].diff()
    df['sentiment_change'] = df.groupby('player_id')['social_sentiment_score'].diff()
    df['form'] = df['goals_assists'] / (df['minutes_played'] + 1)
    df['injury_ratio'] = df['days_injured'] / (df['minutes_played'] + 1)
        
    # Temporal Split: Last 20% of indices after sorting by date globally
    df_sorted = df.sort_values(by='date')
    split_idx = int(len(df_sorted) * 0.8)
    test_df = df.loc[df_sorted.index[split_idx:]].copy()

    eval_df = test_df.dropna(subset=['naive_scaled', 'performance_change']).copy()
        
    features = [
        'market_value_scaled', 
        'lag_1', 'lag_2',
        'performance_rating', 'goals_assists', 'minutes_played', 'days_injured',
        'social_sentiment_score', 'contract_duration_months', 
        'perf_trend_3m', 'goals_trend_3m', 'perf_vol_3m',
        'cumulative_days_injured',
        'performance_change', 'sentiment_change',
        'form', 'injury_ratio',
        'month', 'year'
    ]
    position_cols = [col for col in df.columns if col.startswith('position_')]
    features.extend(position_cols)
        
    X = eval_df[features]
    
    # XGBoost predicts log-return: log(next) - log(curr)
    dtest = xgb.DMatrix(X)
    log_delta_preds = bst.predict(dtest)
    
    # SMALL CHANGE SUPPRESSION: Default to 0 log-return (no change) if pred is very small
    log_delta_preds[np.abs(log_delta_preds) < 0.012] = 0
    
    # Reconstruct predictions in real values
    curr_mval = scaler_y.inverse_transform(eval_df['market_value_scaled'].values.reshape(-1, 1)).flatten()
    preds = curr_mval * np.exp(log_delta_preds)
    
    # Actuals for the next step
    # We must be careful to match indices
    y_next_mval = df.groupby('player_id')['market_value'].shift(-1)
    y_true = y_next_mval.loc[eval_df.index]
    
    valid_mask = y_true.notnull()
    y_true = y_true[valid_mask].values.reshape(-1, 1)
    preds = preds[valid_mask].reshape(-1, 1)
        
    # Baselines in real values
    naive_preds = scaler_y.inverse_transform(eval_df.loc[valid_mask, 'naive_scaled'].values.reshape(-1, 1))
    rolling_preds = scaler_y.inverse_transform(eval_df.loc[valid_mask, 'rolling_scaled'].values.reshape(-1, 1))
        
    # Model Metrics
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    mape = mean_absolute_percentage_error(y_true, preds)
    r2 = r2_score(y_true, preds)
        
    # Baseline MAPE
    naive_mape = mean_absolute_percentage_error(y_true, naive_preds)
    rolling_mape = mean_absolute_percentage_error(y_true, rolling_preds)
        
    # Business logic improvement calculation
    naive_improvement = (naive_mape - mape) / (naive_mape + 1e-6) * 100
        
    if naive_improvement > 0:
        business_explanation = f"Success! The model provides a {naive_improvement:.1f}% improvement over the naive baseline."
    elif max(naive_improvement, (rolling_mape-mape)/rolling_mape*100) < 5:
        business_explanation = "The model is struggling to beat the simple naive baseline. More feature engineering recommended."
    else:
        business_explanation = "The model is showing gains over rolling baselines, but needs more stability."
            
    return rmse, mae, r2, mape, naive_mape, rolling_mape, naive_improvement, business_explanation, y_true, preds

def evaluate_lstm(df, model_m, scaler_y):
    # Sort globally by date
    df_sorted = df.sort_values(by='date')
    split_idx = int(len(df_sorted) * 0.8)
    test_df = df.loc[df_sorted.index[split_idx:]].copy()
        
    def create_sequences(data_x, data_y, seq_length=3):
        xs, ys = [], []
        for i in range(len(data_x) - seq_length):
            xs.append(data_x[i:(i + seq_length)])
            ys.append(data_y[i + seq_length])
        return np.array(xs), np.array(ys)
            
    multi_cols = [
        'performance_rating', 'goals_assists', 'days_injured', 
        'social_sentiment_score', 'market_value_scaled', 'perf_trend_3m',
        'lag_1', 'lag_2', 'perf_vol_3m'
    ]
        
    X_test, y_test = [], []
    for player_id, group in test_df.groupby('player_id'):
        if len(group) <= 3: continue
        dx = group[multi_cols].values
        dy = group[['market_value_scaled']].values
        x, y = create_sequences(dx, dy, 3)
        X_test.extend(x)
        y_test.extend(y)
            
    if not X_test:
        return 0, 0
            
    X_test = np.array(X_test)
    y_test_scaled = np.array(y_test)
        
    preds_scaled = model_m.predict(X_test, verbose=0)
        
    y_true = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1))
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
    
    mape = mean_absolute_percentage_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    
    return r2, mape


def plot_predictions(y_true, preds, title, filename):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, preds, alpha=0.3, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(title)
    plt.xlabel('Actual Transfer Value')
    plt.ylabel('Predicted Transfer Value')
    plt.tight_layout()
    plt.savefig(f'D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\figures\\{filename}')
    plt.close()
    
if __name__ == "__main__":
    os.makedirs('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\figures', exist_ok=True)
    print("Loading models and evaluating...")
    df, scaler_y, model_u, model_m, bst, hist_u, hist_m =load_data_and_models()
        
    print("Plotting LSTM loss curves...")
    plot_loss_curves(hist_u, hist_m)
        
    print("Evaluating XGBoost model over entire dataset...")
    rmse, mae, r2, mape, naive_mape, rolling_mape, naive_improvement, business_explanation, y_true, preds = evaluate_xgboost(df, bst, scaler_y)
        
    print("Evaluating Multivariate LSTM...")
    lstm_r2, lstm_mape = evaluate_lstm(df, model_m, scaler_y)
        
    print("--- XGBoost Final Evaluation (Real Values) ---")
    print(f"RMSE: €{rmse:,.2f}")
    print(f"MAE:  €{mae:,.2f}")
    print(f"MAPE: {mape:.2%}")
    print(f"R2:   {r2:.4f}")
    print(f"\n--- LSTM Evaluation ---")
    print(f"LSTM MAPE: {lstm_mape:.2%}")
    print(f"LSTM R2:   {lstm_r2:.4f}")
    print(f"\n--- Baselines ---")
    print(f"Naive Forecast MAPE: {naive_mape:.2%}")
    print(f"Rolling Avg MAPE: {rolling_mape:.2%}")
    print(f"Improvement over Naive (XGB): {naive_improvement:.1f}%")
    print(f"Business Assessment: {business_explanation}")
        
        # Save metrics for frontend
    metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2),
            "lstm_r2": float(lstm_r2),
            "lstm_mape": float(lstm_mape),
            "naive_mape": float(naive_mape),
            "rolling_mape": float(rolling_mape),
            "naive_improvement": float(naive_improvement),
            "business_explanation": business_explanation
    }
    with open('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\metrics.json', 'w') as f:
        json.dump(metrics, f)
        
    plot_predictions(y_true, preds, 'XGBoost Predicted vs Actual Transfer Values', 'xgb_predictions.png')

    print("Generating player trajectory visualization...")
    # Use a player that has enough data for features
    sample_player_id = df['player_id'].unique()[1]
    player_data = df[df['player_id'] == sample_player_id].copy()
    player_data = player_data.sort_values(by='date')
    
    # On-the-fly feature engineering for this sample player
    player_data['performance_change'] = player_data['performance_rating'].diff()
    player_data['sentiment_change'] = player_data['social_sentiment_score'].diff()
    player_data['form'] = player_data['goals_assists'] / (player_data['minutes_played'] + 1)
    player_data['injury_ratio'] = player_data['days_injured'] / (player_data['minutes_played'] + 1)
    
    player_data = player_data.dropna(subset=['performance_change'])
        
    # Get predictions for just this player using XGB
    features = [
        'market_value_scaled', 
        'lag_1', 'lag_2',
        'performance_rating', 'goals_assists', 'minutes_played', 'days_injured',
        'social_sentiment_score', 'contract_duration_months', 
        'perf_trend_3m', 'goals_trend_3m', 'perf_vol_3m',
        'cumulative_days_injured',
        'performance_change', 'sentiment_change',
        'form', 'injury_ratio',
        'month', 'year'
    ]
    position_cols = [col for col in df.columns if col.startswith('position_')]
    features.extend(position_cols)
        
    dplayer = xgb.DMatrix(player_data[features])
    log_delta_preds = bst.predict(dplayer)
    
    # Reconstruct predictions: current * exp(log_return)
    curr_mval_player = scaler_y.inverse_transform(player_data['market_value_scaled'].values.reshape(-1, 1)).flatten()
    player_preds = curr_mval_player * np.exp(log_delta_preds)
    player_actual = scaler_y.inverse_transform(player_data['market_value_scaled'].values.reshape(-1, 1))
        
    plt.figure(figsize=(12, 6))
    plt.plot(player_data['date'], player_actual, label='Actual Value', marker='o')
    plt.plot(player_data['date'], player_preds, label='Predicted Value (XGB)', marker='x')
    plt.title(f'Transfer Value Over Time for Player {sample_player_id}')
    plt.xlabel('Date')
    plt.ylabel('Transfer Value (€)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\figures\\player_trajectory.png')
    plt.close()
        
    print("Evaluation scripts complete. Visualizations saved into standard '../reports/figures' folder.")
