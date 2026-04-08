import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from datetime import datetime

app = Flask(__name__, static_folder='frontend')
CORS(app)

# Load Model and Scalers
MODEL_PATH = "D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\XGboost\\xgboost_model.json"
SCALER_Y_PATH = "D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_y.pkl"

# Load XGBoost model
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# Load scalers
scaler_y = joblib.load(SCALER_Y_PATH)

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data: {data}")

        # Extract features from request
        player_name = data.get('player_name', 'Unknown Player')
        market_value = float(data.get('market_value', 0))
        performance_rating = float(data.get('performance_rating', 5.0))
        goals_assists = float(data.get('goals_assists', 0))
        minutes_played = float(data.get('minutes_played', 0))
        sentiment = float(data.get('sentiment', 50))
        contract = float(data.get('contract', 36))
        position = data.get('position', 'DF')

        # Scaling market value for input features
        mv_val = np.array([[market_value]])
        mv_scaled = scaler_y.transform(mv_val)[0][0]

        # Derived features
        form = goals_assists / (minutes_played + 1)
        injury_ratio = 0.0 # Mapping form values to sensible defaults
        
        # Build features list in exact order as training
        # [market_value_scaled, lag_1, lag_2, performance_rating, goals_assists, 
        #  minutes_played, days_injured, social_sentiment_score, contract_duration_months, 
        #  perf_trend_3m, goals_trend_3m, perf_vol_3m, cumulative_days_injured, 
        #  performance_change, sentiment_change, form, injury_ratio, month, year]
        
        now = datetime.now()
        
        features_dict = {
            'market_value_scaled': mv_scaled,
            'lag_1': mv_scaled,
            'lag_2': mv_scaled,
            'performance_rating': performance_rating,
            'goals_assists': goals_assists,
            'minutes_played': minutes_played,
            'days_injured': 0,
            'social_sentiment_score': sentiment,
            'contract_duration_months': contract,
            'perf_trend_3m': 0,
            'goals_trend_3m': 0,
            'perf_vol_3m': 0,
            'cumulative_days_injured': 0,
            'performance_change': 0,
            'sentiment_change': 0,
            'form': form,
            'injury_ratio': 0,
            'month': now.month,
            'year': now.year
        }

        # Handle one-hot encoding for positions
        # CSV has: position_Forward, position_Goalkeeper, position_Midfielder
        features_dict['position_Forward'] = 1 if position == 'FW' else 0
        features_dict['position_Goalkeeper'] = 1 if position == 'GK' else 0
        features_dict['position_Midfielder'] = 1 if position == 'MF' else 0

        # Create DataFrame for prediction
        # Ensure the column order matches the training features
        # Training feature list:
        feature_order = [
            'market_value_scaled', 'lag_1', 'lag_2',
            'performance_rating', 'goals_assists',
            'minutes_played', 'days_injured',
            'social_sentiment_score', 'contract_duration_months',
            'perf_trend_3m', 'goals_trend_3m', 'perf_vol_3m',
            'cumulative_days_injured',
            'performance_change', 'sentiment_change',
            'form', 'injury_ratio',
            'month', 'year',
            'position_Forward', 'position_Goalkeeper', 'position_Midfielder'
        ]
        
        X = pd.DataFrame([features_dict], columns=feature_order)
        dmatrix = xgb.DMatrix(X)

        # Predict log-return
        log_delta_pred = bst.predict(dmatrix)[0]
        
        # Reconstruction: final_preds = curr_mval * np.exp(log_delta_preds)
        predicted_value = market_value * np.exp(log_delta_pred)
        
        # Calculate percentage change
        change_percent = ((predicted_value - market_value) / market_value) * 100

        # Response
        return jsonify({
            'player_name': player_name,
            'predicted_value': round(predicted_value, 2),
            'change_percent': round(change_percent, 2),
            'log_return': float(log_delta_pred)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
