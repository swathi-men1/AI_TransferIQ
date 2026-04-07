from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import xgboost as xgb
import numpy as np
import tensorflow as tf

# --- 1. App Initialization ---
app = Flask(__name__)
CORS(app)

# --- 2. Load Pre-trained Models ---
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('transferiq_model.json')

lstm_model = tf.keras.models.load_model('transferiq_lstm.keras')

def predict_single(performance, injury, sentiment, age, contract_years=3, position="MID"):
    
    # --- 3. Base Valuation (XGBoost) ---
    xgb_features = np.array([[performance / 10, injury, (sentiment + 1) / 2, age / 40]])
    raw = xgb_model.predict(xgb_features)[0]
    base_value = abs(raw) * 65000000

    # --- 4. Domain Logic Adjustments (Position & Contract) ---
    position_multiplier = {"FWD": 1.3, "MID": 1.1, "DEF": 0.9, "GK": 0.7}
    contract_multiplier = 1.0 + (contract_years - 2) * 0.15 
    value = base_value * position_multiplier.get(position, 1.0) * contract_multiplier

    # --- 5. Time-Series Prep (Generating historical sequence) ---
    seq = []
    for t in range(3, 0, -1):
        past = value / ((1 + 0.05) ** t)
        seq.append([past / 1e8, (age - t) / 40, performance / 10, injury])

    # --- 6. Future Trend Forecasting (LSTM) ---
    lstm_input = np.array([seq])
    lstm_val = abs(lstm_model.predict(lstm_input, verbose=0)[0][0]) * 1e8
    trend_mult = 1.15 if age <= 24 else (0.85 if age >= 30 else 1.02)

    forecast = [
        value,
        lstm_val * trend_mult,
        lstm_val * (trend_mult ** 2),
        lstm_val * (trend_mult ** 3)
    ]

    # --- 7. Financial Risk & Scenario Analysis ---
    low = value * 0.85
    high = value * 1.15
    best_scenario = value * 1.25
    worst_scenario = value * 0.70

    confidence_score = max(0, min(100, 100 - (injury * 40) - (abs(25 - age) * 1.5)))

    risk_score = (injury * 0.6 + (age / 40) * 0.2 + (5 - contract_years) * 0.05)
    if risk_score < 0.3:
        risk = "Low 🟢"
    elif risk_score < 0.6:
        risk = "Medium 🟡"
    else:
        risk = "High 🔴"

    # --- 8. Explainable AI & Categorization ---
    if value > 80000000:
        tier = "World Class"
    elif value > 40000000:
        tier = "Elite"
    elif value > 15000000:
        tier = "High"
    elif value > 5000000:
        tier = "Mid"
    else:
        tier = "Low"

    if age < 22:
        stage = "Wonderkid"
    elif age < 26:
        stage = "Young Talent"
    elif age < 30:
        stage = "Peak Player"
    else:
        stage = "Veteran"

    reasons = []
    if performance > 8: reasons.append("Elite Performance")
    if sentiment > 0.5: reasons.append("Strong Market Hype")
    if injury > 0.4: reasons.append("High Injury Risk")
    if contract_years <= 1: reasons.append("Expiring Contract")

    explanation = "Value driven by: " + ", ".join(reasons) if reasons else "Balanced overall profile"
    trend = "Increasing 📈" if forecast[-1] > value else "Declining 📉"
    feature_importance = "Performance: 35% | Age: 25% | Contract: 20% | Position: 10% | Sentiment: 10%"

    return {
        "value": float(value),
        "forecast": [float(x) for x in forecast],
        "low": float(low),
        "high": float(high),
        "best_scenario": float(best_scenario),
        "worst_scenario": float(worst_scenario),
        "confidence": float(confidence_score),
        "risk": risk,
        "tier": tier,
        "stage": stage,
        "explanation": explanation,
        "trend": trend,
        "feature_importance": feature_importance,
        "performance": float(performance),
        "injury": float(injury),
        "sentiment": float(sentiment)
    }

# --- 9. API Endpoints ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Handle single player request (Dashboard)
    if "player2" not in data:
        res = predict_single(
            performance=data['performance'],
            injury=data['injury'],
            sentiment=data['sentiment'],
            age=data['age'],
            contract_years=data.get('contract_years', 3),
            position=data.get('position', 'MID')
        )
        return jsonify(res)

    # Handle dual player comparison request
    p1 = predict_single(**data['player1'])
    p2 = predict_single(**data['player2'])
    diff = p1["value"] - p2["value"]

    return jsonify({
        "player1": p1,
        "player2": p2,
        "difference": float(diff)
    })

if __name__ == '__main__':
    app.run(debug=True)