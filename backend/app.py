from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load Models
lstm_model = tf.keras.models.load_model('models/lstm_tuned_final.keras', compile=False)
lstm_model.compile(optimizer='adam', loss='mse')
xgb_model  = joblib.load('models/xgboost_final.pkl')
print("✓ Models loaded!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "AI TransferIQ API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        features = np.array(data['features'], dtype=np.float32)

        # 3 timesteps x 27 features = 81 values
        seq_len      = 3
        num_features = 27

        lstm_input = features.reshape(1, seq_len, num_features)
        xgb_input  = features.reshape(1, -1)

        lstm_pred = lstm_model.predict(lstm_input, verbose=0).flatten()[0]
        xgb_pred  = xgb_model.predict(xgb_input)[0]
        ensemble  = (lstm_pred + xgb_pred) / 2

        return jsonify({
            "lstm_prediction":     round(float(lstm_pred), 4),
            "xgboost_prediction":  round(float(xgb_pred), 4),
            "ensemble_prediction": round(float(ensemble), 4),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)