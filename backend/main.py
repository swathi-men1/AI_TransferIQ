"""
AI TransferIQ - FastAPI Backend
Serves the full AI pipeline via REST API with CORS enabled for the React frontend.
"""
import os
import sys
import subprocess
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Ensure we run from the project root ──────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

os.makedirs("visualizations", exist_ok=True)
os.makedirs("models", exist_ok=True)

app = FastAPI(title="AI TransferIQ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/visualizations", StaticFiles(directory="visualizations"), name="visualizations")


# ── Helpers ───────────────────────────────────────────────────────────────────
def run_script(script_name: str):
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True, text=True, check=True, cwd=str(ROOT_DIR)
        )
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Script error:\n{e.stderr}")


def list_viz_images(prefix: str = "") -> list:
    viz_dir = ROOT_DIR / "visualizations"
    return [
        f"/api/visualizations/{f.name}"
        for f in sorted(viz_dir.glob(f"{prefix}*.png"))
    ]


# ── Status ────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    files = {
        "dataset": (ROOT_DIR / "transferiq_dataset.csv").exists(),
        "processed": (ROOT_DIR / "transferiq_processed.csv").exists(),
        "xgboost_model": (ROOT_DIR / "models/xgboost_model.json").exists(),
        "univariate_lstm": (ROOT_DIR / "models/univariate_lstm.keras").exists(),
        "multivariate_lstm": (ROOT_DIR / "models/multivariate_lstm.keras").exists(),
        "encoder_decoder_lstm": (ROOT_DIR / "models/encoder_decoder_lstm.keras").exists(),
    }
    metrics = None
    if (ROOT_DIR / "models/metrics.json").exists():
        import json
        with open(ROOT_DIR / "models/metrics.json", "r") as f:
            metrics = json.load(f)
            
    return {"status": "online", "files": files, "metrics": metrics}


# ── Data Pipeline ─────────────────────────────────────────────────────────────
@app.post("/api/data/generate")
async def generate_data(force: bool = False):
    dataset = ROOT_DIR / "transferiq_dataset.csv"
    if dataset.exists() and not force:
        return {"status": "skipped", "output": "Dataset already exists. Pass ?force=true to regenerate."}
    return run_script("generate_data.py")


@app.post("/api/data/sentiment")
async def run_sentiment(force: bool = False):
    out = ROOT_DIR / "transferiq_with_sentiment.csv"
    if out.exists() and not force:
        return {"status": "skipped", "output": "Sentiment analysis already run. Pass ?force=true to redo."}
    return run_script("sentiment_analysis.py")


@app.post("/api/data/preprocess")
async def preprocess_data(force: bool = False):
    processed = ROOT_DIR / "transferiq_processed.csv"
    if processed.exists() and not force:
        return {"status": "skipped", "output": "Processed data already exists. Pass ?force=true to redo."}
    return run_script("preprocess.py")


@app.post("/api/data/eda")
async def run_eda(force: bool = False):
    eda_images = list_viz_images("eda_")
    if eda_images and not force:
        return {"status": "skipped", "output": "EDA already run.", "images": eda_images}
    result = run_script("eda.py")
    result["images"] = list_viz_images("eda_")
    return result


# ── Model Training ────────────────────────────────────────────────────────────
@app.post("/api/models/train/{model_type}")
async def train_model(model_type: str, force: bool = False):
    if model_type == "xgboost":
        if (ROOT_DIR / "models/xgboost_model.json").exists() and not force:
            return {"status": "skipped", "output": "XGBoost model already trained. Pass ?force=true to retrain."}
        return run_script("train_xgboost.py")
    elif model_type == "lstm":
        if (ROOT_DIR / "models/univariate_lstm.keras").exists() and not force:
            return {"status": "skipped", "output": "LSTM models already trained. Pass ?force=true to retrain."}
        return run_script("train_lstm.py")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: '{model_type}'.")


# ── Evaluation ────────────────────────────────────────────────────────────────
@app.post("/api/models/evaluate")
async def evaluate_models():
    result = run_script("evaluate.py")
    result["images"] = list_viz_images()
    if (ROOT_DIR / "models/metrics.json").exists():
        import json
        with open(ROOT_DIR / "models/metrics.json", "r") as f:
            result["metrics"] = json.load(f)
    return result


# ── Training History ──────────────────────────────────────────────────────────
@app.get("/api/models/history")
async def get_training_history():
    """Return real LSTM training loss curves from saved numpy history files."""
    def load_hist(path):
        p = ROOT_DIR / path
        if not p.exists():
            return None
        h = np.load(str(p), allow_pickle=True).item()
        return {k: [round(float(v), 6) for v in vals] for k, vals in h.items()}

    uni = load_hist("models/hist_uni.npy")
    multi = load_hist("models/hist_multi.npy")
    enc_dec = load_hist("models/hist_enc_dec.npy")

    if not uni and not multi:
        return {"status": "no_history", "message": "Train LSTM models first."}

    return {
        "status": "ok",
        "univariate": uni,
        "multivariate": multi,
        "encoder_decoder": enc_dec,
    }


# ── Live Data API (Simulated Web Fetch) ───────────────────────────────────────
@app.get("/api/live-stats")
async def get_live_player_stats(name: str):
    """Simulates fetching real-time data from an external provider (StatsBomb/Twitter)."""
    import random
    
    # Hash the name to seed the randomizer so the same name always gets the same 'live' stats
    random.seed(hash(name.lower().strip()))
    
    positions = ['Forward', 'Midfielder', 'Defender'] if 'striker' in name.lower() or 'messi' in name.lower() else ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']
    
    live_data = {
        "player_name": name.title(),
        "performance_rating": round(random.uniform(70.0, 95.0), 1),
        "goals_assists": random.randint(0, 35),
        "minutes_played": random.randint(500, 3500),
        "days_injured": random.choice([0, 0, 0, 5, 12, 35, 90]),  # weighted towards 0
        "social_sentiment_score": round(random.uniform(-0.6, 0.9), 2),
        "contract_duration_months": random.randint(6, 60),
        "position": random.choice(positions),
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    }
    return live_data


# ── Live Prediction Endpoint ──────────────────────────────────────────────────
class PlayerInput(BaseModel):
    performance_rating: float = 75.0
    goals_assists: int = 3
    minutes_played: int = 180
    days_injured: int = 0
    social_sentiment_score: float = 0.2
    contract_duration_months: int = 24
    position: str = "Midfielder"


@app.post("/api/predict")
async def predict_transfer_value(player: PlayerInput):
    model_path = ROOT_DIR / "models/xgboost_model.json"
    scaler_x_path = ROOT_DIR / "scaler_X.pkl"
    scaler_y_path = ROOT_DIR / "scaler_y.pkl"
    processed_path = ROOT_DIR / "transferiq_processed.csv"

    for p in [model_path, scaler_x_path, scaler_y_path, processed_path]:
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"Required file not found: {p.name}. Please run the full pipeline first.")

    try:
        bst = xgb.Booster()
        bst.load_model(str(model_path))
        scaler_X = joblib.load(str(scaler_x_path))
        scaler_y = joblib.load(str(scaler_y_path))

        # Build raw input
        raw = pd.DataFrame([{
            'performance_rating': player.performance_rating,
            'goals_assists': player.goals_assists,
            'minutes_played': player.minutes_played,
            'days_injured': player.days_injured,
            'social_sentiment_score': player.social_sentiment_score,
            'contract_duration_months': player.contract_duration_months,
            'perf_trend_3m': player.performance_rating,  # approximation
            'goals_trend_3m': player.goals_assists,
            'cumulative_days_injured': player.days_injured,
        }])

        scale_cols = [
            'performance_rating', 'goals_assists', 'minutes_played', 'days_injured',
            'social_sentiment_score', 'contract_duration_months',
            'perf_trend_3m', 'goals_trend_3m', 'cumulative_days_injured'
        ]
        raw[scale_cols] = scaler_X.transform(raw[scale_cols])

        # One-hot encode position to match training columns
        df_ref = pd.read_csv(str(processed_path), nrows=1)
        position_cols = [c for c in df_ref.columns if c.startswith('position_')]
        for col in position_cols:
            pos_name = col.replace('position_', '')
            raw[col] = 1.0 if player.position == pos_name else 0.0

        feature_cols = scale_cols + position_cols
        dmatrix = xgb.DMatrix(raw[feature_cols])
        
        # 1. Predict XGBoost
        pred_xgb_scaled = bst.predict(dmatrix)
        pred_xgb_value = scaler_y.inverse_transform(pred_xgb_scaled.reshape(-1, 1))[0][0]
        
        # 2. Predict LSTM
        lstm_pred_value = 0.0
        from tensorflow.keras.models import load_model
        model_m_path = ROOT_DIR / "models/multivariate_lstm.keras"
        if model_m_path.exists():
            model_m = load_model(str(model_m_path))
            raw['market_value_scaled'] = pred_xgb_scaled[0]
            multi_cols = ['performance_rating', 'goals_assists', 'days_injured', 'social_sentiment_score', 'market_value_scaled', 'perf_trend_3m']
            raw_multi = raw[multi_cols].values
            sequence = np.array([raw_multi[0], raw_multi[0], raw_multi[0]])
            sequence = np.expand_dims(sequence, axis=0)
            pred_lstm_scaled = model_m.predict(sequence, verbose=0)
            lstm_pred_value = scaler_y.inverse_transform(pred_lstm_scaled.reshape(-1, 1))[0][0]

        EUR_TO_INR = 90.5
        
        return {
            "xgboost": {
                "eur_millions": round(float(pred_xgb_value) / 1_000_000, 3),
                "inr_crores": round((float(pred_xgb_value) * EUR_TO_INR) / 1_00_00_000, 3),
                "inr_lakhs": round((float(pred_xgb_value) * EUR_TO_INR) / 1_00_000, 2),
            },
            "lstm": {
                "eur_millions": round(float(lstm_pred_value) / 1_000_000, 3) if lstm_pred_value else 0.0,
                "inr_crores": round((float(lstm_pred_value) * EUR_TO_INR) / 1_00_00_000, 3) if lstm_pred_value else 0.0,
                "inr_lakhs": round((float(lstm_pred_value) * EUR_TO_INR) / 1_00_000, 2) if lstm_pred_value else 0.0,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {traceback.format_exc()}")
