<div align="center">

# TransferIQ

### Cricket Player Auction Value Prediction Engine

**Ensemble ML-powered auction valuation using LSTM + XGBoost**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-1A8FFF.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Overview

TransferIQ predicts cricket player auction values by combining two powerful machine learning models into a single ensemble pipeline:

- **LSTM Neural Network (60% weight)** — captures temporal performance trends
- **XGBoost Gradient Boosting (40% weight)** — models non-linear feature interactions

The result is a robust, real-time prediction engine delivered through a clean FastAPI backend with an interactive web dashboard.

### Working Demo

Download the complete working demo files (models, data, server, and frontend) from Google Drive:

[**Download Working Demo**](https://drive.google.com/drive/folders/1TplzK19dHSjUiH4WE1KX9C9BOPOLVgwR?usp=sharing)

---

## Features

| Feature | Description |
|---|---|
| Ensemble Prediction | Weighted combination of LSTM and XGBoost for accurate valuations |
| Confidence Intervals | Every prediction includes upper/lower bounds (±15%) |
| Performance Breakdown | Scores for Performance, Sentiment Impact, Market Demand, and Injury Risk |
| Role-Based Adjustments | Custom multipliers for Batsman, Bowler, All-Rounder, and Wicket-Keeper |
| Interactive Dashboard | Responsive dark-themed UI with animated value displays |
| Health Monitoring | Built-in health check endpoint for system status |
| Fallback Mechanism | Graceful degradation if models fail to load |

---

## Tech Stack

```
Backend    — FastAPI + Uvicorn (ASGI)
ML Models  — TensorFlow/Keras (LSTM) · XGBoost (Gradient Boosting)
Data       — Pandas · NumPy · Scikit-learn
Frontend   — Vanilla HTML/CSS/JS (served by FastAPI)
```

---

## Project Structure

```
transferiqapp/
├── server.py                        # FastAPI server + frontend UI
├── requirements.txt                 # Python dependencies
│
├── TransferIQ/
│   ├── README.md                    # Quick start guide
│   ├── requirements.txt             # Package dependencies
│   ├── data/
│   │   ├── cleaned_data.csv         # Processed player statistics
│   │   └── players.csv              # Raw player data
│   ├── model/
│   │   ├── __init__.py              # Package init
│   │   ├── lstm_model.py            # LSTM model & prediction
│   │   ├── xgboost_model.py         # XGBoost model & prediction
│   │   ├── ensemble_model.py        # Ensemble combination logic
│   │   ├── data_utils.py            # Data loading & preprocessing
│   │   └── model_utils.py           # Model training & evaluation
│   └── notebooks/
│       └── training.ipynb           # Model development notebook
│
└── transfer_iq_app/
    └── app.py                       # Alternative Streamlit frontend
```

---

## Quick Start

### Prerequisites

- Python 3.8+
- pip
- 2 GB RAM minimum

### Install & Run

```bash
# 1. Navigate into the project
cd transferiqapp

# 2. Create & activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r TransferIQ/requirements.txt

# 4. Launch the server
python server.py
```

Once running, open **http://localhost:8000** in your browser.

> **Expected output:**
> ```
> ✅ Models loaded successfully
> INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
> ```

---

## API Reference

### `GET /health`

Returns system status and model availability.

```json
{
  "status": "healthy",
  "models_loaded": true
}
```

---

### `POST /api/predict`

Predicts auction value for a given player profile.

**Request body:**

| Parameter | Type | Range | Description |
|---|---|---|---|
| `name` | string | — | Player name |
| `role` | string | `Batsman`, `Bowler`, `All-Rounder`, `WK` | Playing role |
| `age` | int | 17–45 | Age in years |
| `bat_avg` | float | 0–100+ | Batting average |
| `strike_rate` | float | 80–200+ | Runs per 100 balls |
| `matches` | int | 0+ | Total matches played |
| `economy` | float | 5–12+ | Bowling economy rate |
| `wickets` | int | 0+ | Career wickets |
| `injuries` | int | 0+ | Recent injury count |
| `sentiment_score` | float | -1 to 1 | Public sentiment |
| `base_price` | float | 50+ | Base price in lakhs |

**Example request:**

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Virat Kohli",
    "role": "All-Rounder",
    "age": 35,
    "bat_avg": 45.5,
    "strike_rate": 132.4,
    "matches": 220,
    "economy": 7.8,
    "wickets": 12,
    "injuries": 1,
    "sentiment_score": 0.8,
    "base_price": 1000
  }'
```

**Response:**

```json
{
  "predicted_value": 2392.87,
  "confidence_low": 2033.94,
  "confidence_high": 2751.80,
  "performance_score": 45.3,
  "sentiment_impact": 90.0,
  "market_demand": 85.0,
  "injury_risk": 22.0,
  "model_used": "Ensemble (LSTM + XGBoost) - Real Models",
  "message": "Prediction successful"
}
```

> Auto-generated interactive docs are available at **http://localhost:8000/docs**.

---

## Model Architecture

### LSTM Neural Network

Captures temporal patterns in player performance over time.

```
Input → LSTM(50, ReLU) → Dropout(0.2) → LSTM(50, ReLU) → Dropout(0.2) → Dense(1)
```

| Config | Value |
|---|---|
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Regularization | 20% Dropout per LSTM layer |

**Key features:** batting average multiplier, strike rate impact, wicket multiplier, experience factor, sentiment score, age penalty (after 32), and role-based adjustment.

---

### XGBoost Regressor

Models non-linear feature interactions with hyperparameter tuning.

```
Hyperparameter search via GridSearchCV (3-fold CV)
```

| Hyperparameter | Search Space |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 3, 5, 7 |
| `learning_rate` | 0.01, 0.1, 0.2 |
| `subsample` | 0.8, 0.9, 1.0 |
| `colsample_bytree` | 0.8, 0.9, 1.0 |

**Key features:** batting score, consistency score, bowling effectiveness, economy rate score, overall performance index, sentiment multiplier, experience factor, age factor, role multiplier, and injury impact.

---

### Ensemble Combination

```
Final Prediction = (0.6 × LSTM) + (0.4 × XGBoost)
Confidence Band   = ±15% of final prediction
```

The 60/40 split reflects the LSTM's strength in temporal trend detection while leveraging XGBoost's ability to capture complex feature interactions, resulting in improved generalization over either model alone.

---

## Configuration

### Role Multipliers

| Role | LSTM | XGBoost |
|---|---|---|
| Batsman | 1.20× | 1.25× |
| Bowler | 1.10× | 1.15× |
| All-Rounder | 1.30× | 1.35× |
| Wicket-Keeper | 1.15× | 1.20× |

### Age Penalty (applied after age 32)

```
age_multiplier = max(0.3, 1.0 - (age - 32) × 0.05)
```

### Sentiment Mapping

```
sentiment_multiplier = 0.8 + (sentiment_score × 0.6)
→ range: 0.8 (negative) to 1.4 (positive)
```

### Market Demand by Role

| Role | Demand Score |
|---|---|
| Batsman | 60 |
| Bowler | 68 |
| All-Rounder | 82 |
| Wicket-Keeper | 75 |

---

## Data Flow

```
User Input (HTML Form)
       │
       ▼
  JavaScript fetch() ──POST──► FastAPI /api/predict
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                    LSTM Model           XGBoost Model
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                          Ensemble (0.6 / 0.4)
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                  Confidence Interval    Score Breakdown
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                            JSON Response
                                     │
                                     ▼
                       Dashboard UI Update (animated)
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| **Port 8000 in use** | Kill the process: `lsof -i :8000` then `kill <PID>` |
| **Models not loading** | Verify `tensorflow` and `xgboost` are installed: `pip list \| grep -E "tensorflow\|xgboost"` |
| **Prediction returns 0** | Ensure `base_price > 0` in the request body |
| **CORS errors** | Confirm the server is running with CORS middleware enabled |

> If model imports fail, the application automatically falls back to lightweight prediction functions based on feature calculations.

---

## Retraining Models

1. Place new training data in `TransferIQ/data/`
2. Open `TransferIQ/notebooks/training.ipynb`
3. Run the training pipeline
4. Export updated model files
5. Restart the server: `python server.py`

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

To modify prediction logic, edit the relevant model files:

- **LSTM** → `TransferIQ/model/lstm_model.py` — `predict_lstm()`
- **XGBoost** → `TransferIQ/model/xgboost_model.py` — `predict_xgboost()`

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| FastAPI | ≥ 0.104.0 | Web framework |
| Uvicorn | ≥ 0.24.0 | ASGI server |
| TensorFlow | ≥ 2.12.0 | LSTM neural network |
| XGBoost | ≥ 1.7.0 | Gradient boosting |
| Scikit-learn | ≥ 1.2.0 | ML utilities |
| Pandas | ≥ 1.5.0 | Data manipulation |
| NumPy | ≥ 1.23.0 | Numerical computing |

---

## License

This project is provided as-is for educational and development purposes.

---

<div align="center">

**Built with FastAPI, TensorFlow & XGBoost**

</div>
