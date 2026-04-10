# ⚽ AI Transfer IQ — Dynamic Player Transfer Value Prediction

> An advanced, AI-powered platform designed to predict football player market values with industry-standard precision. By integrating performance statistics, temporal trends, injury histories, and real-time social sentiment data, the system provides a holistic and highly accurate valuation of athletes across global transfer windows.

---

## 🔍 Overview

AI Transfer IQ is an AI-driven system that predicts player transfer market values by utilizing a multi-stage analytical machine learning pipeline to generate actionable insights:

**Exploratory Data Analysis (EDA):** Provides insightful statistical visualizations of player attribute distributions, correlative behaviors, and general market value trends.

**Industry-Level Preprocessing:** Features advanced feature engineering (such as dynamic momentum and risk calculation) alongside Principal Component Analysis (PCA) based dimensionality reduction. This allows the models to handle complex, high-dimensional player data without succumbing to overfitting.

**Time-Series Forecasting (LSTM Architecture):**
- Univariate LSTM: Tracks and forecasts historical transfer value trends using sequence data.
- Multivariate LSTM: Integrates longitudinal performance metrics, injury occurrences, and public sentiment shifts over time to understand complex correlations.
- Encoder-Decoder LSTM: Enables multi-step forecasting to predict valuation trajectories multiple transfer windows into the future.

**Ensemble Modeling (XGBoost):** A highly robust regressor model that operates as the primary valuation engine. It combines all engineered features and time-series outputs to provide the final, definitive transfer value prediction with high predictive power.

**Cinematic Dashboard Frontend:** A premium, glassmorphism-themed web interface built natively in HTML, CSS, and Vanilla JavaScript. It provides real-time interaction and data visualization via a lightweight, high-performance architecture without relying on heavy external frontend frameworks.

---

## 🎬 Working Demo

> ### 👉 [Download Working Demo](https://drive.google.com/drive/folders/1Puylt7d-qedY45FCBKdcKKY0_M2rTBGc?usp=sharing)

### How to Run Locally

```bash
# 1. Start backend
python app.py

# 2. Start frontend (in a new terminal)
cd frontend
python -m http.server 3000

# 3. Open in browser
http://localhost:3000
```

### Sample Prediction

| Input | Value |
|-------|-------|
| Goals + Assists | 25 |
| Minutes Played | 2500 |
| Performance Rating | 0.85 |
| Days Injured | 5 |
| Contract Months Left | 36 |
| Sentiment Score | 0.80 |
| Previous Market Value | €45M |
| Position | Forward |

| Model | Predicted Value |
|-------|----------------|
| LSTM Model | €12.0M |
| XGBoost Model | €21.0M |
| **Ensemble (Best)** | **€16.5M** |

---

## 🏗️ Model Architecture

The intelligence system is built employing a strict, robust, and highly modular machine learning architecture consisting of distinct layers:

**Data Ingestion Layer:** Multi-source data unification encompassing raw performance statistics, physical injury timelines, and Natural Language Processing (VADER) sentiment derivation metrics from simulated social discussions.

**Preprocessing Layer:** Implementation of industry-standard scaling strategies combined dynamically with Principal Component Analysis to effectively isolate the most statistically significant mathematical variance across all columns.

**Forecasting Layer:** Advanced historical sequence modeling applied via recurrent neural networks (short-term memory architectures) comprehensively formatted for multivariate temporal analysis.

**Final Ensemble Layer:** Advanced gradient-boosted decision trees (the XGBoost algorithm) structurally refining the final statistical prediction by properly evaluating the output sequences of the LSTMs against the static, dimensionality-reduced data features.

```
Data Ingestion
├── Performance Data        (StatsBomb Open Data)
├── Market Value Data       (Transfermarkt)
├── Social Sentiment Data   (Twitter API + VADER NLP)
└── Injury History Data
        │
        ▼
Data Preprocessing & Feature Engineering
├── Missing value handling
├── Performance trend features
├── Injury risk metrics
├── Sentiment scoring (VADER NLP)
├── Scaling & normalization
└── One-hot encoding (position)
        │
        ▼
Model Training
├── LSTM (Univariate → Multivariate → Encoder-Decoder)
├── XGBoost (with GridSearchCV tuning)
└── Ensemble (LSTM + XGBoost average)
        │
        ▼
Flask REST API  ←→  Web Dashboard (HTML/CSS/JS)
```

---

## 🎨 Design Philosophy

The AI Transfer IQ user interface is deliberately engineered with a premium cinematic aesthetic, strictly focusing on maximizing native web rendering performance.

**Glassmorphism:** Structural implementation of sleek, translucent user interface layers utilizing native CSS background blurring effects for high-end immersion.

**Dynamic Animations:** Mathematically smooth state transitions and heavily hardware-accelerated CSS keyframe animations that yield an elite interaction experience.

**Functional Micro-interactions:** Highly responsive cursor hover states, algorithmic fluid value numeration logic, and natively integrated charting graphics displaying deterministic multi-step future market values seamlessly over time.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow / Keras (LSTM) |
| Machine Learning | XGBoost |
| Backend API | Flask + Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| NLP / Sentiment | VADER (NLTK) |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Matplotlib, Plotly |
| Development | Python 3.x, Google Colab, VS Code |

---

## 📁 Project Structure

```
AI_TransferIQ/
│
├── data/                          # Raw and processed datasets
│   ├── transferiq_dataset.csv
│   ├── transferiq_dataset_sentiment.csv
│   ├── transferiq_processed.csv
│   └── transferiq_features_final.csv
│
├── models/                        # Trained model files
│   ├── lstm_tuned_final.keras     # Final tuned LSTM model
│   └── xgboost_final.pkl          # Final XGBoost model
│
├── frontend/                      # Web dashboard
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── figures/                       # Generated plots and charts
│   ├── m7_loss_curve.png
│   ├── m7_actual_vs_predicted.png
│   ├── m7_model_comparison.png
│   ├── m7_player_trends.png
│   ├── m7_feature_importance.png
│   └── m7_sentiment_vs_predicted.png
│
├── app.py                         # Flask REST API backend
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---

## 🤖 Models

### LSTM Model
- **Architecture:** 2-layer LSTM (128 → 64 units) with Dropout regularization
- **Input:** Sequences of 3 timesteps × 27 features
- **Output:** Scaled market value (0–1)
- **Training:** 50 epochs, batch size 32, Adam optimizer, MSE loss

### XGBoost Model
- **Best Parameters:** `learning_rate=0.05`, `max_depth=3`, `n_estimators=200`
- **Input:** Flattened feature vector (81 values)
- **Output:** Scaled market value (0–1)
- **Tuning:** GridSearchCV with 3-fold cross validation (18 parameter combinations)

### Ensemble Model
- **Method:** Simple average of LSTM + XGBoost scaled predictions
- **Output:** Converted from scaled (0–1) back to real € values using MinMax inverse transform
- **Value Range:** €847,850 — €60,195,268

---

## 📊 Dataset

| Feature | Description |
|---------|-------------|
| `performance_rating` | Overall player performance score |
| `goals_assists` | Total goals and assists per season |
| `minutes_played` | Total playing time |
| `perf_trend_3m` | 3-month rolling performance trend |
| `perf_3m_avg` | 3-month performance average |
| `ga_per_minute` | Goals+assists per minute played |
| `lag_1`, `lag_2` | Previous timestep performance values |
| `days_injured` | Days missed due to injury |
| `cumulative_days_injured` | Total career injury days |
| `injury_risk` | Calculated injury risk score |
| `injury_impact` | Impact of injury on performance |
| `contract_duration_months` | Remaining contract length |
| `contract_urgency` | How soon contract expires (inverted) |
| `social_sentiment_score` | VADER NLP sentiment score |
| `sentiment_momentum` | Change in sentiment over time |
| `position_Forward` | One-hot encoded position |
| `position_Goalkeeper` | One-hot encoded position |
| `position_Midfielder` | One-hot encoded position |
| `market_value` | Historical transfer market value |
| `market_value_trend` | Direction of market value change |

**Dataset Summary:**
- Total records: 14,400
- Number of players: 100
- Date range: July 2018 — April 2024
- Market value range: **€847,850 — €60,195,268**
- Mean market value: **€22,648,065**
- Total features: **27**

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/AI_TransferIQ.git
cd AI_TransferIQ
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
flask
flask-cors
tensorflow
xgboost
scikit-learn
joblib
numpy
pandas
matplotlib
plotly
```

### Step 3 — Download Model Files
Download the trained models from the [Google Drive](https://drive.google.com/drive/folders/1Puylt7d-qedY45FCBKdcKKY0_M2rTBGc?usp=sharing) and place them in the `models/` folder:
```
models/
├── lstm_tuned_final.keras
└── xgboost_final.pkl
```

---

## 🚀 Running the App

### Terminal 1 — Start Backend
```bash
python app.py
```
Expected output:
```
✓ Models loaded!
* Running on http://127.0.0.1:5000
```

### Terminal 2 — Start Frontend
```bash
cd frontend
python -m http.server 3000
```

### Open in Browser
```
http://localhost:3000
```

---

## 🔌 API Reference

### Health Check
```
GET /health
```
**Response:**
```json
{
  "status": "online",
  "models": ["lstm", "xgboost"]
}
```

### Predict Transfer Value
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": [0.85, 0.5, 0.7, ...]
}
```
> 81 values total — 3 timesteps × 27 features

**Response:**
```json
{
  "status": "success",
  "lstm_euros": 12000000.00,
  "xgb_euros": 21000000.00,
  "ensemble_euros": 16500000.00,
  "lstm_formatted": "€12.0M",
  "xgb_formatted": "€21.0M",
  "ensemble_formatted": "€16.5M"
}
```

---

## 📈 Results

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| LSTM (base) | — | — | — |
| LSTM (tuned) | — | — | — |
| XGBoost (tuned) | — | — | — |
| **Ensemble** | **Best** | **Best** | **Best** |

> Update with actual values from `final_evaluation_report.csv` in the [Google Drive](https://drive.google.com/drive/folders/1Puylt7d-qedY45FCBKdcKKY0_M2rTBGc?usp=sharing)

---

## ⚠️ Current Limitations

- Football-specific dataset only — not generalized to other sports
- Static VADER sentiment scores — no live social media sources
- Trained on historical data only — no real-time market updates
- GPU recommended for model training
- Local run only — no cloud deployment currently
- Risk of overfitting on small player pool; requires broader validation
- No PWA or offline support

---

## 🔮 Future Work

- Integrate real-time data from live football APIs (FotMob, Opta)
- Add player age and nationality as additional features
- Deploy backend to cloud platform (Render / AWS / GCP)
- Deploy frontend to Netlify for public access
- Build mobile app using React Native or Flutter
- Add player-to-player comparison feature
- Improve model accuracy with a larger and more diverse training dataset
- Add model explainability using SHAP values

---

## 📄 License

This project is developed for educational and research purposes only.

---

> ⚽ Built with TensorFlow · XGBoost · Flask · JavaScript
