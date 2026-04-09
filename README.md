# TransferIQ — AI Player Transfer Value Prediction

> **An AI-powered web application that predicts football player transfer values using Machine Learning, NLP sentiment analysis, injury history, and historical market data.**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Screenshots](#screenshots)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Installation & Setup](#installation--setup)
7. [How to Run](#how-to-run)
8. [Model Details](#model-details)
9. [Dataset](#dataset)
10. [Results](#results)
11. [API Endpoints](#api-endpoints)
12. [Development Timeline](#development-timeline)

---

## Project Overview

TransferIQ is a complete end-to-end AI project developed as an internship project. It predicts the **transfer market value** of football players by combining:

- **Performance statistics** — OVA rating, attacking, skill, movement, physical attributes
- **Injury history** — days missed, games missed, risk score, availability percentage
- **Social media sentiment** — NLP-based positive/negative/neutral scoring (VADER)
- **Historical market data** — Transfermarkt values from 2009 to 2021
- **Age curve modeling** — peak age zone 23–28, potential score for young players

The final model is a **Stacking Ensemble** (Gradient Boosting + Extra Trees + Random Forest + Ridge Regression) achieving **R² = 0.761**, predicting **60.9% of players within 10% error** and **77.8% within 25% error**.

The web application uses **Ridge Regression** to serve real-time predictions with full player statistics visualization.

---

## Screenshots

### 1. Player Explorer

Browse all **1,034 players** with live search, multi-filter, and sort. Each card shows the AI-predicted transfer value, OVA rating, position, career stage, injury risk, and sentiment badge.

![Player Explorer](https://github.com/swathi-men1/AI_TransferIQ/blob/a3578318fbc7e1a8f7a0747c430cd7a5926fea20/outputs/screenshots/1.png)


**Features:**
- Search by player name, club, or nationality
- Filter by Position / Career Stage / Injury Risk / Sentiment
- Sort by Value / OVA / Age / Performance / Potential
- 44 pages of results with pagination

---

### 2. Player Detail Modal

Click any player card to open a detailed modal showing the AI-predicted transfer value vs actual market value, plus full stat bars.

![Player Modal](https://github.com/swathi-men1/AI_TransferIQ/blob/04941a0efce275d7a8c578bf8d617a599a03da98/outputs/screenshots/2.png)

**Shown for Bruno Fernandes:**
- AI Predicted: **€127.3M** vs Actual: **€90.0M** (+€37.3M over actual)
- OVA: 87/100 · Career Stage: Developing · Injury Risk: Low · Contract: 5 yrs
- Visual bars: OVA, Performance, Potential, Sentiment

---

### 3. Transfer Value Predictor

Configure a custom player profile using sliders and dropdowns. Ridge Regression model returns an instant prediction with statistics visualization.

![Transfer Value Predictor](https://github.com/swathi-men1/AI_TransferIQ/blob/f5c73dfd12453a515a252b6c5ab45b01d01f7cca/outputs/screenshots/3.png)

**Inputs used in example:**
- Position: Midfielder · Career Stage: Prime (26–29)
- Age: 26 · OVA: 75 · Performance: 48 · Potential: 70 · Contract: 4 yrs
- Injury Risk: Low · Sentiment: Neutral (0.00)

**Output:** €38.9M predicted transfer value with player stats visualization

---

## Features

### Player Explorer Tab

| Feature | Description |
|---|---|
| Live Search | Search by name, club, or nationality |
| Position Filter | Forward / Midfielder / Defender / Goalkeeper |
| Career Stage Filter | Youth / Developing / Prime / Experienced / Veteran |
| Injury Risk Filter | None / Low / Medium / High |
| Sentiment Filter | Positive / Neutral / Negative |
| Sort Options | Value · OVA · Age · Performance · Potential |
| Pagination | 44 pages, 24 players per page |
| Player Modal | Click any card for full stats + value breakdown |

### Transfer Value Predictor Tab

| Input | Type | Range |
|---|---|---|
| Position | Dropdown | Forward / Midfielder / Defender / Goalkeeper |
| Career Stage | Dropdown | Youth / Developing / Prime / Experienced / Veteran |
| Age | Slider | 16 – 39 years |
| OVA Rating | Slider | 57 – 92 |
| Performance Score | Slider | 30.0 – 62.0 |
| Potential Score | Slider | 57 – 92 |
| Contract Years | Slider | 0 – 10 years |
| Injury Risk | Dropdown | None / Low / Medium / High |
| Sentiment Category | Dropdown | Positive / Neutral / Negative |
| Sentiment Score | Slider | -0.70 to +0.90 |

**Output includes:**
- Predicted Transfer Value in €M 
- Top X% ranking among all players
- Statistics bars: OVA Rating, Performance, Potential, Availability, Sentiment, Contract Value
- Key Insights: age factor label, injury impact, sentiment effect

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Models | Scikit-learn (GradientBoosting, ExtraTrees, RandomForest, Ridge) |
| Deep Learning | NumPy (LSTM from scratch with full BPTT + gradient clipping) |
| NLP | VADER Sentiment Analysis |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web Backend | Flask 3.1 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Model Storage | Pickle (.pkl) |
| Scaling | RobustScaler (handles outliers) |

---

## Project Structure

```
TransferIQ/
├── src/                              ← Python source code (ML pipeline)
│   ├── main.py                       ← Full pipeline runner (5 steps)
│   ├── data_cleaning.py              ← Week 2: Cleaning & merging 4 sources
│   ├── feature_engineering.py        ← Week 3-4: 69 features engineered
│   ├── lstm_model.py                 ← Week 5: LSTM from scratch
│   ├── ensemble_model.py             ← Week 6-7: Ensemble + hyperparameter tuning
│   ├── best_model.py                 ← Best model v2 — train + CLI predict
│   └── predict.py                    ← Inference: predict by player name
│
├── webapp/                           ← Flask Web Application
│   ├── app.py                        ← Backend: loads pkl, serves API + prediction
│   ├── requirements.txt              ← Web app dependencies
│   └── templates/
│       └── index.html                ← Full frontend (HTML/CSS/JS — single file)
│
├── data/                             ← All datasets
│   ├── player.csv                    ← Raw: players, 23 FIFA attributes
│   ├── injury.csv                    ← Raw: injury records
│   ├── market_value.csv              ← Raw: market values (2009–2021)
│   ├── sentiment.csv                 ← Raw: NLP sentiment scores
│   ├── cleaned_dataset_final.csv     ← Cleaned + merged (Week 2 output)
│   ├── featured_dataset_final.csv    ← 59 engineered features (Week 3-4 output)
│   └── lstm_timeseries_dataset.csv   ← time-series sequences (Week 5 input)
│
├── models/                           ← Trained & saved ML models
│   ├── best_model_v2.pkl             ← BEST MODEL — Stacking Ensemble 
│   ├── weighted_ensemble.pkl         ← v1 Weighted Ensemble
│   ├── gradient_boosting.pkl         ← Individual Gradient Boosting
│   ├── random_forest.pkl             ← Individual Random Forest
│   ├── ridge_regression.pkl          ← Individual Ridge Regression
│   └── lstm_config.pkl               ← LSTM scalers + feature config
│
├── outputs/
│   ├── charts/                       ← 7 visualization charts
│   │   ├── chart_age_vs_value.png    ← Age vs market value scatter
│   │   ├── chart_feature_importance.png ← Top 10 features bar chart
│   │   ├── chart_model_comparison.png   ← R² comparison across all models
│   │   ├── chart_pred_vs_actual.png     ← Predicted vs actual scatter
│   │   ├── chart_sentiment_impact.png   ← Sentiment effect on value
│   │   ├── chart_value_by_position.png  ← Median value by position
│   │   └── chart_career_stage.png       ← Player count by career stage
│   ├── screenshots/                  ← Project screenshots
│   ├── final_predictions.csv         ← Test set predictions
│   ├── feature_importance_final.csv  ← Feature importance scores
│   ├── full_model_comparison.csv     ← All models R² / RMSE / MAE
│   └── ensemble_results_final.csv    ← Ensemble model results
│
├── requirements.txt                  ← Python dependencies
└── README.md                         ← This file
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
flask>=3.0.0
```

---

## How to Run

### Option 1 — Web Application (Recommended)

```bash
cd webapp
python app.py
```

Open your browser at: **http://localhost:5000**

The app loads `best_model_v2.pkl` at startup, pre-computes predictions for all players, and serves the full Player Explorer + Transfer Value Predictor.

---

### Option 2 — Command Line Prediction

```bash
# Predict by player name
python src/best_model.py predict "Bruno Fernandes"
python src/best_model.py predict "Casemiro"
python src/predict.py --player "Cristiano Ronaldo"
```

**Example output:**
```
Player: Bruno Fernandes
Predicted Transfer Value: €85,855,745
Actual Market Value:      €90,000,000
Error: 4.6%
```

---

### Option 3 — Retrain Everything from Scratch (Optional)

> All processed data and trained models are already included in the package. Use this only if you want to retrain from raw data.

```bash
python src/main.py
```

Runs all 5 pipeline stages:
1. Data Cleaning & Merging
2. Feature Engineering (59 features)
3. LSTM Model Training (NumPy)
4. Ensemble Model Training
5. Best Model v2 Training (Stacking Ensemble)

---

## Model Details

### Architecture

```
4 Raw Data Sources
        ↓
   Data Cleaning
        ↓
Feature Engineering (59 features)
        ↓
┌─────────────────────────────────────────────┐
│           Stacking Ensemble v2              │
│                                             │
│  Gradient    Extra      Random    Ridge     │
│  Boosting    Trees      Forest    Regression│
│  (600 trees) (500 trees)(400 trees)(α=1.0)  │
│       └──────────┴──────────┘               │
│            ↓ meta features ↓                │
│         Ridge Meta-Learner                  │
│       (5-fold cross-validation)             │
└─────────────────────────────────────────────┘
        ↓
Predicted Transfer Value (log scale → €)
```

**Web app uses Ridge Regression directly** for instant, explainable predictions.

### Feature Engineering Highlights

| Category | Examples |
|---|---|
| Age Features | `age_peak_diff`, `is_peak_age`, `potential_score`, `youth_flag` |
| Performance | `performance_score`, `physical_index`, `technical_index`, `injury_adj_performance` |
| Interaction | `ova_x_potential`, `perf_x_pot`, `contract_x_potential` |
| Injury | `availability_score`, `injury_risk_score`, `total_days_missed` |
| Market History | `log_hist_max`, `log_hist_mean`, `hist_growth` |
| Sentiment | `avg_sentiment`, `sentiment_performance_index` |

### Top 10 Features by Importance

| Rank | Feature | Importance | Meaning |
|---|---|---|---|
| 1 | `potential_score` | 40.8% | Young players' future value — dominant factor |
| 2 | `contract_value_proxy` | 4.1% | Contract years × OVA — seller leverage |
| 3 | `age_peak_diff` | 3.4% | Distance from peak age 26 |
| 4 | `injury_adj_performance` | 3.4% | Performance adjusted for availability |
| 5 | `total_injuries` | 3.2% | Career injury count |
| 6 | `avg_growth_rate` | 2.2% | Historical value growth trend |
| 7 | `sentiment_performance_index` | 1.6% | Sentiment-boosted performance score |
| 8 | `ova_x_potential` | ~1.5% | OVA × potential interaction |
| 9 | `perf_x_pot` | ~1.4% | Performance × potential |
| 10 | `hist_growth` | ~1.3% | Past seasons value trend |

### Why Not 90%+ Accuracy?

Transfer value is **40-50% determined by factors unavailable in public datasets:**

- Club's financial situation and budget constraints
- Agent fees and negotiation power (e.g. Mino Raiola-type agents)
- Buying club's specific tactical need
- Private medical reports
- Player's personal contract demands


---

## Dataset

| Source | Records | Description |
|---|---|---|
| FIFA Player Stats | players | OVA, position, physical, technical attributes |
| Transfermarkt | 6,043 records | Market values across 13 seasons (2009–2021) |
| Social Media NLP | 1,069 records | VADER sentiment scores for 407 players |
| Injury History | 9,553 records | Injury type, duration in days, games missed |

**After cleaning and merging:** players × 31 columns → feature engineered to **59 model features**, zero missing values.

**Fuzzy name matching** was used to join sentiment data to the player dataset (358 out of 407 players matched, up from 303 with exact matching).

---

## Results

### Model Performance Comparison

| Model | Week | R² Score | RMSE | Within 10% | Within 25% |
|---|---|---|---|---|---|
| Univariate LSTM | 5 | 0.160 | 1.308 | — | — |
| Multivariate LSTM | 5 | -0.179 | 1.550 | — | — |
| Gradient Boosting v1 | 5 | 0.484 | 1.065 | 9.2% | 22.7% |
| Weighted Ensemble v1 | 6 | 0.495 | 1.053 | 9.2% | 22.7% |
| **Stacking Ensemble v2** | **7** | **0.761** | **0.309** | **60.9%** | **77.8%** |

### Prediction Error Distribution (Best Model v2)

| Error Range | Players |
|---|---|
| Within 5% | 43.0% |
| Within 10% | 60.9% |
| Within 20% | 75.4% |
| Within 25% | 77.8% |
| Within 50% | 85.0% |

### Sample Predictions

| Player | Club | Actual | Predicted | Error |
|---|---|---|---|---|
| João Félix | Atlético Madrid | €70M | €71.3M | 1.8% |
| Casemiro | Manchester United | €70M | €68.0M | 2.8% |
| Bernardo Silva | Manchester City | €70M | €67.8M | 3.1% |
| Bruno Fernandes | Manchester United | €90M | €85.9M | 4.6% |
| Cristiano Ronaldo | Juventus | €45M | €37.9M | 15.8% |

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Main web application |
| `/api/players` | GET | Player list with filters + pagination |
| `/api/predict` | POST | Predict from custom player profile |
| `/api/stats` | GET | Dataset and model statistics |

### `/api/players` — Query Parameters

```
GET /api/players?position=midfielder&stage=prime&risk=low&sort=predicted_value&page=1&per=24
```

| Parameter | Values | Default |
|---|---|---|
| `q` | any text | — |
| `position` | forward / midfielder / defender / goalkeeper | all |
| `stage` | youth / developing / prime / experienced / veteran | all |
| `risk` | none / low / medium / high | all |
| `sentiment` | positive / neutral / negative | all |
| `sort` | predicted_value / ova / age / performance_score / potential_score | predicted_value |
| `page` | integer | 1 |
| `per` | integer | 24 |

### `/api/predict` — Request & Response

**Request:**
```json
{
  "age": 26,
  "ova": 75,
  "performance_score": 48,
  "potential_score": 70,
  "position": "midfielder",
  "injury_risk": "low",
  "sentiment": 0.0,
  "contract_years": 4
}
```

**Response:**
```json
{
  "predicted_value": 38900000,
  "model": "Ridge Regression",
  "percentile": 72.5,
  "stats": {
    "OVA Rating": 51,
    "Performance": 54,
    "Potential": 37,
    "Availability": 99,
    "Sentiment": 50,
    "Contract Value": 40
  },
  "labels": {
    "age": "Peak age (26) — max market premium",
    "age_color": "positive",
    "injury": "Low risk — minor injuries, 99% availability",
    "sentiment": "Neutral — no significant media premium"
  }
}
```

---

## Development Timeline

| Week | Milestone | Deliverable |
|---|---|---|
| Week 1 | Data Collection | 4 raw CSV files (player, injury, market, sentiment) |
| Week 2 | Data Cleaning & Merging | `cleaned_dataset_final.csv` — players, zero missing values |
| Week 3–4 | Feature Engineering | 69 features → `featured_dataset_final.csv` |
| Week 5 | LSTM from Scratch | NumPy BPTT implementation, R²=0.16 (univariate) |
| Week 6 | Ensemble Models | Stacking ensemble, R²=0.495 |
| Week 7 | Hyperparameter Tuning | Best model v2, R²=0.761 |
| Week 8 | Web App + Visualization + Report | Full project delivery |

---

## License

This project was developed as an internship project for educational and research purposes.

---

*Built with Python · Scikit-learn · Flask · NumPy · Pandas · Matplotlib · VADER NLP*
