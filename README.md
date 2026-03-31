# TransferIQ — AI Player Transfer Value Prediction

## Quick Start

```bash
pip install -r requirements.txt

# Train all models (full pipeline)
python src/main.py

# Predict a single player
python src/best_model.py predict "Bruno Fernandes"
python src/predict.py --player "Casemiro"

# Run web app
cd webapp
python app.py
# Open: http://localhost:5000
```

## Project Structure

```
TransferIQ/
├── data/
│   ├── raw/                        # Original 4 CSV datasets
│   │   ├── player.csv              # 1,034 players, FIFA stats
│   │   ├── market_value.csv        # 6,043 records, 2009-2021
│   │   ├── injury.csv              # 9,553 injury records
│   │   └── sentiment.csv          # NLP sentiment scores
│   └── processed/                  # Cleaned & feature-engineered
│       ├── cleaned_dataset_final.csv
│       ├── featured_dataset_final.csv
│       └── lstm_timeseries_dataset.csv
│
├── models/                         # Saved pkl model files
│   ├── best_model_v2.pkl           ← BEST MODEL (R²=0.761)
│   ├── weighted_ensemble.pkl       # v1 ensemble (R²=0.495)
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── ridge_regression.pkl
│   └── lstm_config.pkl
│
├── src/                            # Source code
│   ├── main.py                     # Full pipeline runner
│   ├── data_cleaning.py            # Week 2: cleaning & merging
│   ├── feature_engineering.py     # Week 3-4: 69 features
│   ├── lstm_model.py               # Week 5: LSTM from scratch (NumPy)
│   ├── ensemble_model.py           # Week 6-7: ensemble + tuning
│   ├── best_model.py               # Best model v2 (R²=0.761)
│   └── predict.py                  # Inference script
│
├── webapp/                         # Flask web application
│   ├── app.py                      # Flask backend (uses best_model_v2.pkl)
│   ├── requirements.txt
│   ├── models/
│   │   └── best_model_v2.pkl
│   ├── data/
│   │   └── featured_dataset_final.csv
│   └── templates/
│       └── index.html              # Full frontend
│
├── outputs/                        # Results, charts, reports
│   ├── charts/                     # 7 visualization charts
│   ├── final_predictions.csv
│   ├── feature_importance_final.csv
│   └── full_model_comparison.csv
│
├── TransferIQ_Presentation.pptx   # 10-slide deck
├── TransferIQ_Report.docx         # Full project report
├── requirements.txt
└── README.md
```

## Model Results

| Model | R² | Within 10% | Within 25% |
|---|---|---|---|
| Univariate LSTM | 0.160 | — | — |
| Gradient Boosting v1 | 0.484 | 9.2% | 22.7% |
| Weighted Ensemble v1 | 0.495 | 9.2% | 22.7% |
| **Stacking Ensemble v2** | **0.761** | **60.9%** | **77.8%** |




## Tech Stack

Python 3.12 · NumPy · Scikit-learn · Pandas · Flask · Matplotlib
