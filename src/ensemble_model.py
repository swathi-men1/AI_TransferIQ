"""
TransferIQ — Week 6-7: Ensemble Model + Hyperparameter Tuning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os

PROC_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
OUT_DIR    = os.path.join(os.path.dirname(__file__), '..', 'outputs')

FEATURE_COLS = [
    'Age', 'OVA', 'age_peak_diff', 'is_peak_age', 'age_squared',
    'performance_score', 'physical_index', 'technical_index',
    'Attacking', 'Skill', 'Movement', 'Acceleration', 'Stamina', 'Strength',
    'Short Passing', 'Shot Power', 'Heading Accuracy', 'Jumping',
    'total_injuries', 'total_days_missed', 'avg_days_per_injury',
    'injury_risk_score', 'availability_score',
    'avg_sentiment', 'sentiment_count', 'sentiment_performance_index',
    'Contract_Years', 'contract_value_proxy', 'Height_cm', 'Weight_kg', 'is_left_footed',
    'pos_defender', 'pos_forward', 'pos_goalkeeper', 'pos_midfielder',
    'num_seasons', 'avg_growth_rate', 'value_volatility',
    'OVA_norm', 'performance_score_norm', 'Age_norm', 'avg_sentiment_norm',
    'injury_adj_performance', 'potential_score'
]


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:35s} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {'model': name, 'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'R2': round(r2, 4)}


def run_ensemble():
    print("="*60)
    print("TRANSFERIQ — ENSEMBLE MODEL")
    print("="*60)

    df = pd.read_csv(os.path.join(PROC_DIR, 'featured_dataset_final.csv'))
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=FEATURE_COLS + ['log_market_val'])

    X = df[FEATURE_COLS].values
    y = df['log_market_val'].values
    names = df['Name'].values

    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte, ntr, nte = train_test_split(X_sc, y, names, test_size=0.2, random_state=42)
    print(f"Train: {len(Xtr)} | Test: {len(Xte)}\n")

    # Best hyperparameters (from Week 7 tuning)
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=400, max_depth=5, learning_rate=0.04, subsample=0.8, random_state=42)
    rg = Ridge(alpha=0.5)

    print("Training models...")
    rf.fit(Xtr, ytr); gb.fit(Xtr, ytr); rg.fit(Xtr, ytr)

    prf = rf.predict(Xte); pgb = gb.predict(Xte); prg = rg.predict(Xte)
    r2_rf = r2_score(yte, prf); r2_gb = r2_score(yte, pgb); r2_rg = r2_score(yte, prg)

    print("\nIndividual model results:")
    rows = []
    rows.append(evaluate("Random Forest",    yte, prf))
    rows.append(evaluate("Gradient Boosting",yte, pgb))
    rows.append(evaluate("Ridge Regression", yte, prg))

    # Weighted ensemble
    wt  = r2_rf + r2_gb + r2_rg
    ens = (r2_rf * prf + r2_gb * pgb + r2_rg * prg) / wt
    print("\nEnsemble results:")
    rows.append(evaluate("Weighted Ensemble", yte, ens))

    # Feature importance
    fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': rf.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)

    # Predictions output
    pred_df = pd.DataFrame({
        'Player': nte, 'Actual_log': yte.round(4),
        'Pred_GB': pgb.round(4), 'Pred_Ensemble': ens.round(4),
        'Error': (ens - yte).round(4)
    })

    # Save models as pkl
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, 'random_forest.pkl'), 'wb') as f:
        pickle.dump({'model': rf, 'scaler': scaler, 'features': FEATURE_COLS}, f)
    with open(os.path.join(MODELS_DIR, 'gradient_boosting.pkl'), 'wb') as f:
        pickle.dump({'model': gb, 'scaler': scaler, 'features': FEATURE_COLS}, f)
    with open(os.path.join(MODELS_DIR, 'ridge_regression.pkl'), 'wb') as f:
        pickle.dump({'model': rg, 'scaler': scaler, 'features': FEATURE_COLS}, f)
    with open(os.path.join(MODELS_DIR, 'weighted_ensemble.pkl'), 'wb') as f:
        pickle.dump({'models': {'rf': rf, 'gb': gb, 'ridge': rg},
                     'weights': {'rf': r2_rf, 'gb': r2_gb, 'ridge': r2_rg},
                     'scaler': scaler, 'features': FEATURE_COLS}, f)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'tuned_model_results.csv'), index=False)
    fi.to_csv(os.path.join(OUT_DIR, 'feature_importance_final.csv'), index=False)
    pred_df.to_csv(os.path.join(OUT_DIR, 'final_predictions.csv'), index=False)

    print("\n✓ Models saved to models/ | Results saved to outputs/")
    return rf, gb, rg

if __name__ == '__main__':
    run_ensemble()
