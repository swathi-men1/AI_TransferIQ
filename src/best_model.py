"""
TransferIQ — Best Model v2 (R² = 0.761)
Uses stacking ensemble: GB + ExtraTrees + RandomForest + Ridge
With 59 features including interaction features and historical market data.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
import pickle, os, warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'featured_dataset_final.csv')
MKT_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'market_value.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model_v2.pkl')

BASE_FEATURES = [
    'Age','OVA','age_peak_diff','is_peak_age','age_squared',
    'performance_score','physical_index','technical_index',
    'Attacking','Skill','Movement','Acceleration','Stamina','Strength',
    'Short Passing','Shot Power','Heading Accuracy','Jumping',
    'total_injuries','total_days_missed','avg_days_per_injury',
    'injury_risk_score','availability_score',
    'avg_sentiment','sentiment_count','sentiment_performance_index',
    'Contract_Years','contract_value_proxy','Height_cm','Weight_kg','is_left_footed',
    'pos_defender','pos_forward','pos_goalkeeper','pos_midfielder',
    'num_seasons','avg_growth_rate','value_volatility',
    'OVA_norm','performance_score_norm','Age_norm','avg_sentiment_norm',
    'injury_adj_performance','potential_score',
]

INTERACTION_FEATURES = [
    'ova_x_potential','ova_x_peak','ova_squared','youth_flag',
    'perf_x_avail','sent_x_ova','contract_x_ova','contract_x_potential',
    'inj_x_age','perf_x_pot','ova_x_contract',
]

HIST_FEATURES = [
    'log_hist_max','log_hist_mean','hist_growth','hist_seasons',
]

ALL_FEATURES = BASE_FEATURES + INTERACTION_FEATURES + HIST_FEATURES


def add_interaction_features(df):
    """Add interaction features — no data leakage."""
    for col in ['Attacking','Skill','Movement']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['ova_x_potential']      = df['OVA'] * df['potential_score']
    df['ova_x_peak']           = df['OVA'] * df['is_peak_age']
    df['ova_squared']          = df['OVA'] ** 2
    df['youth_flag']           = (df['Age'] < 23).astype(int) * df['OVA']
    df['perf_x_avail']         = df['performance_score'] * df['availability_score']
    df['sent_x_ova']           = df['avg_sentiment'] * df['OVA']
    df['contract_x_ova']       = df['Contract_Years'] * df['OVA']
    df['contract_x_potential'] = df['Contract_Years'] * df['potential_score']
    df['inj_x_age']            = df['total_days_missed'] / (df['Age'] + 1)
    df['perf_x_pot']           = df['performance_score'] * df['potential_score']
    df['ova_x_contract']       = df['OVA'] * np.log1p(df['Contract_Years'])
    return df


def add_hist_market_features(df, market):
    """Add historical market features (previous seasons only, not current value)."""
    mkt = market.drop_duplicates(subset=['ID','season']).sort_values(['ID','season'])

    def safe_hist(grp):
        if len(grp) <= 1:
            return pd.Series({'hist_max': np.nan, 'hist_mean': np.nan,
                              'hist_growth': np.nan, 'hist_seasons': len(grp)})
        hist = grp.iloc[:-1]
        return pd.Series({
            'hist_max':     hist['market_val'].max(),
            'hist_mean':    hist['market_val'].mean(),
            'hist_growth':  hist['market_val'].pct_change().mean(),
            'hist_seasons': len(hist),
        })

    hist_feats = mkt.groupby('ID').apply(safe_hist).reset_index()
    hist_feats['log_hist_max']  = np.log1p(hist_feats['hist_max'].fillna(0))
    hist_feats['log_hist_mean'] = np.log1p(hist_feats['hist_mean'].fillna(0))
    hist_feats['hist_growth']   = hist_feats['hist_growth'].fillna(0).clip(-1, 5)

    df = df.merge(
        hist_feats[['ID','log_hist_max','log_hist_mean','hist_growth','hist_seasons']],
        on='ID', how='left'
    )
    for col in HIST_FEATURES:
        df[col] = df[col].fillna(df[col].median())
    return df


def prepare_data():
    """Load and prepare full feature matrix."""
    df     = pd.read_csv(DATA_PATH)
    market = pd.read_csv(MKT_PATH)
    df = add_interaction_features(df)
    df = add_hist_market_features(df, market)
    for col in ALL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=ALL_FEATURES + ['log_market_val'])
    return df


def train():
    """Train and save best_model_v2.pkl."""
    print("="*55)
    print("TransferIQ — Best Model v2 Training")
    print("="*55)

    df = prepare_data()
    X  = df[ALL_FEATURES].values
    y  = df['log_market_val'].values
    print(f"Dataset: {X.shape[0]} players x {X.shape[1]} features")

    scaler = RobustScaler()
    X_sc   = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(X_sc, y, test_size=0.2, random_state=42)

    # --- Individual models ---
    gb  = GradientBoostingRegressor(n_estimators=600, max_depth=5, learning_rate=0.03,
                                     subsample=0.8, min_samples_leaf=3, max_features=0.8, random_state=42)
    et  = ExtraTreesRegressor(n_estimators=500, max_depth=18, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf  = RandomForestRegressor(n_estimators=400, max_depth=15, min_samples_leaf=2,
                                 random_state=42, n_jobs=-1)
    rg  = Ridge(alpha=1.0)

    print("Training models...")
    for model, name in [(gb,'GradientBoosting'),(et,'ExtraTrees'),(rf,'RandomForest'),(rg,'Ridge')]:
        model.fit(Xtr, ytr)
        r2 = r2_score(yte, model.predict(Xte))
        print(f"  {name:25s} R²={r2:.4f}")

    # --- Stacking meta-learner ---
    print("Building stacking meta-learner...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    meta_tr = np.zeros((len(Xtr), 4))
    for _, (ti, vi) in enumerate(kf.split(Xtr)):
        for j, Model in enumerate([
            GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42),
            ExtraTreesRegressor(n_estimators=200, max_depth=18, random_state=42, n_jobs=-1),
            RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
            Ridge(alpha=1.0),
        ]):
            Model.fit(Xtr[ti], ytr[ti])
            meta_tr[vi, j] = Model.predict(Xtr[vi])

    meta_te = np.column_stack([gb.predict(Xte), et.predict(Xte),
                                rf.predict(Xte), rg.predict(Xte)])
    meta_lr = Ridge(alpha=0.5)
    meta_lr.fit(meta_tr, ytr)
    stack_pred = meta_lr.predict(meta_te)

    r2_final   = r2_score(yte, stack_pred)
    rmse_final = np.sqrt(mean_squared_error(yte, stack_pred))
    mae_final  = mean_absolute_error(yte, stack_pred)

    errs = np.abs(np.expm1(yte) - np.expm1(stack_pred)) / (np.expm1(yte) + 1) * 100
    print(f"\nFinal Stacking Ensemble:")
    print(f"  R² = {r2_final:.4f}")
    print(f"  RMSE = {rmse_final:.4f}")
    print(f"  Within 10%: {(errs<=10).mean()*100:.1f}%")
    print(f"  Within 25%: {(errs<=25).mean()*100:.1f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({
            'models':       {'gb': gb, 'et': et, 'rf': rf, 'ridge': rg},
            'meta_learner': meta_lr,
            'scaler':       scaler,
            'features':     ALL_FEATURES,
            'r2_test':      r2_final,
            'version':      'v2'
        }, f)
    print(f"\nSaved: {MODEL_PATH}")
    return r2_final


def predict(player_name=None, custom_features=None):
    """
    Predict transfer value.
    player_name: str — lookup from dataset
    custom_features: dict — manual feature values
    """
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    models = bundle['models']
    ml     = bundle['meta_learner']
    scaler = bundle['scaler']
    feats  = bundle['features']

    df     = prepare_data()

    if player_name:
        from difflib import get_close_matches
        match = get_close_matches(player_name, df['Name'].tolist(), n=1, cutoff=0.5)
        if not match:
            print(f"Player '{player_name}' not found.")
            return None
        row = df[df['Name'] == match[0]]
        print(f"Player: {match[0]}")
    elif custom_features:
        row = pd.DataFrame([custom_features])
        for col in feats:
            if col not in row:
                row[col] = df[col].median()
    else:
        print("Provide player_name or custom_features.")
        return None

    X = scaler.transform(row[feats].values)
    meta = np.column_stack([models['gb'].predict(X), models['et'].predict(X),
                             models['rf'].predict(X), models['ridge'].predict(X)])
    pred = np.expm1(ml.predict(meta)[0])

    actual = row['last_market_val'].values[0] if 'last_market_val' in row.columns else None
    print(f"Predicted Transfer Value: €{pred:,.0f}")
    if actual:
        err = abs(pred - actual) / actual * 100
        print(f"Actual Market Value:      €{actual:,.0f}")
        print(f"Error: {err:.1f}%")
    return pred


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        name = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else 'Cristiano Ronaldo'
        predict(name)
    else:
        train()
