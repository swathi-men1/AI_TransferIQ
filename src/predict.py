"""
TransferIQ — Inference: Predict transfer value for a player
Usage:
    python predict.py --player "Cristiano Ronaldo"
    python predict.py --custom '{"Age":28,"OVA":91,"avg_sentiment":0.4,...}'
"""
import pickle, os, argparse, json
import numpy as np
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
PROC_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')


def load_ensemble():
    path = os.path.join(MODELS_DIR, 'weighted_ensemble.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def predict_player(player_name: str):
    df = pd.read_csv(os.path.join(PROC_DIR, 'featured_dataset_final.csv'))
    bundle = load_ensemble()
    features = bundle['features']
    scaler   = bundle['scaler']
    models   = bundle['models']
    weights  = bundle['weights']

    row = df[df['Name'].str.lower() == player_name.lower()]
    if row.empty:
        # Fuzzy match
        from difflib import get_close_matches
        match = get_close_matches(player_name, df['Name'].tolist(), n=1, cutoff=0.5)
        if match:
            row = df[df['Name'] == match[0]]
            print(f"Closest match: {match[0]}")
        else:
            print(f"Player '{player_name}' not found.")
            return None

    for col in features:
        row[col] = pd.to_numeric(row[col], errors='coerce')
    row = row.dropna(subset=features)

    X = scaler.transform(row[features].values)
    prf = models['rf'].predict(X)[0]
    pgb = models['gb'].predict(X)[0]
    prg = models['ridge'].predict(X)[0]
    wt  = weights['rf'] + weights['gb'] + weights['ridge']
    ens = (weights['rf']*prf + weights['gb']*pgb + weights['ridge']*prg) / wt

    predicted_val = np.expm1(ens)
    actual_val    = row['last_market_val'].values[0] if 'last_market_val' in row.columns else None

    print(f"\n{'='*45}")
    print(f"Player : {row['Name'].values[0]}")
    print(f"Age    : {row['Age'].values[0]}")
    print(f"OVA    : {row['OVA'].values[0]}")
    print(f"Club   : {row['Club'].values[0]}")
    print(f"{'='*45}")
    print(f"Predicted Transfer Value : €{predicted_val:,.0f}")
    if actual_val:
        print(f"Actual Market Value      : €{actual_val:,.0f}")
    print(f"{'='*45}")
    return predicted_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TransferIQ — Predict player transfer value')
    parser.add_argument('--player', type=str, default='Cristiano Ronaldo', help='Player name')
    args = parser.parse_args()
    predict_player(args.player)
