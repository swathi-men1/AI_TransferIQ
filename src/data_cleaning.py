"""
TransferIQ — Week 2: Data Cleaning & Preprocessing
"""
import pandas as pd
import numpy as np
from difflib import get_close_matches
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')

def height_to_cm(h):
    try:
        h = str(h).replace('"', '').replace("'", " ").strip()
        parts = h.split()
        return round(int(parts[0]) * 30.48 + int(parts[1]) * 2.54, 1)
    except:
        return np.nan

def weight_to_kg(w):
    try:
        return round(float(str(w).replace('lbs', '')) * 0.453592, 1)
    except:
        return np.nan

def contract_years(c):
    try:
        parts = str(c).split('~')
        return int(parts[1].strip()) - int(parts[0].strip())
    except:
        return np.nan

def clean_player(df):
    df['Height_cm'] = df['Height'].apply(height_to_cm)
    df['Weight_kg'] = df['Weight'].apply(weight_to_kg)
    df['Contract_Years'] = df['Contract'].apply(contract_years)
    df['Club'] = df['Club'].fillna('Unknown')
    df['Jumping'] = df['Jumping'].fillna(df['Jumping'].median())
    df = df.drop(columns=['Height', 'Weight', 'Team & Contract', 'Joined', 'Contract'])
    return df

def clean_injury(df):
    df['from_date'] = pd.to_datetime(df['from_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['end_date'] = df.apply(
        lambda r: r['from_date'] + pd.Timedelta(days=int(r['days_missed']))
        if pd.isnull(r['end_date']) else r['end_date'], axis=1)
    injury_agg = df.groupby('ID').agg(
        total_injuries=('injury_reason', 'count'),
        total_days_missed=('days_missed', 'sum'),
        total_games_missed=('games_missed', 'sum'),
        avg_days_per_injury=('days_missed', 'mean')
    ).reset_index()
    injury_agg['avg_days_per_injury'] = injury_agg['avg_days_per_injury'].round(1)
    injury_agg['injury_risk_score'] = (injury_agg['total_days_missed'] / injury_agg['total_injuries']).round(1)
    return injury_agg

def clean_sentiment(df, player_names):
    sentiment_agg = df.groupby('Name').agg(
        avg_sentiment=('Sentiment_Score', 'mean'),
        sentiment_count=('Sentiment_Score', 'count')
    ).reset_index()
    sentiment_agg['avg_sentiment'] = sentiment_agg['avg_sentiment'].round(3)
    sentiment_agg['sentiment_label'] = sentiment_agg['avg_sentiment'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))

    def fuzzy_match(name, choices, cutoff=0.7):
        matches = get_close_matches(name, choices, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    sentiment_agg['matched_name'] = sentiment_agg['Name'].apply(lambda n: fuzzy_match(n, player_names))
    return sentiment_agg

def run_cleaning():
    print("Loading raw data...")
    player   = pd.read_csv(os.path.join(RAW_DIR, 'player.csv'))
    injury   = pd.read_csv(os.path.join(RAW_DIR, 'injury.csv'))
    market   = pd.read_csv(os.path.join(RAW_DIR, 'market_value.csv'))
    sentiment = pd.read_csv(os.path.join(RAW_DIR, 'sentiment.csv'))

    print("Cleaning player data...")
    player_clean = clean_player(player.copy())

    print("Cleaning injury data...")
    injury_agg = clean_injury(injury.copy())

    print("Processing sentiment data...")
    player_names = list(player_clean['Name'])
    sentiment_agg = clean_sentiment(sentiment.copy(), player_names)
    name_to_id = dict(zip(player_clean['Name'], player_clean['ID']))
    sentiment_agg['ID'] = sentiment_agg['matched_name'].map(name_to_id)

    print("Merging datasets...")
    df = player_clean.merge(injury_agg, on='ID', how='left')
    for col in ['total_injuries', 'total_days_missed', 'total_games_missed', 'avg_days_per_injury', 'injury_risk_score']:
        df[col] = df[col].fillna(0)

    market_latest = market.sort_values('season').groupby('ID').last().reset_index()[['ID', 'market_val', 'season']]
    market_latest.columns = ['ID', 'latest_market_val', 'latest_season']
    df = df.merge(market_latest, on='ID', how='left')

    sent_merge = sentiment_agg[['ID', 'avg_sentiment', 'sentiment_label', 'sentiment_count']].dropna(subset=['ID'])
    sent_merge['ID'] = sent_merge['ID'].astype(int)
    df = df.merge(sent_merge, on='ID', how='left')
    df['avg_sentiment'] = df['avg_sentiment'].fillna(0)
    df['sentiment_label'] = df['sentiment_label'].fillna('neutral')
    df['sentiment_count'] = df['sentiment_count'].fillna(0)

    df = df.sort_values('sentiment_count', ascending=False).drop_duplicates(subset=['ID'], keep='first')

    # Fill remaining missing
    for col in ['Height_cm', 'Weight_kg']:
        df[col] = df[col].fillna(df[col].median())
    df['Contract_Years'] = df['Contract_Years'].fillna(df['Contract_Years'].median())
    df['latest_market_val'] = df.groupby('BP')['latest_market_val'].transform(lambda x: x.fillna(x.median()))
    df['latest_market_val'] = df['latest_market_val'].fillna(df['latest_market_val'].median())
    df['latest_season'] = df['latest_season'].fillna(df['latest_season'].median())

    os.makedirs(PROC_DIR, exist_ok=True)
    out_path = os.path.join(PROC_DIR, 'cleaned_dataset_final.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | Shape: {df.shape}")
    return df

if __name__ == '__main__':
    run_cleaning()
