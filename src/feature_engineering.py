"""
TransferIQ — Week 3-4: Feature Engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
RAW_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

def career_stage(age):
    if age < 21:    return 'youth'
    elif age <= 25: return 'developing'
    elif age <= 29: return 'prime'
    elif age <= 32: return 'experienced'
    else:           return 'veteran'

def injury_risk_cat(days):
    if days == 0:      return 'none'
    elif days < 50:    return 'low'
    elif days < 150:   return 'medium'
    else:              return 'high'

def position_group(bp):
    if bp in ['GK']:                         return 'goalkeeper'
    elif bp in ['CB','LB','RB','LWB','RWB']: return 'defender'
    elif bp in ['CDM','CM','CAM','LM','RM']: return 'midfielder'
    elif bp in ['ST','CF','LW','RW']:        return 'forward'
    else:                                    return 'other'

def trend_cat(g):
    if g > 0.15:    return 'rising_fast'
    elif g > 0.05:  return 'rising'
    elif g > -0.05: return 'stable'
    elif g > -0.15: return 'declining'
    else:           return 'declining_fast'

def run_feature_engineering():
    df = pd.read_csv(os.path.join(PROC_DIR, 'cleaned_dataset_final.csv'))
    market = pd.read_csv(os.path.join(RAW_DIR, 'market_value.csv'))

    # Fix string columns
    for col in ['Attacking', 'Skill', 'Movement']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print("Engineering age features...")
    df['age_peak_diff']  = df['Age'] - 26
    df['is_peak_age']    = ((df['Age'] >= 23) & (df['Age'] <= 28)).astype(int)
    df['age_squared']    = df['Age'] ** 2
    df['career_stage']   = df['Age'].apply(career_stage)

    print("Engineering performance features...")
    df['physical_index']  = ((df['Acceleration'] + df['Strength'] + df['Jumping'] + df['Stamina']) / 4).round(2)
    df['technical_index'] = ((df['Short Passing'] + df['Shot Power'] + df['Heading Accuracy']) / 3).round(2)
    df['performance_score'] = (
        df['OVA'] * 0.4 + (df['Attacking'] / 10) * 0.2 +
        (df['Skill'] / 10) * 0.2 + (df['Movement'] / 10) * 0.2).round(2)

    print("Engineering position features...")
    df['position_group'] = df['BP'].apply(position_group)
    pos_dummies = pd.get_dummies(df['position_group'], prefix='pos')
    df = pd.concat([df, pos_dummies], axis=1)

    print("Engineering injury features...")
    df['injury_risk_category'] = df['total_days_missed'].apply(injury_risk_cat)
    df['availability_score']   = (1 - (df['total_days_missed'] / df['total_days_missed'].max())).round(3)

    df['is_left_footed']       = (df['foot'] == 'Left').astype(int)
    df['contract_value_proxy'] = (df['Contract_Years'] * df['OVA']).round(1)

    print("Engineering market time-series features...")
    market_clean = market.drop_duplicates(subset=['ID', 'season']).copy()
    results = []
    for pid, grp in market_clean.sort_values(['ID', 'season']).groupby('ID'):
        grp = grp.copy()
        grp['val_growth'] = grp['market_val'].pct_change().fillna(0)
        results.append(grp)
    market_ts = pd.concat(results, ignore_index=True)

    market_agg = market_ts.groupby('ID').agg(
        num_seasons=('season', 'count'),
        max_market_val=('market_val', 'max'),
        min_market_val=('market_val', 'min'),
        mean_market_val=('market_val', 'mean'),
        avg_growth_rate=('val_growth', 'mean'),
        last_market_val=('market_val', 'last')
    ).reset_index()
    market_agg['value_volatility'] = ((market_agg['max_market_val'] - market_agg['min_market_val']) / market_agg['mean_market_val']).round(3)
    market_agg['avg_growth_rate']  = market_agg['avg_growth_rate'].round(4)

    df = df.merge(market_agg, on='ID', how='left')
    for col in ['num_seasons', 'max_market_val', 'min_market_val', 'mean_market_val', 'avg_growth_rate', 'last_market_val', 'value_volatility']:
        df[col] = df[col].fillna(df[col].median())

    print("Engineering advanced features...")
    df['sentiment_performance_index'] = (df['performance_score'] * (1 + df['avg_sentiment'] * 0.1)).round(3)
    df['log_market_val']              = np.log1p(df['last_market_val'])
    df['injury_adj_performance']      = (df['performance_score'] * df['availability_score']).round(3)
    df['potential_score']             = np.where(df['Age'] < 25, df['OVA'] * (1 + (25 - df['Age']) * 0.02), df['OVA']).round(2)
    df['value_trend']                 = df['avg_growth_rate'].apply(trend_cat)

    print("Normalizing features...")
    cols_to_scale = ['OVA', 'performance_score', 'physical_index', 'technical_index',
                     'total_days_missed', 'avg_sentiment', 'Contract_Years', 'Age']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols_to_scale])
    scaled_df = pd.DataFrame(scaled, columns=[f'{c}_norm' for c in cols_to_scale])
    df = pd.concat([df.reset_index(drop=True), scaled_df], axis=1)

    # Build LSTM time-series dataset
    player_base = df[['ID', 'Age', 'OVA', 'performance_score', 'physical_index',
                       'technical_index', 'avg_sentiment', 'availability_score',
                       'injury_risk_score', 'is_peak_age', 'age_peak_diff',
                       'is_left_footed', 'contract_value_proxy', 'sentiment_performance_index'] +
                      [c for c in df.columns if c.startswith('pos_')]].copy()

    lstm_df = market_clean.merge(player_base, on='ID', how='left').dropna()
    lstm_df = lstm_df.sort_values(['ID', 'season']).reset_index(drop=True)
    lstm_df['log_market_val'] = np.log1p(lstm_df['market_val'])
    name_map = dict(zip(market_clean['ID'], market_clean['Name']))
    lstm_df['Name'] = lstm_df['ID'].map(name_map)

    out1 = os.path.join(PROC_DIR, 'featured_dataset_final.csv')
    out2 = os.path.join(PROC_DIR, 'lstm_timeseries_dataset.csv')
    df.to_csv(out1, index=False)
    lstm_df.to_csv(out2, index=False)
    print(f"Saved: {out1} | Shape: {df.shape}")
    print(f"Saved: {out2} | Shape: {lstm_df.shape}")
    return df, lstm_df

if __name__ == '__main__':
    run_feature_engineering()
