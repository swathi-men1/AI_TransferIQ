"""
TransferIQ - Data Loading & Preprocessing Utilities
====================================================
Handles data loading, cleaning, feature engineering, and preprocessing
for the IPL Player Market Value Prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


def load_auction_data(uploaded_files=None, data_dir=None):
    """
    Load IPL auction CSV files from uploaded Streamlit files or a directory.
    """
    auction_data = {}
    
    if uploaded_files:
        for f in uploaded_files:
            try:
                df = pd.read_csv(f)
                year = _extract_year(f.name, df)
                auction_data[str(year)] = df
            except Exception as e:
                print(f"Error loading {f.name}: {e}")
    
    return auction_data


def _extract_year(filename, df):
    """Extract auction year from filename or DataFrame content."""
    import re
    match = re.search(r'(20\d{2})', filename)
    if match:
        return int(match.group(1))
    for col in df.columns:
        if 'year' in col.lower() or 'season' in col.lower():
            return int(df[col].iloc[0])
    return 2024


def clean_auction_df(df, year):
    """Clean and standardize a single auction DataFrame."""
    df = df.copy()
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    
    if 'Price_in_$' in df.columns:
        df['Winning_Bid'] = (
            df['Price_in_$'].astype(str).str.replace(',', '').astype(float) * 55
        )
    elif 'Price_in_rs' in df.columns:
        df['Winning_Bid'] = (
            df['Price_in_rs'].astype(str).str.replace(',', '').astype(float)
        )
    elif 'Winning_Bid_in_Rs' in df.columns:
        df['Winning_Bid'] = (
            df['Winning_Bid_in_Rs'].astype(str).str.replace(',', '').astype(float)
        )
    else:
        price_cols = [c for c in df.columns if 'price' in c.lower() or 'bid' in c.lower()]
        if price_cols:
            col = price_cols[0]
            df['Winning_Bid'] = (
                df[col].astype(str).str.replace(',', '').str.replace('$', '').str.strip().astype(float)
            )
    
    for col in ['Role', 'Specialism', 'Player_Type', 'Category', 'Playing_Role']:
        if col in df.columns:
            df['Role'] = df[col]
            break
    if 'Role' not in df.columns:
        df['Role'] = 'Unknown'
    
    for col in ['Name', 'Player_Name', 'Player']:
        if col in df.columns:
            df['Name'] = df[col]
            break
    
    for col in ['TeamName', 'Team', 'Franchise']:
        if col in df.columns:
            df['TeamName'] = df[col]
            break
    
    df['Season'] = int(year)
    return df


def build_master_dataset(auction_data, batting_df=None, bowling_df=None, injury_df=None, sentiment_df=None):
    """Build the master dataset by merging all data sources."""
    master_df_list = []
    for year, df in auction_data.items():
        if 'Winning_Bid' in df.columns:
            cleaned = clean_auction_df(df, year)
            master_df_list.append(cleaned)
    
    if not master_df_list:
        raise ValueError("No valid auction data found.")
    
    master_df = pd.concat(master_df_list, ignore_index=True)
    master_df.drop_duplicates(inplace=True)
    master_df.fillna({'Role': 'Unknown', 'TeamName': 'Unknown'}, inplace=True)
    
    if 'Winning_Bid' in master_df.columns:
        master_df['Winning_Bid'] = pd.to_numeric(master_df['Winning_Bid'], errors='coerce')
        master_df['Winning_Bid'] = master_df['Winning_Bid'].fillna(master_df['Winning_Bid'].median())
    
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        master_df[col] = master_df[col].fillna(0)
    
    return master_df


def prepare_features_and_target(master_df, target_col='Winning_Bid'):
    """Prepare feature matrix X and target vector y."""
    if target_col not in master_df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    
    drop_cols = [target_col, 'Name', 'Season', 'user_name', 'year']
    drop_cols = [c for c in drop_cols if c in master_df.columns]
    
    X = master_df.drop(columns=drop_cols, errors='ignore')
    X = X.select_dtypes(include=[np.number, 'bool']).fillna(0)
    
    bool_cols = X.select_dtypes(include=['bool']).columns
    X[bool_cols] = X[bool_cols].astype(int)
    
    y = master_df[target_col]
    return X, y, X.columns.tolist()


def scale_features(X_train, X_test=None):
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    return X_train_scaled, X_test_scaled, scaler


def preprocess_single_input(features_dict, feature_names, scaler=None):
    """Preprocess a single player input for model prediction."""
    arr = np.array([[features_dict.get(f, 0) for f in feature_names]], dtype=float)
    if scaler is not None:
        arr = scaler.transform(arr)
    return arr