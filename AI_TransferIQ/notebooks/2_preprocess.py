import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import numpy as np
class Preprocess:
    def __init__(self,df):
        self.df=df
    def preprocess_data(file_path):
        print("Loading data...")
        df = pd.read_csv(file_path)
        
        # 1. Feature Engineering: Rolling averages for performance trends
        # We need to sort by player and date to calculate rolling stats correctly
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['player_id', 'date'])
        
        print("Engineering features...")
        # Performance trend (3-month rolling average)
        df['perf_trend_3m'] = df.groupby('player_id')['performance_rating'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df['goals_trend_3m'] = df.groupby('player_id')['goals_assists'].transform(lambda x: x.rolling(window=3, min_periods=1).sum())
        
        # New features: Lagged market values and volatility
        df['lag_1'] = df.groupby('player_id')['market_value'].shift(1)
        df['lag_2'] = df.groupby('player_id')['market_value'].shift(2)
        df['perf_vol_3m'] = df.groupby('player_id')['performance_rating'].transform(lambda x: x.rolling(window=3, min_periods=1).std())
        
        # Log transformation for stabilized returns
        df['ln_market_value'] = np.log1p(df['market_value'])
        
        # Time-based features
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # Cumulative injuries (proxy for injury proneness)
        df['cumulative_days_injured'] = df.groupby('player_id')['days_injured'].cumsum()
        
        # 2. One-hot encoding for categorical variable 'position'
        print("Encoding categorical variables...")
        df = pd.get_dummies(df, columns=['position'], drop_first=True)
        
        # 3. Scaling numerical data
        print("Scaling numerical features...")
        features_to_scale = [
            'performance_rating', 'goals_assists', 'minutes_played', 'days_injured',
            'social_sentiment_score', 'contract_duration_months', 
            'perf_trend_3m', 'goals_trend_3m', 'cumulative_days_injured',
            'lag_1', 'lag_2', 'perf_vol_3m', 'month', 'year'
        ]
        
        # We will use MinMaxScaler to keep ranges 0-1
        scaler_X = MinMaxScaler()
        df[features_to_scale] = scaler_X.fit_transform(df[features_to_scale])
        
        # Scale the target variable (market value) as well, it helps Neural Networks
        scaler_y = MinMaxScaler()
        df['market_value_scaled'] = scaler_y.fit_transform(df[['market_value']])
        
        # Separate scaler for log-values if needed (for LSTM, etc.)
        scaler_ln_y = MinMaxScaler()
        df['ln_market_value_scaled'] = scaler_ln_y.fit_transform(df[['ln_market_value']])
        
        # Save the scalers for inverse transformation later during evaluation
        joblib.dump(scaler_X, 'D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_X.pkl')
        joblib.dump(scaler_y, 'D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_y.pkl')
        joblib.dump(scaler_ln_y, 'D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\scaler_ln_y.pkl')
        
        # Handle any generated NaNs from rolling (min_periods=1 should prevent this but just in case)
        df = df.fillna(0)
        
        print("Data preprocessing complete.")
        return df

if __name__ == "__main__":
    df_processed = Preprocess.preprocess_data("D:\\New folder (5)\\infosys\\AI_TransferIQ\\data\\transferiq_dataset.csv")
    df_processed.to_csv("D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\transferiq_processed.csv", index=False)
    print(f"Processed dataset saved to transferiq_processed.csv with shape {df_processed.shape}")
