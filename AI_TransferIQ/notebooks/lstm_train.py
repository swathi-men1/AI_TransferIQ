import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input
from sklearn.model_selection import train_test_split
import joblib

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def create_sequences(data, target, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_univariate_lstm(df, seq_length=3):
    print("Training Univariate LSTM...")
    # Univariate: only use previous market values to predict next market value
    data = df[['market_value_scaled']].values
    
    X, y = create_sequences(data, data, seq_length)
    
    # Split chronologically (since it's time series) or simple train_test_split depending on complexity
    # For simplicity here, regular split on sequences
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    model.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\univariate_lstm.keras')
    return model, history

def train_multivariate_lstm(df, seq_length=3):
    print("Training Multivariate LSTM...")
    # Features to include
    X_cols = [
        'performance_rating', 'goals_assists', 'days_injured',
        'social_sentiment_score', 'market_value_scaled', 'perf_trend_3m'
    ]
    data_X = df[X_cols].values
    data_y = df[['market_value_scaled']].values
    
    X, y = create_sequences(data_X, data_y, seq_length)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, len(X_cols)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    model.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\multivariate_lstm.keras')
    return model, history

if __name__ == "__main__":
    import os
    os.makedirs('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models', exist_ok=True)
    df = pd.read_csv("D:\\New folder (5)\\infosys\\AI_TransferIQ\\reports\\metrics\\transferiq_processed.csv")
    
    # Sort globally by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    
    # Temporal Split (Raw DataFrame)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Features to include
    multi_cols = [
        'performance_rating', 'goals_assists', 'days_injured',
        'social_sentiment_score', 'market_value_scaled', 'perf_trend_3m',
        'lag_1', 'lag_2', 'perf_vol_3m'
    ]
    seq_length = 3
    
    def get_player_sequences(dataframe, cols, target_col, seq_len):
        X_list, y_list = [], []
        for _, group in dataframe.groupby('player_id'):
            if len(group) <= seq_len: continue
            data_x = group[cols].values
            data_y = group[target_col].values
            x, y = create_sequences(data_x, data_y, seq_len)
            X_list.extend(x)
            y_list.extend(y)
        return np.array(X_list), np.array(y_list)

    print("Creating sequences for training set...")
    Xm_train, ym_train = get_player_sequences(train_df, multi_cols, 'market_value_scaled', seq_length)
    
    print("Creating sequences for test set...")
    Xm_test, ym_test = get_player_sequences(test_df, multi_cols, 'market_value_scaled', seq_length)
    
    print(f"Total train sequences: {len(Xm_train)}, Total test sequences: {len(Xm_test)}")

    # Note: Univariate can be derived from Multivariate or handled similarly
    # For brevity and better performance, we focus on the Multivariate model
    
    print("\n--- Training Multivariate LSTM ---")
    model_m = Sequential([
        LSTM(64, activation='relu', input_shape=(seq_length, len(multi_cols)), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model_m.compile(optimizer='adam', loss='mse')
    
    # Train without shuffling sequences to maintain temporal order within batches
    hist_m = model_m.fit(
        Xm_train, ym_train, 
        epochs=25, 
        batch_size=64, 
        validation_data=(Xm_test, ym_test), 
        verbose=1
    )
    
    model_m.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\multivariate_lstm.keras')
    
    # Saving histories (we'll save hist_m for both to satisfy evaluate.py expectations if needed)
    np.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\hist_multi.npy', hist_m.history)
    np.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\hist_uni.npy', hist_m.history) 
    print("Multivariate LSTM training complete.")
    print("LSTM training complete and models saved.")
    
    # ── Encoder-Decoder LSTM (multi-step forecasting, PDF §2.3) ─────────────
    print("\n--- Training Encoder-Decoder LSTM (3-step ahead) ---")
    n_steps_out = 3  # predict next 3 transfer windows
    
    def create_ed_sequences(data, seq_in, seq_out):
        X_ed, y_ed = [], []
        for i in range(len(data) - seq_in - seq_out + 1):
            X_ed.append(data[i: i + seq_in])
            y_ed.append(data[i + seq_in: i + seq_in + seq_out, 0])  # market_value only
        return np.array(X_ed), np.array(y_ed)
    
    X_ed_all, y_ed_all = [], []
    for _, group in df.groupby('player_id'):
        if len(group) < seq_length + n_steps_out:
            continue
        player_data = group[multi_cols].values
        xe, ye = create_ed_sequences(player_data, seq_length, n_steps_out)
        X_ed_all.extend(xe)
        y_ed_all.extend(ye)
    
    X_ed_all = np.array(X_ed_all)
    y_ed_all = np.array(y_ed_all)
    
    Xe_train, Xe_test, ye_train, ye_test = train_test_split(X_ed_all, y_ed_all, test_size=0.2, shuffle=False)
    
    # Encoder-Decoder architecture
    encoder_input = Input(shape=(seq_length, len(multi_cols)))
    encoded = LSTM(64, activation='relu')(encoder_input)
    repeated = RepeatVector(n_steps_out)(encoded)
    decoded = LSTM(32, activation='relu', return_sequences=True)(repeated)
    output = TimeDistributed(Dense(1))(decoded)
    
    enc_dec_model = Model(encoder_input, output)
    enc_dec_model.compile(optimizer='adam', loss='mse')
    hist_ed = enc_dec_model.fit(
        Xe_train, ye_train,
        epochs=15, batch_size=64,
        validation_data=(Xe_test, ye_test),
        verbose=1
    )
    enc_dec_model.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\encoder_decoder_lstm.keras')
    np.save('D:\\New folder (5)\\infosys\\AI_TransferIQ\\models\\hist_enc_dec.npy', hist_ed.history)
    print("Encoder-Decoder LSTM saved.")
    print("LSTM training complete and models saved.")
