"""
TransferIQ — Week 5: LSTM Model (NumPy from scratch with BPTT)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle, os

PROC_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

FEATURE_COLS = [
    'OVA', 'performance_score', 'physical_index', 'technical_index',
    'avg_sentiment', 'availability_score', 'injury_risk_score',
    'is_peak_age', 'age_peak_diff', 'is_left_footed',
    'pos_defender', 'pos_forward', 'pos_goalkeeper', 'pos_midfielder'
]
SEQUENCE_LEN = 3


class NumpyLSTM:
    """LSTM implemented from scratch using NumPy with full BPTT and gradient clipping."""

    def __init__(self, input_size, hidden_size=24):
        np.random.seed(42)
        s = np.sqrt(2.0 / (input_size + hidden_size))
        concat = input_size + hidden_size
        self.Wf = np.random.randn(hidden_size, concat) * s
        self.Wi = np.random.randn(hidden_size, concat) * s
        self.Wo = np.random.randn(hidden_size, concat) * s
        self.Wc = np.random.randn(hidden_size, concat) * s
        self.bf = np.ones((hidden_size,))
        self.bi = np.zeros((hidden_size,))
        self.bo = np.zeros((hidden_size,))
        self.bc = np.zeros((hidden_size,))
        self.Wy = np.random.randn(hidden_size) * s
        self.by = 0.0
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.losses = []

    def _sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    def _forward_seq(self, seq):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        cache = []
        for t in range(len(seq)):
            x = seq[t]
            z = np.concatenate([x, h])
            f = self._sigmoid(self.Wf @ z + self.bf)
            i = self._sigmoid(self.Wi @ z + self.bi)
            o = self._sigmoid(self.Wo @ z + self.bo)
            g = np.tanh(self.Wc @ z + self.bc)
            c_new = f * c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((x, h, c, f, i, o, g, c_new, h_new, z))
            h, c = h_new, c_new
        return h, c, cache

    def predict(self, X):
        return np.array([self.Wy @ self._forward_seq(seq)[0] + self.by for seq in X])

    def fit(self, X, y, epochs=80, lr=0.002, batch_size=32, verbose=True):
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            epoch_loss = 0
            for start in range(0, n, batch_size):
                bidx = idx[start:start + batch_size]
                dWy=np.zeros_like(self.Wy); dby=0.0
                dWf=np.zeros_like(self.Wf); dWi=np.zeros_like(self.Wi)
                dWo=np.zeros_like(self.Wo); dWc=np.zeros_like(self.Wc)
                dbf=np.zeros_like(self.bf); dbi=np.zeros_like(self.bi)
                dbo=np.zeros_like(self.bo); dbc=np.zeros_like(self.bc)

                for i in bidx:
                    h_last, _, cache = self._forward_seq(X[i])
                    pred = self.Wy @ h_last + self.by
                    err  = pred - y[i]
                    epoch_loss += err ** 2
                    dWy += err * h_last; dby += err
                    dh = err * self.Wy; dc = np.zeros(self.hidden_size)

                    for t in reversed(range(len(X[i]))):
                        x, h_prev, c_prev, f, ig, o, g, c_new, h_new, z = cache[t]
                        tanh_c = np.tanh(c_new)
                        do = dh * tanh_c
                        dc += dh * o * (1 - tanh_c ** 2)
                        df_g = dc * c_prev; di_g = dc * g; dg_g = dc * ig; dc = dc * f

                        df_pre = df_g * f * (1-f); di_pre = di_g * ig * (1-ig)
                        do_pre = do * o * (1-o);   dg_pre = dg_g * (1-g**2)

                        dWf += np.outer(df_pre, z); dbf += df_pre
                        dWi += np.outer(di_pre, z); dbi += di_pre
                        dWo += np.outer(do_pre, z); dbo += do_pre
                        dWc += np.outer(dg_pre, z); dbc += dg_pre

                        dz = self.Wf.T@df_pre + self.Wi.T@di_pre + self.Wo.T@do_pre + self.Wc.T@dg_pre
                        dh = dz[self.input_size:]

                clip = 1.0
                bs = len(bidx)
                for arr in [dWf, dWi, dWo, dWc, dWy, dbf, dbi, dbo, dbc]:
                    np.clip(arr, -clip, clip, out=arr)

                self.Wf-=lr*dWf/bs; self.bf-=lr*dbf/bs
                self.Wi-=lr*dWi/bs; self.bi-=lr*dbi/bs
                self.Wo-=lr*dWo/bs; self.bo-=lr*dbo/bs
                self.Wc-=lr*dWc/bs; self.bc-=lr*dbc/bs
                self.Wy-=lr*dWy/bs; self.by-=lr*dby/bs

            self.losses.append(float(epoch_loss / n))
            if verbose and (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} — MSE: {epoch_loss/n:.4f}")
        return self


def build_sequences(df, feature_cols, target_col, seq_len=3):
    X_seqs, y_targets = [], []
    for pid, grp in df.groupby('ID'):
        grp = grp.sort_values('season').reset_index(drop=True)
        if len(grp) < seq_len + 1:
            continue
        for i in range(len(grp) - seq_len):
            X_seqs.append(grp[feature_cols].values[i:i + seq_len])
            y_targets.append(grp[target_col].values[i + seq_len])
    return np.array(X_seqs), np.array(y_targets)


def run_lstm():
    print("="*55)
    print("TRANSFERIQ — LSTM MODEL")
    print("="*55)
    df = pd.read_csv(os.path.join(PROC_DIR, 'lstm_timeseries_dataset.csv'))

    scaler_X = MinMaxScaler(); scaler_y = MinMaxScaler()
    df[FEATURE_COLS] = scaler_X.fit_transform(df[FEATURE_COLS])
    df[['log_market_val']] = scaler_y.fit_transform(df[['log_market_val']])

    X, y = build_sequences(df, FEATURE_COLS, 'log_market_val', SEQUENCE_LEN)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Sequences: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")

    # Univariate LSTM
    print("\n[1/2] Univariate LSTM (OVA only)...")
    lstm_uni = NumpyLSTM(input_size=1, hidden_size=16)
    lstm_uni.fit(X_train[:, :, :1], y_train, epochs=80, lr=0.005)
    y_pred_uni  = scaler_y.inverse_transform(lstm_uni.predict(X_test[:, :, :1]).reshape(-1, 1)).flatten()
    y_test_inv  = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    r2_uni  = r2_score(y_test_inv, y_pred_uni)
    print(f"  R²={r2_uni:.4f}  RMSE={np.sqrt(mean_squared_error(y_test_inv,y_pred_uni)):.4f}")

    # Multivariate LSTM
    print("\n[2/2] Multivariate LSTM (all features)...")
    lstm_multi = NumpyLSTM(input_size=len(FEATURE_COLS), hidden_size=32)
    lstm_multi.fit(X_train, y_train, epochs=80, lr=0.002)
    y_pred_multi = scaler_y.inverse_transform(lstm_multi.predict(X_test).reshape(-1, 1)).flatten()
    r2_multi = r2_score(y_test_inv, y_pred_multi)
    print(f"  R²={r2_multi:.4f}  RMSE={np.sqrt(mean_squared_error(y_test_inv,y_pred_multi)):.4f}")

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'lstm_univariate.pkl'), 'wb') as f:
        pickle.dump({'model': lstm_uni, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    with open(os.path.join(MODELS_DIR, 'lstm_multivariate.pkl'), 'wb') as f:
        pickle.dump({'model': lstm_multi, 'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    print("\n✓ Models saved to models/")
    return lstm_uni, lstm_multi

if __name__ == '__main__':
    run_lstm()
