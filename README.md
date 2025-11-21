# requirements (pip install): yfinance pandas numpy scikit-learn xgboost shap matplotlib
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import datetime as dt
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV from yfinance for `ticker`. Returns a DataFrame with a DatetimeIndex.
    """
    # Make sure interval provided (original code referenced an undefined 'interval' variable)
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} (period={period}, interval={interval})")
    # drop NA rows (important for rolling features)
    df = df.dropna().copy()
    df.index.name = "date"
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce features from OHLCV dataframe. Returns X (features).
    """
    X = pd.DataFrame(index=df.index)
    # basic price/return features
    X['close'] = df['Close']
    X['open'] = df['Open']
    X['high'] = df['High']
    X['low'] = df['Low']
    X['volume'] = df['Volume']

    # daily return and lagged returns
    X['ret'] = X['close'].pct_change(1)
    X['ret_1'] = X['ret'].shift(1)   # previous day's return
    X['ret_3'] = X['ret'].rolling(window=3).mean()

    # rolling volatility (std of returns)
    X['vol_7'] = X['ret'].rolling(window=7).std()
    X['vol_21'] = X['ret'].rolling(window=21).std()

    # rolling average volume and volume spike metric
    X['vol_mean_21'] = X['volume'].rolling(window=21).mean()
    # to avoid division by zero, add tiny epsilon
    X['vol_spike'] = X['volume'] / (X['vol_mean_21'] + 1e-9)

    # drop rows with NA created by rolling calculations
    X = X.dropna()
    return X

def detect_anomalies_iso(X: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """
    Fit IsolationForest on features and return a dataframe with anomaly score and flags.
    """
    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    iso.fit(X)
    # decision_function: higher means more normal; flip sign so higher = more anomalous
    anomaly_score = -iso.decision_function(X)
    preds = iso.predict(X)  # -1 for anomaly, 1 for normal

    X_out = X.copy()
    X_out['anomaly_score'] = anomaly_score
    X_out['anomaly'] = preds
    X_out['anomaly_flag'] = (preds == -1).astype(int)
    return X_out

def plot_anomalies(X_with_flags: pd.DataFrame, n_top: int = 20):
    """
    Plot close price and annotate top anomalies.
    """
    dfp = X_with_flags.copy()
    top = dfp.sort_values('anomaly_score', ascending=False).head(n_top)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dfp.index, dfp['close'], label='Close')
    ax.scatter(top.index, top['close'], color='red', marker='x', s=60, label='Top anomalies')
    ax.set_title('Close price and top anomaly flags')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def train_xgb_supervised(X: pd.DataFrame, label_col: str = 'label', features: list = None):
    """
    Train a simple XGBoost classifier on provided features and return model, X_test, y_test, y_pred_proba, y_pred.
    Expects the DataFrame to contain the label_col.
    """
    if features is None:
        # pick some reasonable defaults if user didn't provide
        features = ['ret', 'ret_1', 'ret_3', 'vol_7', 'vol_21', 'vol_spike']

    # ensure no NA
    data = X.dropna().copy()
    X_feats = data[features]
    y = data[label_col].astype(int)

    # time-based train-test split: use first 80% as train, last 20% as test (no shuffle)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X_feats.iloc[:split_idx], X_feats.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    return model, X_test, y_test, y_pred_proba, y_pred

def main():
    ticker = "AAPL"
    period = "3y"

    # 1) Load data
    df = load_data(ticker=ticker, period=period, interval='1d')
    print(f"Loaded rows: {len(df)} from {ticker}")

    # 2) Feature engineering
    X = feature_engineer(df)
    print("Feature engineering produced rows:", len(X))

    # 3) Unsupervised anomaly detection
    X_anom = detect_anomalies_iso(X, contamination=0.01)
    print("Anomalies flagged:", X_anom['anomaly_flag'].sum())

    # top anomalies
    top_anoms = X_anom.sort_values('anomaly_score', ascending=False).head(20)
    print("Top anomalies (head):")
    print(top_anoms[['close', 'anomaly_score', 'vol_spike']].head(10))

    # plotting
    plot_anomalies(X_anom, n_top=15)

    # 4) Supervised demo (synthetic labels) - ONLY for demo / illustrative purposes
    # Create a synthetic label: large volume spikes are suspicious
    X_sup = X_anom.copy()
    # e.g. label top 1% vol_spike as suspicious
    threshold = X_sup['vol_spike'].quantile(0.99)
    X_sup['label'] = (X_sup['vol_spike'] > threshold).astype(int)
    print(f"Number of synthetic positive labels: {X_sup['label'].sum()}")

    # Train XGBoost supervised model
    features = ['ret', 'ret_1', 'ret_3', 'vol_7', 'vol_21', 'vol_spike']
    model, X_test, y_test, y_proba, y_pred = train_xgb_supervised(X_sup, label_col='label', features=features)

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float('nan')

    print("Supervised (synthetic) metrics:")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1:        {f1:.4f}")
    print(f" ROC AUC:   {roc:.4f}")

    # SHAP explainability (TreeExplainer works with XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # summary plot (will pop up)
    shap.summary_plot(shap_values, X_test, show=True)

    # Save top anomalies CSV
    outcols = ['close', 'anomaly_score', 'vol_spike', 'anomaly_flag']
    outfname = "top_anomalies.csv"
    top_anoms[outcols].to_csv(outfname)
    print(f"Saved top anomalies to: {os.path.abspath(outfname)}")

if __name__ == "__main__":
    main()
