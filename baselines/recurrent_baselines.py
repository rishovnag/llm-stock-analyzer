# Recurrent baselines (LSTM, GRU) for the equity-index movement tasks.
# File: baselines/recurrent_baselines.py
#
# These are the frequency-driven recurrent baselines the paper compares the
# LLM ensemble against (Table: "Comparison with recurrent baselines"). They are
# trained and evaluated under the SAME label definitions and the SAME fixed
# train/test split as the LLM-ensemble fixed-split runs, so the numbers are
# directly comparable.
#
# Design choices (kept deliberately standard so the comparison is fair and
# reproducible):
#   * Input: a look-back window of L daily feature vectors (returns + a small
#     set of trailing technical features), standardised with statistics
#     estimated on the training portion only (no leakage).
#   * Architecture: a single recurrent layer (LSTM or GRU) -> dropout -> dense
#     sigmoid. Class imbalance is left intact (no resampling), matching the
#     paper's "no data-level rebalancing" stance; we only pass class weights to
#     the loss so the rare class is not entirely ignored.
#   * Output: Accuracy and F1 on the held-out test segment.
#
# Requires: tensorflow>=2.11 (see requirements-baselines.txt). Kept out of the
# core requirements.txt so the LLM pipeline has no deep-learning dependency.

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MOVEMENT_FILE_PATTERNS = {
    "direction": "Direction",
    "sig_move_1pct": "SigMove_1pct",
    "sig_move_2pct": "SigMove_2pct",
    "sig_move_3pct": "SigMove_3pct",
}


def find_dataset(data_dir: Path, symbol: str, movement_type: str) -> Path:
    pat = MOVEMENT_FILE_PATTERNS[movement_type]
    for p in data_dir.glob(f"{symbol}*.csv"):
        if pat in p.name and "neg" not in p.name:
            return p
    raise FileNotFoundError(f"No {movement_type} file for {symbol} in {data_dir}")


def load_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, thousands=",")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """A compact, fully causal trailing feature set."""
    price = df["Adj_Close"] if "Adj_Close" in df.columns else df["Close"]
    out = pd.DataFrame(index=df.index)
    ret = np.log(price / price.shift(1))
    out["ret"] = ret
    out["ret_abs"] = ret.abs()
    out["roll_mean_5"] = ret.rolling(5).mean()
    out["roll_std_10"] = ret.rolling(10).std()
    out["roll_std_20"] = ret.rolling(20).std()
    out["mom_5"] = price.pct_change(5)
    out["mom_10"] = price.pct_change(10)
    # RSI(14)
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi14"] = 100 - 100 / (1 + rs)
    if "Volume" in df.columns:
        vol = df["Volume"].astype(float)
        out["vol_ratio"] = vol / vol.rolling(20).mean()
    else:
        out["vol_ratio"] = 1.0
    return out.fillna(0.0)


def make_windows(features: np.ndarray, labels: np.ndarray, lookback: int):
    X, y = [], []
    for t in range(lookback, len(features)):
        X.append(features[t - lookback:t])
        y.append(labels[t])
    return np.asarray(X), np.asarray(y)


def build_model(kind: str, lookback: int, n_features: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    rnn = layers.LSTM if kind == "lstm" else layers.GRU
    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        rnn(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_one(data_dir: Path, symbol: str, movement_type: str, kind: str,
                 train_ratio: float = 0.8, lookback: int = 20, epochs: int = 15,
                 seed: int = 42) -> dict:
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    df = load_frame(find_dataset(data_dir, symbol, movement_type))
    movement_col = "Direction" if "Direction" in df.columns else "Significant_Move"
    feats = build_features(df)
    labels = df[movement_col].astype(float).values

    # Causal standardisation: fit scaler on the training portion only.
    split = int(len(df) * train_ratio)
    mu = feats.iloc[:split].mean()
    sigma = feats.iloc[:split].std().replace(0, 1.0)
    feats_z = ((feats - mu) / sigma).values

    X, y = make_windows(feats_z, labels, lookback)
    # Window index t aligns to label at t; split on the window axis.
    win_split = split - lookback
    win_split = max(1, min(win_split, len(X) - 1))
    X_tr, X_te = X[:win_split], X[win_split:]
    y_tr, y_te = y[:win_split], y[win_split:]

    valid = ~np.isnan(y_tr)
    X_tr, y_tr = X_tr[valid], y_tr[valid]
    valid_te = ~np.isnan(y_te)
    X_te, y_te = X_te[valid_te], y_te[valid_te]

    if len(np.unique(y_tr)) < 2 or len(X_te) == 0:
        return {"symbol": symbol, "movement_type": movement_type, "model": kind,
                "accuracy": float("nan"), "f1_score": float("nan"),
                "note": "degenerate split"}

    pos = max(1, int(y_tr.sum()))
    neg = max(1, int((1 - y_tr).sum()))
    class_weight = {0: 1.0, 1: neg / pos}  # imbalance-aware loss, no resampling

    model = build_model(kind, lookback, X.shape[2])
    model.fit(X_tr, y_tr, validation_split=0.1, epochs=epochs, batch_size=32,
              class_weight=class_weight, verbose=0)

    prob = model.predict(X_te, verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    return {
        "symbol": symbol, "movement_type": movement_type, "model": kind,
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1_score": float(f1_score(y_te, pred, zero_division=0)),
        "test_size": int(len(y_te)),
    }


def main():
    ap = argparse.ArgumentParser(description="LSTM/GRU recurrent baselines")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--symbols", nargs="+", default=["NIFTY50", "NIFTYBANK", "NIFTYIT"])
    ap.add_argument("--movements", nargs="+",
                    default=["direction", "sig_move_1pct", "sig_move_2pct", "sig_move_3pct"])
    ap.add_argument("--models", nargs="+", default=["lstm", "gru"])
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="baselines/baseline_results.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    for kind in args.models:
        for symbol in args.symbols:
            for mv in args.movements:
                try:
                    r = evaluate_one(data_dir, symbol, mv, kind,
                                     train_ratio=args.train_ratio,
                                     lookback=args.lookback, epochs=args.epochs)
                    logger.info(f"{kind} {symbol} {mv}: "
                                f"acc={r.get('accuracy'):.4f} f1={r.get('f1_score'):.4f}")
                    rows.append(r)
                except Exception as e:
                    logger.error(f"{kind} {symbol} {mv} failed: {e}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    logger.info(f"Wrote {len(rows)} baseline results to {out}")


if __name__ == "__main__":
    main()
