"""
windows.py (any-positive labeling)
----------------------------------
Build sliding windows and split by battery.
- Window label is 1 if ANY sample in window is fault.
- Prints positive/negative counts for each split.
Run: python -m src.windows
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import config

def make_sliding_windows(df: pd.DataFrame, win: int, stride: int, features, label_col="fault"):
    X, y = [], []
    n = len(df)
    for start in range(0, n - win + 1, stride):
        end = start + win
        seg = df.iloc[start:end]
        lab = int(seg[label_col].max() >= 1)   # any-positive rule
        X.append(seg[features].values.astype("float32"))
        y.append(lab)
    X = np.stack(X) if X else np.empty((0, win, len(features)), dtype="float32")
    y = np.array(y, dtype="int64")
    return X, y

def split_by_battery(df: pd.DataFrame):
    tr = df[df["battery_id"].isin(config.TRAIN_BATTS)]
    va = df[df["battery_id"].isin(config.VAL_BATTS)]
    te = df[df["battery_id"].isin(config.TEST_BATTS)]
    return tr, va, te

def dump_set(name, sdf, out_dir):
    X, y = make_sliding_windows(sdf, config.WIN, config.STRIDE, config.FEATURES)
    np.save(out_dir / f"X_{name}.npy", X)
    np.save(out_dir / f"y_{name}.npy", y)
    pos = int(y.sum()); neg = int((y==0).sum())
    rate = float(y.mean()) if len(y) else 0.0
    print(f"{name}: X{X.shape} y{y.shape}  pos={pos} neg={neg} pos_rate={rate:.4f}")

def main():
    parquet = config.PROCESSED_PARQUET
    out_dir = config.WINDOWS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet)
    tr_df, va_df, te_df = split_by_battery(df)

    print("[windows] batteries by split:")
    print("  train:", sorted(set(tr_df['battery_id'])))
    print("  val:  ", sorted(set(va_df['battery_id'])))
    print("  test: ", sorted(set(te_df['battery_id'])))

    dump_set("train", tr_df, out_dir)
    dump_set("val",   va_df, out_dir)
    dump_set("test",  te_df, out_dir)

if __name__ == "__main__":
    main()
