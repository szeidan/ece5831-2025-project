"""
preprocess.py (robust labels)
-----------------------------
- Loads .mat files
- Extracts discharge cycles
- Fills missing capacity values forward per battery
- Labels: fault = 1 if capacity < CAPACITY_THRESHOLD
- Normalizes signals per battery (z-score)
- Writes a single parquet and prints label stats
Run:  python -m src.preprocess
"""
from pathlib import Path
import numpy as np
import pandas as pd
from . import config
from .dataio import load_battery_mat, extract_discharge_cycles

def normalize_per_battery(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["voltage_v", "current_a", "temp_c"]
    stats = df[cols].agg(["mean", "std"])
    for c in cols:
        m, s = stats.loc["mean", c], stats.loc["std", c]
        s = s if (s is not None and s > 1e-12) else 1.0
        df[c + "_z"] = (df[c] - m) / s
    return df

def main():
    data_dir = config.DATA_DIR
    out_parquet = config.PROCESSED_PARQUET
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for mat_path in sorted(data_dir.glob("*.mat")):
        bid, root = load_battery_mat(mat_path)
        df = extract_discharge_cycles(root)
        if df.empty:
            continue
        df["battery_id"] = bid
        # Fill missing capacity forward per battery
        df["capacity_ah"] = df["capacity_ah"].ffill()
        df["fault"] = (df["capacity_ah"] < config.CAPACITY_THRESHOLD).astype(int)
        df = normalize_per_battery(df)
        frames.append(df)

    if not frames:
        raise SystemExit("No discharge data found. Check Data/battery_data for .mat files.")

    out_df = pd.concat(frames, ignore_index=True)
    out_df.to_parquet(out_parquet, index=False)

    print(f"[preprocess] wrote {out_parquet} rows={len(out_df)}")
    print("[preprocess] per-battery fault rate:")
    print(out_df.groupby("battery_id")["fault"].mean().round(4))
    print("[preprocess] overall fault rate:", round(float(out_df["fault"].mean()), 4))

if __name__ == "__main__":
    main()
