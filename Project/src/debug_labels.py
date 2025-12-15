"""
debug_labels.py
----------------
Quick checks to ensure you have positives in each split.
Run: python -m src.debug_labels
"""
import pandas as pd
from . import config

def main():
    df = pd.read_parquet(config.PROCESSED_PARQUET)
    print("Per-battery sample counts:")
    print(df.groupby("battery_id").size())
    print("\nPer-battery fault rates:")
    print(df.groupby("battery_id")["fault"].mean().round(4))
    print("\nOverall fault rate:", round(float(df["fault"].mean()), 4))

if __name__ == "__main__":
    main()
