"""
Central configuration for paths and parameters.
Edit these values to match your local setup.
"""
from pathlib import Path

# Root is inferred from this file's location: Project/src/config.py -> Project/
ROOT = Path(__file__).resolve().parents[1]

# Data folder where your .mat files live (you said: Project/Data/battery_data)
DATA_DIR = ROOT / "Data" / "battery_data"

# Intermediate outputs
PROCESSED_PARQUET = ROOT / "Data" / "processed.parquet"
WINDOWS_DIR = ROOT / "Data" / "windows"

# Training/eval outputs
RUNS_DIR = ROOT / "runs"

# Labeling rule: fault if capacity (Ah) drops below this
CAPACITY_THRESHOLD = 1.4

# Sliding-window params
WIN = 256
STRIDE = 64

# Features (z-scored) used in windows
FEATURES = ["voltage_v_z", "current_a_z", "temp_c_z"]

# Train on the B0005 battery
TRAIN_BATTS = ["B0005"]

# Validate on the B0006 battery
VAL_BATTS   = ["B0006"]

# Test on the B0018 battery
TEST_BATTS  = ["B0018"]

