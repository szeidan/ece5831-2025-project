"""
dataio.py
----------
Utilities for safely loading the NASA .mat battery files using SciPy.
Keeps all MATLAB-struct handling here so the rest of the code sees clean pandas data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

def load_battery_mat(mat_path: Path):
    """
    Load one .mat file and return (battery_id, root_struct).

    The MATLAB file typically stores data as a top-level struct named after the file,
    e.g., B0005.mat -> key 'B0005'. We fall back to the first non-private key otherwise.
    """
    key = mat_path.stem  # e.g., B0005
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if key in mat:
        root = mat[key]
    else:
        # Fall back to first non-private key
        root = next(v for k, v in mat.items() if not k.startswith("__"))
    return key, root

def extract_discharge_cycles(root_struct) -> pd.DataFrame:
    """
    Extract only discharge cycles from the MATLAB struct to a tidy DataFrame.
    Columns: time_s, voltage_v, current_a, temp_c, capacity_ah
    """
    rows = []
    try:
        cycles = root_struct.cycle  # MATLAB struct array
    except Exception as e:
        # Unexpected structure
        return pd.DataFrame(columns=["time_s","voltage_v","current_a","temp_c","capacity_ah"])

    for c in np.atleast_1d(cycles):
        ctype = str(getattr(c, "type", ""))
        if "discharge" not in ctype.lower():
            continue
        d = c.data
        V = np.array(getattr(d, "Voltage_measured", [])).flatten()
        I = np.array(getattr(d, "Current_measured", [])).flatten()
        T = np.array(getattr(d, "Temperature_measured", [])).flatten()
        t = np.array(getattr(d, "Time", [])).flatten()
        cap_arr = np.array(getattr(d, "Capacity", [])).flatten()
        cap = float(cap_arr[0]) if cap_arr.size > 0 else np.nan

        n = min(len(V), len(I), len(T), len(t))  # align lengths safely
        if n == 0:
            continue

        rows.append(pd.DataFrame({
            "time_s": t[:n],
            "voltage_v": V[:n],
            "current_a": I[:n],
            "temp_c": T[:n],
            "capacity_ah": cap,
        }))

    if not rows:
        return pd.DataFrame(columns=["time_s","voltage_v","current_a","temp_c","capacity_ah"])

    return pd.concat(rows, ignore_index=True)
