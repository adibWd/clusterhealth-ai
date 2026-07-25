"""
Feature engineering shared by both the XGBoost (tabular) and BiGRU-Attention
(sequential) models.

Two views of the same raw telemetry:
  1. TABULAR  -> rolling statistics per window, consumed by XGBoost
  2. SEQUENCE -> raw ordered windows, consumed by the BiGRU-Attention model
"""

import numpy as np
import pandas as pd

RAW_FEATURES = [
    "gpu_temp_c", "gpu_util_pct", "gpu_mem_util_pct", "power_draw_w",
    "fan_speed_pct", "ecc_sbe_errors", "ecc_dbe_errors", "cpu_temp_c",
    "cpu_util_pct", "disk_io_mbps", "network_io_mbps", "pcie_replay_errors",
]

WINDOW = 10  # timesteps of history used to make one prediction


def build_windows(df: pd.DataFrame, window: int = WINDOW):
    """
    Slide a window of `window` timesteps over each device's time series.
    Returns:
      seq_X   : (n_samples, window, n_features)  -- for BiGRU
      tab_X   : (n_samples, n_tabular_features)  -- for XGBoost
      y       : (n_samples,)
      meta    : DataFrame with device_id / timestep of the LAST step in window
    """
    if window < 2:
        raise ValueError("window must contain at least two timesteps")
    required = {"device_id", "timestep", "failure_within_horizon", *RAW_FEATURES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Telemetry data is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Telemetry data is empty")

    seq_list, tab_rows, y_list, meta_rows = [], [], [], []

    for device_id, g in df.groupby("device_id"):
        g = g.sort_values("timestep").reset_index(drop=True)
        vals = g[RAW_FEATURES].to_numpy(dtype=np.float32)
        labels = g["failure_within_horizon"].values

        for end in range(window, len(g) + 1):
            start = end - window
            win = vals[start:end]              # (window, n_features)
            label = labels[end - 1]            # label at last step of window

            seq_list.append(win)
            y_list.append(label)

            # tabular summary stats over the window
            mean = win.mean(axis=0)
            std = win.std(axis=0)
            last = win[-1]
            slope = (win[-1] - win[0]) / window
            tab_rows.append(np.concatenate([mean, std, last, slope]))

            meta_rows.append((device_id, g["timestep"].iloc[end - 1]))

    if not seq_list:
        raise ValueError(f"No device has at least {window} telemetry timesteps")
    seq_X = np.array(seq_list, dtype=np.float32)
    tab_X = np.array(tab_rows, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    meta = pd.DataFrame(meta_rows, columns=["device_id", "timestep"])

    return seq_X, tab_X, y, meta


def tabular_feature_names():
    names = []
    for suffix in ["mean", "std", "last", "slope"]:
        names += [f"{f}_{suffix}" for f in RAW_FEATURES]
    return names
