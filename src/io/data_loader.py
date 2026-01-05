
# src/io/data_loader.py
# Data ingestion + alignment for energy-tracking MPC
# - Reads 15-min day-ahead power forecast (target)
# - Reads 5-min solar forecast (power)
# - Reads 5-min solar actual (power, optional)
# - Converts 15-min power → 15-min target energy (kWh)
# - Maps 5-min samples to parent 15-min blocks and substeps (0,1,2)

from __future__ import annotations
import pandas as pd
from typing import Optional


# ---------- Readers ----------

def read_day_ahead_power_15min(path: str) -> pd.DataFrame:
    """
    Read the 15-minute day-ahead expected generation (power in kW).
    Required columns: timestamp, expected_power_kw
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "expected_power_kw"}
    if not required.issubset(df.columns):
        raise ValueError(f"'{path}' must contain columns: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def read_forecast_5min(path: str) -> pd.DataFrame:
    """
    Read the 5-minute solar forecast (power in kW).
    Required columns: timestamp, solar_forecast_kw
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "solar_forecast_kw"}
    if not required.issubset(df.columns):
        raise ValueError(f"'{path}' must contain columns: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def read_actual_5min(path: str) -> pd.DataFrame:
    """
    Read the 5-minute solar actuals (power in kW).
    Required columns: timestamp, solar_actual_kw
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "solar_actual_kw"}
    if not required.issubset(df.columns):
        raise ValueError(f"'{path}' must contain columns: {required}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------- Helpers for energy-tracking MPC ----------

def to_target_energy_15min(df15_power: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 15-min expected power (kW) to target energy per block (kWh):
      E_target_kwh = expected_power_kw * 0.25 (since 15-min = 0.25 h)
    Returns DataFrame with ['timestamp', 'E_target_kwh'] at 15-min resolution.
    """
    df = df15_power.copy()
    df["E_target_kwh"] = df["expected_power_kw"] * 0.25
    return df[["timestamp", "E_target_kwh"]]


def floor_to_15min(ts: pd.Timestamp) -> pd.Timestamp:
    """Floor a timestamp to its 15-min block start, e.g., 06:07 → 06:00."""
    minute = (ts.minute // 15) * 15
    return ts.replace(minute=minute, second=0, microsecond=0)


def build_tracking_frame(
    df15_power: pd.DataFrame,
    df5_forecast: pd.DataFrame,
    df5_actual: Optional[pd.DataFrame],
    dt5_min: int = 5
) -> pd.DataFrame:
    """
    Build the 5-min tracking frame used by the MPC:
      Columns:
        - timestamp (5-min grid)
        - block_start, block_end (15-min boundaries)
        - substep_in_block ∈ {0,1,2}
        - E_target_kwh (target energy for the 15-min block)
        - solar_forecast_kw (5-min forecast power)
        - solar_actual_kw (5-min actual power, optional)
        - actual_available (bool) — whether actual is present at that timestamp

    MPC behavior enabled by this frame:
      At each 5-min tick (e.g., 06:00, 06:05, 06:10), the controller
      uses actuals for elapsed substeps in the current 15-min block and
      forecast for the remaining substeps, then dispatches BESS so that
      by block_end the cumulative delivered energy matches E_target_kwh.
    """
    # 1) Target energy per 15-min block (ensure regular 15-min grid)
    dfE = to_target_energy_15min(df15_power).copy()
    dfE = dfE.set_index("timestamp").asfreq("15min", method="pad")

    # 2) Build 5-min timeline from the intersection range (forecast bounds)
    start_ts = df5_forecast["timestamp"].min()
    end_ts   = df5_forecast["timestamp"].max()
    full5 = pd.date_range(start_ts, end_ts, freq=f"{dt5_min}min")
    df5 = pd.DataFrame({"timestamp": full5})

    # 3) Map each 5-min sample to its 15-min block and substep
    df5["block_start"] = df5["timestamp"].apply(floor_to_15min)
    df5["block_end"]   = df5["block_start"] + pd.Timedelta(minutes=15)
    df5["substep_in_block"] = (
        (df5["timestamp"] - df5["block_start"]).dt.total_seconds() // (dt5_min * 60)
    ).astype(int)  # 0 at :00, 1 at :05, 2 at :10

    # 4) Attach target energy for the block
    dfE2 = dfE.reset_index().rename(columns={"timestamp": "block_start"})
    df5  = df5.merge(dfE2, on="block_start", how="left")

    # 5) Attach forecast power (non-negative, fill missing with 0)
    df5  = df5.merge(df5_forecast, on="timestamp", how="left")
    df5["solar_forecast_kw"] = df5["solar_forecast_kw"].fillna(0.0).clip(lower=0.0)

    # 6) Attach actual power (optional) and flag availability
    if df5_actual is not None:
        df5 = df5.merge(df5_actual, on="timestamp", how="left")
        df5["actual_available"] = df5["solar_actual_kw"].notna()
    else:
        df5["solar_actual_kw"]  = pd.NA
        df5["actual_available"] = False

    # Final ordering & cleanup
    df5 = df5.sort_values("timestamp").reset_index(drop=True)
    return df5
