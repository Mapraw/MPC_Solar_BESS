
# src/runtime/utils.py
# Utilities for online EMS runtime: file polling, reading latest CSVs, and fallbacks

from __future__ import annotations
import os, glob
import pandas as pd
from typing import Optional, Tuple

def ontime_csv(path_glob: str, ontime: str) -> Optional[str]:
    """
    Return the latest file path that matches the glob, or None if none exists.
    Example: ontime_csv('data/inbox/forecast/*.csv')
    """
    # files = glob.glob(path_glob)
    # if not files:
    #     return None
    # # Sort by modified time
    # files.sort(key=lambda p: os.path.getmtime(p))
    return ontime

def latest_csv(path_glob: str) -> Optional[str]:
    """
    Return the latest file path that matches the glob, or None if none exists.
    Example: latest_csv('data/inbox/forecast/*.csv')
    """
    files = glob.glob(path_glob)
    if not files:
        return None
    # Sort by modified time
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

def safe_read_csv(path: Optional[str], parse_dates_cols: Tuple[str, ...]) -> Optional[pd.DataFrame]:
    """
    Read CSV safely and return a DataFrame, or None if path is None or file missing.
    """
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=list(parse_dates_cols))
    # Sort by timestamp if present
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def dedup_and_clip_to_day(df: pd.DataFrame, day_start: pd.Timestamp, day_end: pd.Timestamp) -> pd.DataFrame:
    """
    Remove duplicate timestamps and clip to the [day_start, day_end] window.
    """
    df = df.drop_duplicates(subset=["timestamp"])
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= day_end)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def forward_fill_day_ahead_to_5min(df15: pd.DataFrame, dt5_min: int) -> pd.DataFrame:
    """
    Fallback: if 5-min forecast is not available, create one by forward-filling day-ahead 15-min values.
    Returns a 5-min DataFrame: ['timestamp', 'solar_forecast_kw'].
    """
    df15 = df15.copy().set_index("timestamp").asfreq("15min", method="pad")
    full5 = pd.date_range(df15.index.min(), df15.index.max(), freq=f"{dt5_min}min")
    df5 = df15.reindex(full5, method="pad").rename_axis("timestamp").reset_index()
    df5 = df5.rename(columns={"expected_power_kw": "solar_forecast_kw"})
    df5["solar_forecast_kw"] = df5["solar_forecast_kw"].clip(lower=0.0)
    return df5[["timestamp", "solar_forecast_kw"]]

def merge_forecast_actual(df5f: pd.DataFrame, df5a: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge forecast and actual (if provided). Do not fill actual where missing; we keep track of availability.
    Ensures non-negative forecast.
    """
    df = df5f.copy()
    df["solar_forecast_kw"] = df["solar_forecast_kw"].fillna(0.0).clip(lower=0.0)
    if df5a is not None:
        df = df.merge(df5a, on="timestamp", how="left")
        df["actual_available"] = df["solar_actual_kw"].notna()
    else:
        df["solar_actual_kw"] = pd.NA
        df["actual_available"] = False
    return df

def short_forecast_from_day_ahead(df15: pd.DataFrame, t_now: pd.Timestamp, dt5_min: int = 5) -> pd.DataFrame:
    next_ts = pd.date_range(t_now, t_now + pd.Timedelta(minutes=10), freq=f"{dt5_min}min")
    df15_ff = df15.set_index("timestamp").asfreq("15min", method="pad")
    df5 = df15_ff.reindex(pd.date_range(df15_ff.index.min(), df15_ff.index.max(), freq=f"{dt5_min}min"), method="pad")
    df5 = df5.rename_axis("timestamp").reset_index().rename(columns={"expected_power_kw": "solar_forecast_kw"})
    return df5[df5["timestamp"].isin(next_ts)][["timestamp", "solar_forecast_kw"]]
