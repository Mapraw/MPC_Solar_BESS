
"""
scripts/mimic_streams.py

Generate mimic CSV inputs for an online Solar+BESS MPC:
- Day-ahead (15-min) expected power
- Real-time 5-min forecast (updated every 5 minutes)
- Real-time 5-min actuals (updated every 5 minutes)

File formats:
  data/inbox/day_ahead_YYYYMMDD.csv
    timestamp,expected_power_kw

  data/inbox/forecast/forecast_YYYYMMDD_HHMM.csv
    timestamp,solar_forecast_kw

  data/inbox/actual/actual_YYYYMMDD_HHMM.csv
    timestamp,solar_actual_kw

Usage examples:
  # Generate all files for Jan 3, 2026 in batch mode (no sleep):
  python scripts/mimic_streams.py --date 2026-01-03 --mode batch

  # Simulate real-time (sleep each 5-min tick; use --fast to skip sleeping):
  python scripts/mimic_streams.py --date 2026-01-03 --mode realtime
  python scripts/mimic_streams.py --date 2026-01-03 --mode realtime --fast
"""

from __future__ import annotations
import os
import time
import argparse
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ----------------------- Configuration (defaults) -----------------------

PEAK_KW = 60000        # Peak solar power ~60 MW at noon
MU_HOUR = 12.5         # Center of bell curve ~ 12:30
SIGMA_HOUR = 3.5       # Width of bell curve
DAY_AHEAD_BIAS = 0.95  # Day-ahead conservative bias
FORECAST_VAR = 0.10    # +/-10% variability vs bell curve
ACTUAL_VAR = 0.05      # +/-5% around forecast for actuals

DT15_MIN = 15
DT5_MIN = 5

# Folders
BASE_DIR = "data/inbox"
DAY_AHEAD_DIR = BASE_DIR
FORECAST_DIR = os.path.join(BASE_DIR, "forecast")
ACTUAL_DIR = os.path.join(BASE_DIR, "actual")


# ----------------------- Helpers -----------------------

def ensure_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(FORECAST_DIR, exist_ok=True)
    os.makedirs(ACTUAL_DIR, exist_ok=True)


def bell_solar_kw(hour_float: float,
                  peak_kw: float = PEAK_KW,
                  mu_hour: float = MU_HOUR,
                  sigma_hour: float = SIGMA_HOUR) -> float:
    """Gaussian-like solar power profile. Clipped to zero at night."""
    val = peak_kw * math.exp(-0.5 * ((hour_float - mu_hour) / sigma_hour) ** 2)
    # Night zeroing (approx sunrise ~06:00, sunset ~18:30)
    if hour_float < 6.0 or hour_float > 18.5:
        val = 0.0
    return max(0.0, val)


def atomic_write_csv(df: pd.DataFrame, final_path: str):
    """
    Write CSV atomically: write to a temp file and replace the target.
    Avoids partial reads by downstream processes.
    """
    temp_path = final_path + ".tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, final_path)


def date_range_local(date_str: str, freq_min: int) -> pd.DatetimeIndex:
    """
    Create a naive (local) date range at given minute frequency for the full day.
    e.g., 2026-01-03, freq=5 -> 2026-01-03 00:00 ... 23:55
    """
    day_start = pd.Timestamp(date_str + "T00:00:00")
    day_end = pd.Timestamp(date_str + "T23:55:00")
    return pd.date_range(day_start, day_end, freq=f"{freq_min}min")


def make_day_ahead_15min(date_str: str) -> pd.DataFrame:
    """
    Create 15-min day-ahead expected power (kW) for the given date.
    """
    ts15 = date_range_local(date_str, DT15_MIN)
    rows = []
    for ts in ts15:
        hr = ts.hour + ts.minute / 60.0
        base_kw = bell_solar_kw(hr)
        expected_kw = base_kw * DAY_AHEAD_BIAS
        rows.append((ts.isoformat(), round(expected_kw, 2)))
    return pd.DataFrame(rows, columns=["timestamp", "expected_power_kw"])


def make_forecast_5min_for_day(date_str: str, seed: int = 42) -> pd.DataFrame:
    """
    Create 5-min forecast (kW) for the entire day.
    """
    rng = np.random.default_rng(seed)
    ts5 = date_range_local(date_str, DT5_MIN)
    rows = []
    for ts in ts5:
        hr = ts.hour + ts.minute / 60.0
        base_kw = bell_solar_kw(hr)
        noise = (rng.random() - 0.5) * 2.0 * FORECAST_VAR * base_kw  # +/- variability
        forecast_kw = max(0.0, base_kw + noise)
        rows.append((ts.isoformat(), round(forecast_kw, 2)))
    return pd.DataFrame(rows, columns=["timestamp", "solar_forecast_kw"])


def make_actual_5min_upto(date_str: str, upto_ts: pd.Timestamp,
                          forecast_df: pd.DataFrame,
                          seed: int = 123) -> pd.DataFrame:
    """
    Create 5-min actuals (kW) up to a given timestamp (inclusive),
    centered around the forecast values with smaller noise.
    """
    rng = np.random.default_rng(seed)
    # Use timestamps up to upto_ts (inclusive) from the forecast index
    mask = pd.to_datetime(forecast_df["timestamp"]) <= upto_ts
    df_sub = forecast_df.loc[mask].copy()
    # Add small noise around forecast
    actuals = []
    for _, r in df_sub.iterrows():
        fc = float(r["solar_forecast_kw"])
        noise = (rng.random() - 0.5) * 2.0 * ACTUAL_VAR * fc
        actual_kw = max(0.0, fc + noise)
        actuals.append((r["timestamp"], round(actual_kw, 2)))
    return pd.DataFrame(actuals, columns=["timestamp", "solar_actual_kw"])


def day_ahead_path(date_str: str) -> str:
    return os.path.join(DAY_AHEAD_DIR, f"day_ahead_{pd.Timestamp(date_str).strftime('%Y%m%d')}.csv")


def forecast_path(date_str: str, ts: pd.Timestamp) -> str:
    stamp = ts.strftime("%Y%m%d_%H%M")
    return os.path.join(FORECAST_DIR, f"forecast_{stamp}.csv")


def actual_path(date_str: str, ts: pd.Timestamp) -> str:
    stamp = ts.strftime("%Y%m%d_%H%M")
    return os.path.join(ACTUAL_DIR, f"actual_{stamp}.csv")


# ----------------------- Generators (Batch & Realtime) -----------------------

def generate_batch_files(date_str: str):
    """
    Generate day-ahead and a set of forecast & actual files for the entire day at once.
    - Day-ahead: full day 15-min
    - Forecast: one file per 5-min tick (full day forecast in each file)
    - Actual: one file per 5-min tick (actuals up to that tick)
    """
    ensure_dirs()

    # Day-ahead
    df15 = make_day_ahead_15min(date_str)
    atomic_write_csv(df15, day_ahead_path(date_str))
    print(f"[BATCH] Wrote {day_ahead_path(date_str)}")

    # One baseline full-day forecast
    df5f = make_forecast_5min_for_day(date_str, seed=42)

    # For each 5-min tick: write a forecast file (full-day forecast)
    # and an actual file (up to current tick)
    ts5 = date_range_local(date_str, DT5_MIN)
    for ts in ts5:
        fpath = forecast_path(date_str, ts)
        apath = actual_path(date_str, ts)

        # Forecast file: full-day forecast (could also write partial if desired)
        atomic_write_csv(df5f, fpath)

        # Actual file: actuals up to 'ts'
        df5a = make_actual_5min_upto(date_str, ts, df5f, seed=123)
        atomic_write_csv(df5a, apath)

    print(f"[BATCH] Wrote {len(ts5)} forecast files and {len(ts5)} actual files.")


def simulate_realtime(date_str: str, fast: bool = False):
    """
    Simulate real-time operation:
      - Write day-ahead once.
      - Every 5 minutes:
          * Write a new forecast file (full-day forecast with slight random seed change).
          * Write/update actuals up to 'now'.
      - Sleep to emulate wall clock (skip if fast=True).
    """
    ensure_dirs()

    # Day-ahead
    df15 = make_day_ahead_15min(date_str)
    atomic_write_csv(df15, day_ahead_path(date_str))
    print(f"[RT] Wrote {day_ahead_path(date_str)}")

    # Day timeline
    ts5 = date_range_local(date_str, DT5_MIN)

    for i, ts in enumerate(ts5):
        t0 = time.time()

        # Forecast: regenerate full-day forecast each tick with a new seed
        # (mimics updated model input). You can make seed depend on i.
        df5f = make_forecast_5min_for_day(date_str, seed=42 + i)
        atomic_write_csv(df5f, forecast_path(date_str, ts))

        # Actuals: generate up to current ts (append-like behavior)
        df5a = make_actual_5min_upto(date_str, ts, df5f, seed=123 + i)
        atomic_write_csv(df5a, actual_path(date_str, ts))

        print(f"[RT {ts}] forecast={os.path.basename(forecast_path(date_str, ts))}, "
              f"actual={os.path.basename(actual_path(date_str, ts))}, rows_actual={len(df5a)}")

        # Sleep to next 5-min boundary unless fast mode
        if not fast:
            elapsed = time.time() - t0
            sleep_sec = max(0.0, DT5_MIN * 60 - elapsed)
            time.sleep(sleep_sec)


# ----------------------- CLI -----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Mimic CSV streams for Solar+BESS EMS MPC.")
    p.add_argument("--date", type=str, required=True, help="Simulation date (YYYY-MM-DD)")
    p.add_argument("--mode", type=str, choices=["batch", "realtime"], default="batch",
                   help="Generate all files in batch or simulate realtime.")
    p.add_argument("--fast", action="store_true", help="Realtime without sleeping.")
    return p.parse_args()


def main():
    args = parse_args()
    date_str = args.date

    if args.mode == "batch":
        generate_batch_files(date_str)
    else:
        simulate_realtime(date_str, fast=args.fast)


if __name__ == "__main__":
    main()
