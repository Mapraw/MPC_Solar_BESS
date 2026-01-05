
# online.py
# Run the EMS MPC online: poll latest CSVs and execute one tick per 5 minutes

from __future__ import annotations
import time
import pandas as pd
from importlib import import_module

from src.runtime.realtime_runner import RealTimeEMS

def align_to_next_5min(now: pd.Timestamp, dt_min: int = 5) -> pd.Timestamp:
    """
    Given current datetime (now), return the next timestamp aligned to 5-minute boundaries.
    """
    minute = ((now.minute // dt_min) + 1) * dt_min
    if minute >= 60:
        # roll hour
        next_ts = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
    else:
        next_ts = now.replace(minute=minute, second=0, microsecond=0)
    return next_ts

def main():
    CONFIG = import_module("config").CONFIG
    runner = RealTimeEMS(CONFIG)

    # Start from day_start aligned 5-min boundary
    t_now = pd.Timestamp(CONFIG["time"]["day_start"])

    # Loop until day_end inclusive
    while t_now <= pd.Timestamp(CONFIG["time"]["day_end"]):
        t0_wall = time.time()
        try:
            result = runner.tick(t_now)
            print(f"[{t_now}] p_bess={result['battery_power_kw']:.0f} kW, "
                  f"grid={result['grid_output_kw']:.0f} kW, SOC={result['soc_kwh']:.0f} kWh")
            print(result)
        except Exception as e:
            # If inputs not ready, log and continue; fallback to forecast-only handled inside
            print(f"[{t_now}] Tick failed: {e}")

        # Sleep until next 5-min boundary (real-time mode; in test just advance immediately)
        next_t = t_now + pd.Timedelta(minutes=CONFIG["time"]["dt_minutes_rtu"])
        # Real systems might poll every ~30s inside the 5-min window to catch new files.
        # Here we sleep the remaining time (minus compute time).
        elapsed = time.time() - t0_wall
        sleep_sec = max(0.0, CONFIG["time"]["dt_minutes_rtu"] * 60 - elapsed)
        time.sleep(sleep_sec)

        t_now = next_t

if __name__ == "__main__":
    main()
