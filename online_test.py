
# online.py
# Run the EMS MPC online: poll latest CSVs and execute one tick per 5 minutes

from __future__ import annotations
import time
import pandas as pd
from importlib import import_module
import os 
from src.runtime.realtime_runner import RealTimeEMS
from src.plotting.plots import plot_day, plot_soc, plot_block_energy, plot_block_energy_errors_summary

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
    online_index = 0
    results = []
    # Loop until day_end inclusive
    while t_now <= pd.Timestamp(CONFIG["time"]["day_end"]):
        t0_wall = time.time()
        try:
            result = runner.tick(t_now, online_index)
            results.append(result)
            print(f"[{t_now}] p_bess={result['battery_power_kw']:.0f} kW, "
                  f"grid={result['grid_output_kw']:.0f} kW, SOC={result['soc_kwh']:.0f} kWh")
            # print(result)
        except Exception as e:
            # If inputs not ready, log and continue; fallback to forecast-only handled inside
            print(f"[{t_now}] Tick failed: {e}")

        # Sleep until next 5-min boundary (real-time mode; in test just advance immediately)
        next_t = t_now + pd.Timedelta(minutes=CONFIG["time"]["dt_minutes_rtu"])
        # Real systems might poll every ~30s inside the 5-min window to catch new files.
        # Here we sleep the remaining time (minus compute time).
        elapsed = time.time() - t0_wall
        sleep_sec = max(0.0, CONFIG["time"]["dt_minutes_rtu"] * 2 - elapsed)
        # time.sleep(sleep_sec)

        t_now = next_t
        online_index += 1
        print("............................")

    ##################### plot result #############
    df = pd.DataFrame(results)

    # Save CSV results
    out_csv = os.path.join(CONFIG["tracking"]["log_dir"], "mpc_block_energy_results.csv")
    df.to_csv(out_csv, index=False)

    # Plots
    if CONFIG["tracking"]["save_plots"]:
        plot_day(df, os.path.join(CONFIG["tracking"]["log_dir"], "power_tracking.png"))
        plot_soc(df, os.path.join(CONFIG["tracking"]["log_dir"], "soc.png"))

    # Summary: tracking error (power-domain; quick view)
    mae_kw = (df["grid_output_kw"] - df["target_power_kw"]).abs().mean()
    print(f"Mean Abs Tracking Error (power-domain): {mae_kw:,.0f} kW")
    print(f"Saved results to: {out_csv}")


    # main.py (after plotting day/SOC)
    # Block energy diagnostics (e.g., 06:00 and 09:00 blocks)
    for hh in ["06:00:00", "09:00:00"]:
        ts_block = pd.Timestamp(f"{df['timestamp'].dt.date.iloc[0]} {hh}")
        out_png = os.path.join(CONFIG["tracking"]["log_dir"], f"block_{hh.replace(':','')}_energy.png")
        try:
            plot_block_energy(df, ts_block, out_png)
            print(f"Saved block energy diagnostic: {out_png}")
        except Exception as e:
            print(f"Block {hh} diagnostic skipped: {e}")
    
    # Summary of worst block energy errors
    plot_block_energy_errors_summary(
        df,
        os.path.join(CONFIG["tracking"]["log_dir"], "block_energy_errors_summary.png"),
        top_n=12,
        ascending=False
    )



if __name__ == "__main__":
    main()
