
import os, time, math
import pandas as pd
import numpy as np
from datetime import timedelta

PEAK_KW = 45000
MU_HOUR = 12.5
SIGMA_HOUR = 3.5

def bell(hr):
    val = PEAK_KW * math.exp(-0.5 * ((hr - MU_HOUR)/SIGMA_HOUR)**2)
    return 0.0 if hr < 6.0 or hr > 18.5 else val

def atomic_write(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def main(date_str="2026-01-03", fast=True):
    # Ensure folders
    os.makedirs("data/inbox/forecast", exist_ok=True)
    os.makedirs("data/inbox/actual", exist_ok=True)
    os.makedirs("data/inbox", exist_ok=True)

    # Day-ahead (full day, 15-min)
    ts15 = pd.date_range(f"{date_str}T00:00:00", f"{date_str}T23:55:00", freq="15min")
    da_rows = []
    for ts in ts15:
        hr = ts.hour + ts.minute/60.0
        da_rows.append([ts.isoformat(), round(bell(hr)*0.95, 2)])
    day_ahead = pd.DataFrame(da_rows, columns=["timestamp","expected_power_kw"])
    atomic_write(day_ahead, f"data/inbox/day_ahead_{pd.Timestamp(date_str).strftime('%Y%m%d')}.csv")

    # Simulate from 06:00 to 18:15
    t = pd.Timestamp(f"{date_str}T06:00:00")
    end = pd.Timestamp(f"{date_str}T18:15:00")
    # t = pd.Timestamp(f"{date_str}T06:05:00")
    # end = pd.Timestamp(f"{date_str}T18:50:00")
    rng = np.random.default_rng(42)

    while t <= end:
        # Forecast for next 3 steps: t, t+5, t+10 (use bell + variability)
        next_ts = pd.date_range(t, t + timedelta(minutes=10), freq="5min")
        fc_rows = []
        for ts in next_ts:
            hr = ts.hour + ts.minute/60.0
            base = bell(hr)
            noise = (rng.random() - 0.5) * 0.2 * base
            fc_rows.append([ts.isoformat(), round(max(0.0, base + noise), 2)])
        df_fc = pd.DataFrame(fc_rows, columns=["timestamp","solar_forecast_kw"])
        fc_path = f"data/inbox/forecast/forecast_{ts.strftime('%Y%m%d_%H%M')}.csv"
        atomic_write(df_fc, fc_path)

        # Actual for current t only (centered around forecast[t])
        fc_now = float(df_fc[df_fc["timestamp"]==t.isoformat()]["solar_forecast_kw"])
        actual_now = max(0.0, fc_now + (rng.random() - 0.5) * 0.1 * fc_now)
        df_act = pd.DataFrame([[t.isoformat(), round(actual_now, 2)]], columns=["timestamp","solar_actual_kw"])
        act_path = f"data/inbox/actual/actual_{t.strftime('%Y%m%d_%H%M')}.csv"
        atomic_write(df_act, act_path)

        print(f"[{t}] wrote {os.path.basename(fc_path)} & {os.path.basename(act_path)}")

        # Advance
        if not fast:
            time.sleep(5*60)  # sleep to next tick
        t += timedelta(minutes=5)

if __name__ == "__main__":
    main()
