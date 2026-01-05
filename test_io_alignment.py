
from config import CONFIG
import pandas as pd
from src.io.data_loader import (
    read_day_ahead_power_15min, read_forecast_5min, read_actual_5min, build_tracking_frame
)

df15 = read_day_ahead_power_15min("data/day_ahead_15min.csv")
df5f = read_forecast_5min("data/rt_5min.csv")

# If you have actuals (optional), include them; otherwise pass None
try:
    df5a = read_actual_5min("data/solar_actual_5min.csv")
except FileNotFoundError:
    df5a = None

tracking = build_tracking_frame(df15, df5f, df5a, dt5_min=CONFIG["time"]["dt_minutes_rtu"])

print(tracking.head(12))

# Inspect the 06:00 block (substeps at 06:00, 06:05, 06:10)
blk = tracking[tracking["block_start"] == pd.Timestamp("2026-01-03T06:00:00")]
print("\n06:00 block:")
# print(blk[["timestamp", "substep_in_block", "E_target_kwh", "solar_actual_kw", "solar_forecast_kw", "actual_available"]])
print(blk[["timestamp", "E_target_kwh", "solar_actual_kw", "solar_forecast_kw"]])