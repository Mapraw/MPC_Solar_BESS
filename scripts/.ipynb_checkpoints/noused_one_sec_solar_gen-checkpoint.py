
#!/usr/bin/env python3
"""
Generate 1-second interval solar power data for 1 day.

Outputs:
  - data/day_ahead_1s.csv       -> ["timestamp", "expected_power_kw"]
  - data/rt_1s.csv              -> ["timestamp", "solar_forecast_kw"]

Model:
  - Day-ahead (expected): A smooth Gaussian-shaped solar curve clipped to 0 at night,
    derated by 5% from an ideal peak.
  - Real-time forecast: Adds noise and slow-varying "cloud" modulation to the base curve.

Configuration:
  Edit the CONFIG dict below as needed.
"""

import os
import csv
import math
import random
from datetime import datetime, timedelta, timezone

# -----------------------------
# Configuration (edit as needed)
# -----------------------------
CONFIG = {
    "time": {
        # ISO 8601 local times; they will be treated as naive (no tz) or use tz_offset_minutes below
        "day_start": "2026-01-23T00:00:00",
        "day_end":   "2026-01-23T23:59:59",
        # Timezone offset in minutes (e.g., GMT+7 -> 420). Only used for display in ISO strings.
        "tz_offset_minutes": 420,
        # Output interval (seconds) -> set to 1 for 1-second data
        "dt_seconds": 1,
    },
    "solar_model": {
        # Ideal clear-sky peak AC output in kW (adjust to your plant size)
        "peak_kw": 45000.0,
        # Gaussian center (hour of day) and spread controlling the bell shape
        "mu_hour": 12.5,
        "sigma_hour": 3.5,
        # Nighttime cutoffs (outside these hours, power = 0)
        "sunrise_hour": 6.0,
        "sunset_hour": 18.5,
        # Day-ahead derate factor (e.g., 0.95 = 5% derate from ideal)
        "day_ahead_derate": 0.95,
    },
    "forecast_noise": {
        # Seed for reproducibility (set to None for random each run)
        "seed": 42,
        # Instant random noise amplitude as a fraction of base power (e.g., 0.2 = ±10% range)
        "instant_noise_frac": 0.20,
        # Slow "cloud" modulation: target fraction range (e.g., 0.7–1.0 multiplier on base)
        "cloud_min_frac": 0.70,
        "cloud_max_frac": 1.00,
        # How often cloud target changes (seconds). Larger -> slower variations.
        "cloud_target_update_sec": 180,  # every 3 minutes
        # How quickly the cloud factor eases toward the target (0..1 per step)
        "cloud_smoothing_alpha": 0.002,  # small alpha -> smooth transitions
    },
    "output": {
        "directory": "data",
        "day_ahead_filename": "day_ahead_1s.csv",
        "rt_filename": "rt_1s.csv",
    },
}


def _naive_to_tzaware(dt_naive: datetime, tz_offset_minutes: int) -> datetime:
    """Attach a fixed-offset timezone to a naive datetime (for ISO formatting)."""
    tz = timezone(timedelta(minutes=tz_offset_minutes))
    return dt_naive.replace(tzinfo=tz)


def _gaussian_power(hour_float: float, peak_kw: float, mu: float, sigma: float,
                    sunrise: float, sunset: float) -> float:
    """Gaussian-shaped solar curve clipped outside [sunrise, sunset]."""
    if hour_float < sunrise or hour_float > sunset:
        return 0.0
    # Gaussian bell curve
    return peak_kw * math.exp(-0.5 * ((hour_float - mu) / sigma) ** 2)


def _format_iso(dt: datetime) -> str:
    """ISO 8601 string with timezone if present."""
    return dt.isoformat()


def generate_day_ahead_and_rt():
    # Seed RNG if requested
    seed = CONFIG["forecast_noise"]["seed"]
    if seed is not None:
        random.seed(seed)

    # Parse times
    start = datetime.fromisoformat(CONFIG["time"]["day_start"])
    end = datetime.fromisoformat(CONFIG["time"]["day_end"])
    dt = timedelta(seconds=CONFIG["time"]["dt_seconds"])
    tz_offset = CONFIG["time"]["tz_offset_minutes"]

    # Solar model parameters
    peak_kw = CONFIG["solar_model"]["peak_kw"]
    mu = CONFIG["solar_model"]["mu_hour"]
    sigma = CONFIG["solar_model"]["sigma_hour"]
    sunrise = CONFIG["solar_model"]["sunrise_hour"]
    sunset = CONFIG["solar_model"]["sunset_hour"]
    day_ahead_derate = CONFIG["solar_model"]["day_ahead_derate"]

    # Forecast noise parameters
    instant_noise_frac = CONFIG["forecast_noise"]["instant_noise_frac"]
    cloud_min = CONFIG["forecast_noise"]["cloud_min_frac"]
    cloud_max = CONFIG["forecast_noise"]["cloud_max_frac"]
    cloud_update_sec = CONFIG["forecast_noise"]["cloud_target_update_sec"]
    cloud_alpha = CONFIG["forecast_noise"]["cloud_smoothing_alpha"]

    # Output setup
    out_dir = CONFIG["output"]["directory"]
    os.makedirs(out_dir, exist_ok=True)
    day_ahead_path = os.path.join(out_dir, CONFIG["output"]["day_ahead_filename"])
    rt_path = os.path.join(out_dir, CONFIG["output"]["rt_filename"])

    # Prepare cloud modulation state
    cloud_factor = 1.0  # starts clear
    cloud_target = 1.0
    next_cloud_target_update = start

    # Open writers
    with open(day_ahead_path, "w", newline="") as f_day, open(rt_path, "w", newline="") as f_rt:
        day_writer = csv.writer(f_day)
        rt_writer = csv.writer(f_rt)
        day_writer.writerow(["timestamp", "expected_power_kw"])
        rt_writer.writerow(["timestamp", "solar_forecast_kw"])

        cur = start
        while cur <= end:
            # Attach timezone for output formatting (doesn't change math)
            cur_tz = _naive_to_tzaware(cur, tz_offset)

            # Time-of-day in hours
            hr = cur.hour + cur.minute / 60.0 + cur.second / 3600.0

            # Base ideal curve
            base_kw = _gaussian_power(hr, peak_kw, mu, sigma, sunrise, sunset)

            # Day-ahead expected (derated)
            expected_kw = base_kw * day_ahead_derate

            # Update cloud target occasionally to simulate passing clouds
            if cur >= next_cloud_target_update:
                cloud_target = random.uniform(cloud_min, cloud_max)
                next_cloud_target_update = cur + timedelta(seconds=cloud_update_sec)

            # Smoothly move cloud_factor toward cloud_target
            cloud_factor = (1 - cloud_alpha) * cloud_factor + cloud_alpha * cloud_target

            # Apply cloud modulation
            modulated_kw = base_kw * cloud_factor

            # Apply instantaneous noise (higher power -> proportionally larger noise absolute value)
            # Range ~ ±(instant_noise_frac/2) of base (since (rand-0.5) * frac)
            instant_noise = (random.random() - 0.5) * instant_noise_frac * modulated_kw
            forecast_kw = max(0.0, modulated_kw + instant_noise)

            # Write rows
            day_writer.writerow([_format_iso(cur_tz), f"{expected_kw:.2f}"])
            rt_writer.writerow([_format_iso(cur_tz), f"{forecast_kw:.2f}"])

            cur += dt

    print(f"Wrote:\n  - {day_ahead_path}\n  - {rt_path}")


if __name__ == "__main__":
    generate_day_ahead_and_rt()
