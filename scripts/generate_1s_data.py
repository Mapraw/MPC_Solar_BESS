
import numpy as np
import pandas as pd

def simulate_pv_one_day_df(
    date_str="2026-01-26",       # Local calendar date for the simulation
    tz="UTC",                    # Timezone for the index (e.g., "Asia/Bangkok")
    peak_kw=5.0,                 # PV peak power (kW)
    sunrise_h=6, sunrise_m=30,   # Local sunrise (hh:mm)
    sunset_h=18, sunset_m=0,     # Local sunset (hh:mm)
    gamma_shape=1.0,             # 1.0 = half-sine; >1 flattens peak a bit
    add_noise=False,
    noise_std_fraction=0.02,     # 2% of instantaneous power as noise std
    random_seed=42,
    include_energy_wh=True       # Add cumulative energy in Wh
):
    """
    Returns:
        df: pd.DataFrame with DateTimeIndex at 1s frequency
            Columns:
                - power_kW
                - energy_Wh (optional)
    Notes:
        - Power is 0 outside [sunrise, sunset] and exactly 0 at sunrise/sunset.
        - Index is timezone-aware and spans the given date from 00:00:00 to 23:59:59.
    """
    # ---- Time index (one full day at 1-second resolution) ----
    start = pd.Timestamp(f"{date_str} 00:00:00").tz_localize(tz)
    end   = pd.Timestamp(f"{date_str} 23:59:59").tz_localize(tz)
    idx = pd.date_range(start=start, end=end, freq="s")  # 86,400 rows

    # ---- Convert local sunrise/sunset to seconds from midnight ----
    sunrise_sec = sunrise_h * 3600 + sunrise_m * 60
    sunset_sec  = sunset_h * 3600 + sunset_m * 60
    daylen = max(1, sunset_sec - sunrise_sec)  # avoid divide-by-zero

    # ---- Compute power profile in kW ----
    t_s = np.arange(0, 24 * 3600, dtype=np.int32)
    p_kw = np.zeros_like(t_s, dtype=np.float64)

    # Daylight mask including endpoints so power=0 at sunrise and sunset
    daylight = (t_s >= sunrise_sec) & (t_s <= sunset_sec)
    if np.any(daylight):
        x = (t_s[daylight] - sunrise_sec) / daylen  # normalized [0,1]
        x = np.clip(x, 0.0, 1.0)
        profile = np.sin(np.pi * x)  # half-sine: 0 at both ends, 1 at mid
        if gamma_shape != 1.0:
            profile = np.power(profile, gamma_shape)
        p_day_kw = peak_kw * profile

        # Optional noise proportional to instantaneous power
        if add_noise:
            rng = np.random.default_rng(seed=random_seed)
            noise = rng.normal(loc=0.0, scale=noise_std_fraction, size=p_day_kw.shape)
            p_day_kw = np.maximum(0.0, p_day_kw * (1.0 + noise))

        p_kw[daylight] = p_day_kw

    p_kw = np.clip(p_kw, 0.0, None)  # guard tiny negatives

    # ---- Build DataFrame ----
    df = pd.DataFrame({"power_kW": p_kw}, index=idx)
    df.index.name = "time"  # index label

    # ---- Optional: cumulative energy in Wh (trapezoidal 1-second integration) ----
    if include_energy_wh:
        # Since dt = 1 second, Wh increment per second = kW * (1/3600) h
        # Use cumulative sum (left Riemann) which is adequate at 1s resolution.
        df["energy_Wh"] = (df["power_kW"] / 3600.0).cumsum()

    return df

# ---------------- Example usage ----------------
if __name__ == "__main__":
    df = simulate_pv_one_day_df(
        date_str="2026-01-26",
        tz="Asia/Bangkok",
        peak_kw=5.0,
        sunrise_h=6, sunrise_m=30,
        sunset_h=19, sunset_m=0,
        gamma_shape=1.1,
        add_noise=False,
        include_energy_wh=True
    )

    print(df.head(10))
    print("\nAt sunrise:", df.loc[df.index.time == pd.Timestamp('06:30:00').time()].iloc[0])
    print("\nAt sunset:",  df.loc[df.index.time == pd.Timestamp('18:00:00').time()].iloc[0])
    print("\nDaily energy (kWh) ≈", df["energy_Wh"].iloc[-1] / 1000.0)

    # Optional: save to CSV
    df.to_csv("data/day_ahead_1s.csv")
