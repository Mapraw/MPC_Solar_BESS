import pandas as pd

# ---------------- Config ----------------
IN_FILE  = "day_ahead_1s.csv"
OUT_FILE = "day_ahead_15min_conv.csv"

# Mapping of column -> aggregation at 15-minute level
# Customize to your dataset:
AGG_MAP = {
    "power_kW":  "mean",  # typical for instantaneous power readings
    "energy_Wh": "sum",   # typical for energy increments
    # Add more columns here as needed, e.g.:
    # "voltage_V": "mean",
    # "current_A": "mean",
    # "pf": "mean",
}

# Resample params (adjust if you need different alignment rules)
FREQ       = "15min"   # 15-minute frequency; alternatives: "900S"
LABEL      = "left"  # label the window at the start timestamp
CLOSE      = "left"  # interval closure: (left, right, both, neither)
DROP_TZ    = False   # set True to strip timezone before saving
# ----------------------------------------


def main():
    # 1) Load
    df = pd.read_csv(IN_FILE, parse_dates=["time"])
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column in the CSV.")

    # 2) Index by time
    df = df.set_index("time")
    
    # If timestamps are naive but represent local time, you can localize:
    # df = df.tz_localize("Asia/Bangkok")

    # 3) Ensure all needed columns exist
    missing = [c for c in AGG_MAP.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for aggregation: {missing}")

    # 4) Resample to 15-minute windows
    #    You can also decide fill strategy before resampling if needed.
    #    For example, if energy_Wh is cumulative instead of per-second increments,
    #    you'd compute the difference first. (See notes below.)
    out = (
        df
        .resample(FREQ, label=LABEL, closed=CLOSE)
        .agg(AGG_MAP)
    )

    # 5) Optional: fill gaps if you want every 15-min bin across full range
    # out = out.asfreq(FREQ)  # creates bins with NaNs if missing; fill as you wish:
    # out["power_kW"] = out["power_kW"].fillna(0)
    # out["energy_Wh"] = out["energy_Wh"].fillna(0)

    # 6) Optional: drop timezone for CSVs that must be tz-naive
    if DROP_TZ and out.index.tz is not None:
        out.index = out.index.tz_convert(None)

    # 7) Save
    out.to_csv(OUT_FILE)
    print(f"Saved aggregated 15-min file → {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()