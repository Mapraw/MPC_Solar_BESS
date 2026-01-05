
import math, random, os, csv
from datetime import datetime, timedelta
from config import CONFIG


def main():
    random.seed(42)
    start = datetime.fromisoformat(CONFIG["time"]["day_start"])
    end = datetime.fromisoformat(CONFIG["time"]["day_end"])
    dt15 = timedelta(minutes=CONFIG["time"]["dt_minutes_day_ahead"])
    dt5 = timedelta(minutes=CONFIG["time"]["dt_minutes_rtu"])
    os.makedirs("data", exist_ok=True)

    # Day-ahead expected (15-min)
    with open("data/day_ahead_15min.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "expected_power_kw"])
        cur = start
        while cur <= end:
            hr = cur.hour + cur.minute / 60.0
            peak_kw = 60000
            mu, sigma = 12.5, 3.5
            solar_kw = peak_kw * math.exp(-0.5 * ((hr - mu) / sigma) ** 2)
            if hr < 6 or hr > 18.5: solar_kw = 0.0
            expected_kw = solar_kw * 0.95
            w.writerow([cur.isoformat(), f"{expected_kw:.2f}"])
            cur += dt15

    # Real-time forecast (5-min)
    with open("data/rt_5min.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "solar_forecast_kw"])
        cur = start
        while cur <= end:
            hr = cur.hour + cur.minute / 60.0
            peak_kw = 60000
            mu, sigma = 12.5, 3.5
            base_kw = peak_kw * math.exp(-0.5 * ((hr - mu) / sigma) ** 2)
            if hr < 6 or hr > 18.5: base_kw = 0.0
            noise = (random.random() - 0.5) * 0.2 * base_kw
            forecast_kw = max(0.0, base_kw + noise)
            w.writerow([cur.isoformat(), f"{forecast_kw:.2f}"])
            cur += dt5

    print("Example data written to data/day_ahead_15min.csv and data/rt_5min.csv")

if __name__ == "__main__":
    main()
