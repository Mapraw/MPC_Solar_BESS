# solar_reader.py
import pandas as pd

class SolarData:
    """
    Simple solar CSV loader + helper functions.
    Designed for 1-second simulation loops.
    """

    def __init__(self, df: pd.DataFrame, value_col: str):
        self.df = df
        self.value_col = value_col

    @classmethod
    def from_csv(cls, path: str, time_col: str = None, value_col: str = "power_kw"):
        """
        Load solar data from CSV.
        Auto-detect time column if not given.
        """

        raw = pd.read_csv(path)

        # Auto-detect time column
        if time_col is None:
            for c in raw.columns:
                try:
                    pd.to_datetime(raw[c])
                    time_col = c
                    break
                except:
                    continue
            if time_col is None:
                raise ValueError("No datetime column found.")

        # Parse timestamps
        raw[time_col] = pd.to_datetime(raw[time_col])

        # Build dataframe with datetime index
        df = raw.set_index(time_col).sort_index()

        # Ensure value column exists
        if value_col not in df.columns:
            raise ValueError(f"Column {value_col} not in CSV.")

        return cls(df, value_col)

    # ---------------- Simulation helpers ---------------- #

    def get_value(self, timestamp):
        """Return the value at an exact timestamp."""
        try:
            return float(self.df.at[timestamp, self.value_col])
        except:
            return None

    def get_range(self, start, end):
        """Slice data between timestamps."""
        return self.df.loc[start:end][self.value_col]

    def get_by_index(self, start_idx, end_idx):
        """Slice by integer index."""
        return self.df.iloc[start_idx:end_idx][self.value_col]

    def stream(self, dt_s=1):
        """
        Stream data in dt-second steps (default = 1 second).
        Yields (timestamp, value).
        """
        if dt_s == 1:
            for ts, row in self.df.iterrows():
                yield ts, float(row[self.value_col])
        else:
            # Downsample using mean (power signal)
            grouped = self.df[self.value_col].resample(f"{dt_s}S").mean()
            for ts, val in grouped.items():
                yield ts, float(val)