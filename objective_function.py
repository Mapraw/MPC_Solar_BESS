# -*- coding: utf-8 -*-
"""
fit_revenue_15min_exact_blocks.py

Compute FiT revenue per 15-minute interval when input timestamps are aligned
to exact 15-min blocks (e.g., 00:00, 00:15, 00:30, 00:45, ...).

Implements clauses 17.1–17.3 including the 14/15 adjustment mapped to the
blocks starting at 06:00, 16:00, and 18:00.

Usage:
- Run directly to see a demo.
- Import compute_payment_for_interval(...) for single-interval use.
- Import compute_revenue_rows(...) / load_csv_and_compute(...) for batch use.
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import time, datetime
from typing import Optional, Tuple, Iterable, Union
import pandas as pd

# -------------------------------
# Constants
# -------------------------------
ADJUST_FACTOR = 14.0 / 15.0  # factor for special sub-intervals
PENALTY_RATE = 0.12          # 12%

# -------------------------------
# Window detection & adjustments
# -------------------------------
def _in_range(t: time, start: time, end: time, inclusive_end: bool = True) -> bool:
    if inclusive_end:
        return (t >= start) and (t <= end)
    return (t >= start) and (t < end)

def detect_time_window_for_aligned_blocks(ts: datetime) -> Tuple[int, bool]:
    """
    Return (window_id, adjusted_subinterval) for 15-min aligned timestamps.

    Windows:
      1) 09:00–16:00
      2) 18:01–24:00 and 00:00–06:00
      3) 06:01–09:00 and 16:01–18:00

    Adjustment mapping (because input blocks are aligned at :00/:15/etc):
      - 06:01–06:15 applies to block starting 06:00  -> ts.time() == 06:00 -> adjust 14/15
      - 16:01–16:15 applies to block starting 16:00  -> ts.time() == 16:00 -> adjust 14/15
      - 18:01–18:15 applies to block starting 18:00  -> ts.time() == 18:00 -> adjust 14/15
    """
    t = ts.time()

    # Apply the exact-block mapping for the 14/15 special intervals
    adjusted = (t == time(6, 0)) or (t == time(16, 0)) or (t == time(18, 0))

    # Window 1: 09:00–16:00 (inclusive per text)
    if _in_range(t, time(9, 0), time(16, 0)):
        # Note: even though 16:00 is included in Window 1, the 16:00 block is also a special
        # adjusted block (for 16:01–16:15) but that special clause belongs to Window 3 context.
        # However, the text for Window 3 adjustment specifically names 16:01–16:15, so we still
        # apply 14/15 at 16:00; window classification remains by ranges below.
        return 1, (t == time(16, 0))  # adjust at 16:00 per clause 17.3.3

    # Window 2: 18:01–24:00 and 00:00–06:00
    if (t >= time(0, 0) and t <= time(6, 0)) or (t >= time(18, 1) and t <= time(23, 59, 59)):
        # 18:00 block is special for Window 2 (17.2.4)
        return 2, (t == time(18, 0))

    # Window 3: 06:01–09:00 and 16:01–18:00
    if _in_range(t, time(6, 15), time(9, 0)) or _in_range(t, time(16, 15), time(18, 0), inclusive_end=True):
        # Note: 06:00 and 16:00 are handled above as adjustments on boundary blocks.
        return 3, False

    # If exactly 06:00 → belongs to earlier day’s overnight window (Window 2) but
    # is adjusted for 06:01–06:15 clause → handled above in Window 2 return.
    # If exactly 18:00 → boundary → belongs to Window 3 end but special adjusted for 18:01–18:15 under Window 2. 
    # The mapping above returned Window 2 for 18:00 with adjusted=True.

    # If something slips through (shouldn't), raise.
    raise ValueError(f"Timestamp {ts} does not fall into any defined window.")

# -------------------------------
# Data classes
# -------------------------------
@dataclass
class IntervalInput:
    ts_start: datetime
    e_read_kwh: float          # Energy in this 15-minute block
    fit_rate: float            # Currency per kWh
    contract_kwh: Optional[float] = None
    egat_plan_kwh: Optional[float] = None
    has_egat_plan_in_win3: Optional[bool] = None  # only for Window 3

@dataclass
class IntervalResult:
    window_id: int
    adjusted_subinterval: bool
    e_use_kwh: float
    base_kwh: float
    payable_kwh: float
    shortfall_kwh: float
    penalty_currency: float
    payment_currency: float

# -------------------------------
# Core computation
# -------------------------------
def compute_payment_for_interval(params: IntervalInput) -> IntervalResult:
    """
    Compute revenue for one 15-min block with exact 15-min aligned timestamps.
    Applies:
      - Caps (no payment above base)
      - Penalty 12% * FiT * shortfall when actual < base
      - 14/15 adjustment for blocks starting at 06:00, 16:00, 18:00
    """
    window_id, adjusted = detect_time_window_for_aligned_blocks(params.ts_start)

    # Step 1: adjust energy for special boundary blocks
    e_use = params.e_read_kwh * (ADJUST_FACTOR if adjusted else 1.0)

    # Step 2: determine base by window
    if window_id == 1:
        if params.contract_kwh is None:
            raise ValueError("contract_kwh is required for Window 1")
        base = float(params.contract_kwh)

    elif window_id == 2:
        if params.egat_plan_kwh is None:
            raise ValueError("egat_plan_kwh is required for Window 2")
        base = float(params.egat_plan_kwh)

    else:  # window 3
        if params.has_egat_plan_in_win3 is None:
            raise ValueError("has_egat_plan_in_win3 must be specified for Window 3")
        if params.has_egat_plan_in_win3:
            if params.egat_plan_kwh is None:
                raise ValueError("egat_plan_kwh is required for Window 3 when has_egat_plan_in_win3=True")
            base = float(params.egat_plan_kwh)
        else:
            if params.contract_kwh is None:
                raise ValueError("contract_kwh is required for Window 3 when has_egat_plan_in_win3=False")
            base = float(params.contract_kwh)

    # Step 3: compute payable and penalty
    if e_use > base:
        payable = base
        shortfall = 0.0
        penalty = 0.0
    else:
        payable = e_use
        shortfall = max(base - e_use, 0.0)
        penalty = shortfall * params.fit_rate * PENALTY_RATE

    payment = payable * params.fit_rate - penalty

    return IntervalResult(
        window_id=window_id,
        adjusted_subinterval=adjusted,
        e_use_kwh=e_use,
        base_kwh=base,
        payable_kwh=payable,
        shortfall_kwh=shortfall,
        penalty_currency=penalty,
        payment_currency=payment,
    )

# -------------------------------
# Batch helpers (list / CSV)
# -------------------------------
def _row_to_input(row: dict) -> IntervalInput:
    ts = pd.to_datetime(row["ts_start"]).to_pydatetime()
    def _opt_float(v):
        return float(v) if (v is not None and not pd.isna(v)) else None
    def _opt_bool(v):
        return bool(v) if v is not None and v is not pd.NA else None

    return IntervalInput(
        ts_start=ts,
        e_read_kwh=float(row["e_read_kwh"]),
        fit_rate=float(row["fit_rate"]),
        contract_kwh=_opt_float(row.get("contract_kwh")),
        egat_plan_kwh=_opt_float(row.get("egat_plan_kwh")),
        has_egat_plan_in_win3=_opt_bool(row.get("has_egat_plan_in_win3")),
    )

def compute_revenue_rows(data: Union[pd.DataFrame, Iterable[dict]]) -> pd.DataFrame:
    """
    Compute 15-minute revenue for each row.

    Required columns: ts_start (aligned to :00, :15, :30, :45), e_read_kwh, fit_rate
    Optional columns by window:
      - Window 1: contract_kwh
      - Window 2: egat_plan_kwh
      - Window 3: has_egat_plan_in_win3 + (contract_kwh or egat_plan_kwh)
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(list(data))

    for col in ["ts_start", "e_read_kwh", "fit_rate"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    outputs = []
    for _, row in df.iterrows():
        res = compute_payment_for_interval(_row_to_input(row))
        outputs.append({
            "window_id": res.window_id,
            "adjusted_subinterval": res.adjusted_subinterval,
            "e_use_kwh": res.e_use_kwh,
            "base_kwh": res.base_kwh,
            "payable_kwh": res.payable_kwh,
            "shortfall_kwh": res.shortfall_kwh,
            "penalty_currency": res.penalty_currency,
            "payment_currency": res.payment_currency,
        })

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(outputs)], axis=1)
    return out

def load_csv_and_compute(csv_path: str, datetime_column: str = "ts_start") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    return compute_revenue_rows(df)

def save_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)