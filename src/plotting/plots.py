
# src/plotting/plots.py
# Visualization utilities for Solar+BESS EMS MPC:
#   - Power tracking (grid output vs target vs solar forecast)
#   - Battery SOC
#   - Energy-domain diagnostics per 15-minute block
#   - Optional summary of block energy errors
#
# Usage:
#   from src.plotting.plots import plot_day, plot_soc, plot_block_energy, plot_block_energy_errors_summary

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Optional


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    """Ensure df[col] is a pandas datetime dtype."""
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col])


def plot_day(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot daily power tracking:
      - Solar (5-min forecast)
      - Target (15-min expected, repeated across each block)
      - Grid output (solar + battery)
      - Battery power (step plot)

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with columns:
        ['timestamp', 'solar_forecast_kw', 'target_power_kw',
         'grid_output_kw', 'battery_power_kw']
    save_path : str | None
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.pyplot
    """
    for col in ["timestamp"]:
        _ensure_datetime(df, col)

    plt.figure(figsize=(14, 8))
    plt.plot(df["timestamp"], df["solar_forecast_kw"], label="Solar (forecast, 5-min)", color="orange", linewidth=1.5)
    plt.plot(df["timestamp"], df["target_power_kw"], label="Target (15-min expected)", color="green", linestyle="--", linewidth=1.5)
    plt.plot(df["timestamp"], df["grid_output_kw"], label="Grid Output (solar + BESS)", color="blue", linewidth=1.8)
    plt.step(df["timestamp"], df["battery_power_kw"], where="post", label="BESS Power (AC kW)", color="purple", linewidth=1.5)

    plt.title("EMS MPC — Power Tracking (Solar + BESS → Target)")
    plt.xlabel("Time")
    plt.ylabel("Power (kW)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt


def plot_soc(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot battery state of charge (SOC) over the day.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with columns:
        ['timestamp', 'soc_kwh']
    save_path : str | None
        If provided, saves the figure to this path.

    Returns
    -------
    matplotlib.pyplot
    """
    for col in ["timestamp"]:
        _ensure_datetime(df, col)

    plt.figure(figsize=(14, 4))
    plt.plot(df["timestamp"], df["soc_kwh"], label="SOC (kWh)", color="black", linewidth=1.8)
    plt.title("Battery State of Charge (SOC)")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt


def plot_block_energy(df: pd.DataFrame, block_start_ts: pd.Timestamp, save_path: Optional[str] = None):
    """
    Plot cumulative energy within a 15-min block:
      - Uses solar_actual_kw if available, else solar_forecast_kw
      - Adds battery energy for each 5-min step
      - Compares cumulative (solar + battery) with E_target_kwh

    Parameters
    ----------
    df : pd.DataFrame
        The results DataFrame returned by the simulator
        (must contain columns:
         'timestamp', 'block_start', 'substep_in_block',
         'solar_forecast_kw', 'solar_actual_kw', 'battery_power_kw', 'E_target_kwh').
    block_start_ts : pd.Timestamp
        The 15-min block start timestamp to diagnose (e.g., '2026-01-03 06:00:00').
    save_path : str | None
        Optional file path to save the plot image.

    Returns
    -------
    matplotlib.pyplot
    """
    for col in ["timestamp", "block_start"]:
        _ensure_datetime(df, col)

    dt_h = 5 / 60.0  # 5 minutes in hours
    blk = df[df["block_start"] == block_start_ts].sort_values("substep_in_block").copy()
    if blk.empty:
        raise ValueError(f"No rows found for block_start {block_start_ts}")

    # Use actual solar where available; fallback to forecast otherwise
    solar_kw = blk["solar_actual_kw"].where(blk["solar_actual_kw"].notna(), blk["solar_forecast_kw"])
    batt_kw = blk["battery_power_kw"]

    # Per-step energy contributions (kWh)
    step_energy_kwh = solar_kw * dt_h + batt_kw * dt_h
    cum_energy_kwh = step_energy_kwh.cumsum()

    # Target energy for the block (constant across its 3 rows)
    target_E_kwh = float(blk["E_target_kwh"].iloc[0])

    # Plot
    plt.figure(figsize=(9, 5))
    # Cumulative step curve
    plt.step(blk["timestamp"], cum_energy_kwh, where="post",
             label="Cumulative (Solar + BESS)", color="blue", linewidth=2.0)
    # Target line
    plt.hlines(target_E_kwh,
               xmin=blk["timestamp"].min(), xmax=blk["timestamp"].max(),
               colors="green", linestyles="--",
               label=f"Target Energy ({target_E_kwh:.1f} kWh)")

    # Optional overlays for clarity: per-step energy bars (solar and battery)
    # Narrow-width bars to avoid overlapping the step curve
    bar_width_days = (5.0 / (24 * 60)) * 0.6  # 5 minutes in days * shrink factor
    plt.bar(blk["timestamp"], (solar_kw * dt_h), width=bar_width_days,
            alpha=0.35, color="orange", label="Solar step (kWh)")
    plt.bar(blk["timestamp"], (batt_kw * dt_h), width=bar_width_days,
            alpha=0.35, color="purple", label="BESS step (kWh)")

    plt.title(f"Block Energy Diagnostic: {block_start_ts}")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt


def plot_block_energy_errors_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    top_n: int = 10,
    ascending: bool = False,
):
    """
    Summarize block energy errors (delivered vs target) across the day.

    For each 15-min block, compute:
      E_delivered_kwh = sum((solar_actual_or_forecast + battery_power) * dt_h) over 3 substeps
      E_target_kwh    = constant per block
      error_kwh       = E_delivered_kwh - E_target_kwh

    Then plot the top-N blocks with largest absolute error (default: descending).

    Parameters
    ----------
    df : pd.DataFrame
        Simulation results with columns:
        ['block_start', 'substep_in_block',
         'solar_forecast_kw', 'solar_actual_kw', 'battery_power_kw', 'E_target_kwh'].
    save_path : str | None
        If provided, saves the figure to this path.
    top_n : int
        Number of blocks to show (sorted by |error_kwh|).
    ascending : bool
        If False (default), show largest errors first.

    Returns
    -------
    matplotlib.pyplot
    """
    for col in ["block_start"]:
        _ensure_datetime(df, col)

    dt_h = 5 / 60.0

    # Use actual if available; otherwise forecast
    solar_kw = df["solar_actual_kw"].where(df["solar_actual_kw"].notna(), df["solar_forecast_kw"])
    step_energy_kwh = solar_kw * dt_h + df["battery_power_kw"] * dt_h

    # Aggregate by block
    agg = (
        pd.DataFrame({
            "block_start": df["block_start"],
            "step_energy_kwh": step_energy_kwh,
            "E_target_kwh": df["E_target_kwh"],
        })
        .groupby("block_start", as_index=False)
        .agg(E_delivered_kwh=("step_energy_kwh", "sum"),
             E_target_kwh=("E_target_kwh", "first"))
    )
    agg["error_kwh"] = agg["E_delivered_kwh"] - agg["E_target_kwh"]
    agg["abs_error_kwh"] = agg["error_kwh"].abs()

    # Select top-N by absolute error
    agg_sorted = agg.sort_values("abs_error_kwh", ascending=ascending).head(top_n)

    # Plot bar chart
    plt.figure(figsize=(12, 5))
    colors = np.where(agg_sorted["error_kwh"] >= 0, "tab:blue", "tab:red")
    plt.bar(agg_sorted["block_start"].dt.strftime("%H:%M"),
            agg_sorted["error_kwh"], color=colors)

    plt.title(f"Top {len(agg_sorted)} Block Energy Errors (kWh)")
    plt.xlabel("Block start (HH:MM)")
    plt.ylabel("Error (kWh) — delivered minus target")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt
