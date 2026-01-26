
# src/runtime/realtime_runner.py
# Real-time EMS runner that persists battery state and computes one MPC action per 5-min tick

from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from importlib import import_module

from src.io.data_loader import (
    read_day_ahead_power_15min, build_tracking_frame, floor_to_15min
)
from src.models.battery import BatteryParams, BatteryState
from src.runtime.utils import (
    latest_csv, safe_read_csv, dedup_and_clip_to_day,
    forward_fill_day_ahead_to_5min, merge_forecast_actual, ontime_csv
)

class RealTimeEMS:
    def __init__(self, CONFIG: Dict):
        self.CONFIG = CONFIG
        self.dt_minutes = CONFIG["time"]["dt_minutes_rtu"]
        self.dt_h = self.dt_minutes / 60.0
        self.log_dir = CONFIG["tracking"]["log_dir"]
        os.makedirs(self.log_dir, exist_ok=True)

        # Build battery & state
        self.batt = BatteryParams(**CONFIG["battery"])
        self.state = BatteryState(self.batt)

        # Choose controller
        use_qp = CONFIG.get("mpc", {}).get("use_qp", False)
        if use_qp:
            from src.mpc.qp_block_mpc import QPBlockEnergyMPC
            w = CONFIG.get("mpc", {}).get("qp_weights", {})
            self.ctrl = QPBlockEnergyMPC(
                dt_minutes=self.dt_minutes,
                p_discharge_max_kw=self.batt.p_discharge_max_kw,
                p_charge_max_kw=self.batt.p_charge_max_kw,
                eta_charge=self.batt.eta_charge,
                eta_discharge=self.batt.eta_discharge,
                ramp_rate_kw_per_step=CONFIG["time"].get("ramp_rate_kw_per_step"),
                w_track=w.get("w_track", 1.0),
                w_mag=w.get("w_mag", 1e-4),
                w_smooth=w.get("w_smooth", 1e-2),
                w_block_energy=w.get("w_block_energy", 10.0),
                w_terminal_soc=w.get("w_terminal_soc", 0.5),
                soc_terminal_kwh=CONFIG["battery"].get("soc_terminal_kwh"),
            )
            self.use_qp = True
        else:
            from src.mpc.mpc_controller import BlockEnergyMPC
            self.ctrl = BlockEnergyMPC(
                dt_minutes=self.dt_minutes,
                p_discharge_max_kw=self.batt.p_discharge_max_kw,
                p_charge_max_kw=self.batt.p_charge_max_kw,
                eta_charge=self.batt.eta_charge,
                eta_discharge=self.batt.eta_discharge,
                ramp_rate_kw_per_step=CONFIG["time"].get("ramp_rate_kw_per_step"),
                soc_terminal_kwh=CONFIG["battery"].get("soc_terminal_kwh"),
                terminal_weight=CONFIG.get("mpc", {}).get("terminal_soc_soft_weight", 0.0),
            )
            self.use_qp = False

        # Day window
        self.day_start = pd.Timestamp(CONFIG["time"]["day_start"])
        self.day_end   = pd.Timestamp(CONFIG["time"]["day_end"])

        # Read day-ahead once
        self.df15 = read_day_ahead_power_15min(self.day_ahead_path())
        self.df15 = dedup_and_clip_to_day(self.df15, self.day_start, self.day_end)

        # Output file (append per tick)
        self.out_csv = os.path.join(self.log_dir, "online_mpc_results.csv")
        if os.path.exists(self.out_csv):
            os.remove(self.out_csv)

    # ---------- Paths ----------
    def day_ahead_path(self) -> str:
        date_str = self.day_start.strftime("%Y%m%d")
        return f"data/inbox/day_ahead_{date_str}.csv"

    def ontime_forecast_path(self, online_index) -> Optional[str]:
        return ontime_csv("data/inbox/forecast/*.csv", online_index)

    def ontime_actual_path(self, online_index) -> Optional[str]:
        return ontime_csv("data/inbox/actual/*.csv", online_index)
    
    def latest_forecast_path(self) -> Optional[str]:
        return latest_csv("data/inbox/forecast/*.csv")

    def latest_actual_path(self) -> Optional[str]:
        return latest_csv("data/inbox/actual/*.csv")

    def _append_csv(self, row: Dict):
        """Append one row to online_mpc_results.csv (header on first write)."""
        df = pd.DataFrame([row])
        header = not os.path.exists(self.out_csv)
        df.to_csv(self.out_csv, mode="a", header=header, index=False)

    def tick(self, t_now: pd.Timestamp, online_index: int):
        """
        Execute one MPC action at time t_now (aligned to 5-min grid).
        Accepts minimal streaming files:
          - actual: 1 row at t_now
          - forecast: 3 rows for t_now, t_now+5m, t_now+10m
        """
    
        # Read latest forecast + actual (each a short file)
        # df5f_short = safe_read_csv(self.latest_forecast_path(), ("timestamp",))
        # df5a_short = safe_read_csv(self.latest_actual_path(), ("timestamp",))
        df5f_short = safe_read_csv(self.ontime_forecast_path(online_index), ("timestamp",))
        df5a_short = safe_read_csv(self.ontime_actual_path(online_index), ("timestamp",))
    
        # Fallbacks if missing
        if df5f_short is None or df5f_short.empty:
            # Build a synthetic 3-row forecast from day-ahead (forward-fill)
            # using t_now, t_now+5m, t_now+10m
            three_ts = pd.date_range(t_now, t_now + pd.Timedelta(minutes=10), freq=f"{self.dt_minutes}min")
            df15_ff = forward_fill_day_ahead_to_5min(self.df15, self.dt_minutes)
            df5f_short = df15_ff[df15_ff["timestamp"].isin(three_ts)].copy()
            df5f_short = df5f_short.rename(columns={"solar_forecast_kw": "solar_forecast_kw"})  # no change, just explicit
    
        # Ensure only next 3 substeps (t_now, +5, +10)
        three_ts = pd.date_range(t_now, t_now + pd.Timedelta(minutes=10), freq=f"{self.dt_minutes}min")
        df5f_short = df5f_short[df5f_short["timestamp"].isin(three_ts)].drop_duplicates("timestamp").sort_values("timestamp")
        # print("three_ts", three_ts)
        # print("df5f_short", df5f_short)
        
        # Actual: keep only t_now row (if present)
        if df5a_short is not None and not df5a_short.empty:
            df5a_short = df5a_short[df5a_short["timestamp"] == t_now].drop_duplicates("timestamp")
        else:
            df5a_short = None
    
        # Build a minimal tracking frame for the CURRENT block only
        # We need E_target_kwh for the block starting at floor_to_15min(t_now)
        block_start = floor_to_15min(t_now)
        block_end   = block_start + pd.Timedelta(minutes=15)
    
        # Get the day-ahead target power at block_start and convert to energy
        dfE = self.df15.set_index("timestamp").asfreq("15min", method="pad")
        if block_start not in dfE.index:
            raise RuntimeError(f"No day-ahead target at block_start={block_start}")
        E_target_kwh = float(dfE.loc[block_start, "expected_power_kw"]) * 0.25
    
        # Compose the 3-row block (timestamps and substeps)
        block_ts = pd.date_range(block_start, block_end - pd.Timedelta(minutes=5), freq=f"{self.dt_minutes}min")
        df_block = pd.DataFrame({"timestamp": block_ts})
        df_block["block_start"] = block_start
        df_block["block_end"]   = block_end
        df_block["substep_in_block"] = ((df_block["timestamp"] - block_start).dt.total_seconds() // (self.dt_minutes * 60)).astype(int)
        df_block["E_target_kwh"] = E_target_kwh
    
        # Merge forecast short into the block (forecast for t_now..t_now+10m)
        df_block = df_block.merge(df5f_short, on="timestamp", how="left")
        df_block["solar_forecast_kw"] = df_block["solar_forecast_kw"].fillna(0.0).clip(lower=0.0)
    
        # Merge actual (single row at t_now, if present)
        if df5a_short is not None:
            df_block = df_block.merge(df5a_short, on="timestamp", how="left")
            df_block["actual_available"] = df_block["solar_actual_kw"].notna()
        else:
            df_block["solar_actual_kw"] = pd.NA
            df_block["actual_available"] = False
    
        # Current index inside block (must match t_now)
        try:
            idx_cur = int(np.where(df_block["timestamp"].to_numpy() == t_now)[0][0])
        except Exception:
            raise RuntimeError(f"t_now={t_now} not within current block rows {block_ts.tolist()}")
    
        # Prepare arrays for controller
        timestamps = df_block["timestamp"].to_numpy()
        substeps   = df_block["substep_in_block"].astype(int).to_numpy()
        E_targets  = df_block["E_target_kwh"].astype(float).to_numpy()
        fc_kw      = df_block["solar_forecast_kw"].astype(float).to_numpy()
        act_kw     = pd.to_numeric(df_block["solar_actual_kw"], errors="coerce").fillna(np.nan).to_numpy()
        has_act    = df_block["actual_available"].astype(bool).to_numpy()
    
        br = {
            "timestamps": timestamps,
            "substeps": substeps,
            "E_target_kwh": E_targets,
            "solar_forecast_kw": fc_kw,
            "solar_actual_kw": act_kw,
            "actual_available": has_act,
            "current_index": idx_cur,
        }
        # print("br", br)
        remaining_steps_day = int((self.day_end - t_now).total_seconds() // (self.dt_minutes * 60)) + 1
        target_power_kw_block = float(E_targets[0] / 0.25)
        
        # Compute MPC setpoint (QP or heuristic)
        if self.use_qp:
            p_kw = self.ctrl.compute_current_setpoint(
                E0_kwh=self.state.energy_kwh,
                soc_min_kwh=self.batt.soc_min_kwh,
                soc_max_kwh=self.batt.soc_max_kwh,
                last_p_kw=self.state.last_p_kw,
                block_rows=br,
                remaining_steps_day=remaining_steps_day,
                target_power_kw_block=target_power_kw_block,
            )
        else:
            p_kw = self.ctrl.compute_current_setpoint(
                E_kwh=self.state.energy_kwh,
                soc_min_kwh=self.batt.soc_min_kwh,
                soc_max_kwh=self.batt.soc_max_kwh,
                last_p_kw=self.state.last_p_kw,
                block_rows=br,
                remaining_steps_day=remaining_steps_day,
            )
    
        # Use actual at t_now if available, else forecast at t_now
        solar_now_kw = float(act_kw[idx_cur]) if (has_act[idx_cur] and not np.isnan(act_kw[idx_cur])) else float(fc_kw[idx_cur])
        grid_out_kw = solar_now_kw + p_kw
    
        # Advance battery state by one 5-min step
        self.state.step(p_kw, self.dt_minutes)
    
        # Log one row
        row = {
            "timestamp": pd.Timestamp(t_now),
            "block_start": pd.Timestamp(block_start),
            "substep_in_block": int(substeps[idx_cur]),
            "E_target_kwh": float(E_targets[0]),
            "target_power_kw": target_power_kw_block,
            "solar_forecast_kw": float(fc_kw[idx_cur]),
            "solar_actual_kw": float(act_kw[idx_cur]) if not np.isnan(act_kw[idx_cur]) else np.nan,
            "actual_available": bool(has_act[idx_cur]),
            "battery_power_kw": float(p_kw),
            "grid_output_kw": float(grid_out_kw),
            "soc_kwh": float(self.state.energy_kwh),
        }
        self._append_csv(row)
        return row
