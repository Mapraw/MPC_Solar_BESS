
# src/simulation/simulator.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

from src.models.battery import BatteryParams, BatteryState
from src.mpc.mpc_controller import BlockEnergyMPC

def run_day_with_block_energy_mpc(
    config: Dict,
    df_tracking: pd.DataFrame,
) -> pd.DataFrame:
    batt = BatteryParams(**config["battery"])
    state = BatteryState(batt)


    use_qp = config.get("mpc", {}).get("use_qp", False)
    print(use_qp, "use_qp status")
    if use_qp:
        from src.mpc.qp_block_mpc import QPBlockEnergyMPC
        w = config.get("mpc", {}).get("qp_weights", {})
        ctrl = QPBlockEnergyMPC(
            dt_minutes=config["time"]["dt_minutes_rtu"],
            p_discharge_max_kw=batt.p_discharge_max_kw,
            p_charge_max_kw=batt.p_charge_max_kw,
            eta_charge=batt.eta_charge,
            eta_discharge=batt.eta_discharge,
            ramp_rate_kw_per_step=config["time"].get("ramp_rate_kw_per_step"),
            w_track=w.get("w_track", 1.0),
            w_mag=w.get("w_mag", 1e-4),
            w_smooth=w.get("w_smooth", 1e-2),
            w_block_energy=w.get("w_block_energy", 10.0),
            w_terminal_soc=w.get("w_terminal_soc", 0.5),
            soc_terminal_kwh=config["battery"].get("soc_terminal_kwh"),
        )
    else:
        from src.mpc.mpc_controller import BlockEnergyMPC
        ctrl = BlockEnergyMPC(
            dt_minutes=config["time"]["dt_minutes_rtu"],
            p_discharge_max_kw=batt.p_discharge_max_kw,
            p_charge_max_kw=batt.p_charge_max_kw,
            eta_charge=batt.eta_charge,
            eta_discharge=batt.eta_discharge,
            ramp_rate_kw_per_step=config["time"].get("ramp_rate_kw_per_step"),
            soc_terminal_kwh=config["battery"].get("soc_terminal_kwh"),
            terminal_weight=config.get("mpc", {}).get("terminal_soc_soft_weight", 0.0),
        )

    # src/simulation/simulator.py (PATCH)
    # ctrl = BlockEnergyMPC(
    #     dt_minutes=config["time"]["dt_minutes_rtu"],
    #     p_discharge_max_kw=batt.p_discharge_max_kw,
    #     p_charge_max_kw=batt.p_charge_max_kw,
    #     eta_charge=batt.eta_charge,
    #     eta_discharge=batt.eta_discharge,
    #     ramp_rate_kw_per_step=config["time"].get("ramp_rate_kw_per_step"),
    #     # NEW:
    #     soc_terminal_kwh=config["battery"].get("soc_terminal_kwh"),
    #     terminal_weight=config.get("mpc", {}).get("terminal_soc_soft_weight", 0.0),
    # )

    dt_minutes = config["time"]["dt_minutes_rtu"]

    results = []
    # Iterate through each 5-min timestamp
    for i in range(len(df_tracking)):
        row = df_tracking.iloc[i]
        cur_ts = row["timestamp"]
        cur_block_start = row["block_start"]
        cur_substep = int(row["substep_in_block"])

        # Extract the 3 rows of the current block (order by substep)
        block_rows = df_tracking[df_tracking["block_start"] == cur_block_start].sort_values("substep_in_block")

        # ---------- FIXED: robust array preparation ----------
        timestamps = block_rows["timestamp"].to_numpy()
        substeps   = block_rows["substep_in_block"].astype(int).to_numpy()
        E_targets  = block_rows["E_target_kwh"].astype(float).to_numpy()
        fc_kw      = block_rows["solar_forecast_kw"].astype(float).to_numpy()

        if "solar_actual_kw" in block_rows.columns:
            act_kw = (
                pd.to_numeric(block_rows["solar_actual_kw"], errors="coerce")
                .fillna(np.nan)
                .to_numpy()
            )
            has_act = block_rows["actual_available"].astype(bool).to_numpy()
        else:
            act_kw = np.full(len(block_rows), np.nan, dtype=float)
            has_act = np.zeros(len(block_rows), dtype=bool)

        idx_cur = int(np.where(block_rows["timestamp"].to_numpy() == cur_ts)[0][0])

        br = {
            "timestamps": timestamps,
            "substeps": substeps,
            "E_target_kwh": E_targets,
            "solar_forecast_kw": fc_kw,
            "solar_actual_kw": act_kw,
            "actual_available": has_act,
            "current_index": idx_cur,
        }
        # -----------------------------------------------

        # Compute the current setpoint for the BESS

        # Inside the 5-min loop:
        remaining_steps_day = len(df_tracking) - i
        p_kw = ctrl.compute_current_setpoint(
            E_kwh=state.energy_kwh,
            soc_min_kwh=batt.soc_min_kwh,
            soc_max_kwh=batt.soc_max_kwh,
            last_p_kw=state.last_p_kw,
            block_rows=br,
            remaining_steps_day=remaining_steps_day,  # NEW
        )

        # Forecast-based grid output for current step (for visualization)
        solar_fc_kw = float(row["solar_forecast_kw"])
        grid_out_kw = solar_fc_kw + p_kw

        # Advance battery state by one 5-min step
        state.step(p_kw, dt_minutes)

        # Convenience: 15-min expected power repeated across the block for plotting
        target_power_kw = float(row["E_target_kwh"] / 0.25)  # E_target / 0.25 h = kW

        results.append({
            "timestamp": pd.Timestamp(cur_ts),
            "block_start": pd.Timestamp(cur_block_start),
            "substep_in_block": cur_substep,
            "E_target_kwh": float(row["E_target_kwh"]),
            "target_power_kw": target_power_kw,
            "solar_forecast_kw": solar_fc_kw,
            "solar_actual_kw": float(row["solar_actual_kw"]) if pd.notna(row["solar_actual_kw"]) else np.nan,
            "actual_available": bool(row["actual_available"]),
            "battery_power_kw": float(p_kw),
            "grid_output_kw": float(grid_out_kw),
            "soc_kwh": float(state.energy_kwh),
        })

    return pd.DataFrame(results)
