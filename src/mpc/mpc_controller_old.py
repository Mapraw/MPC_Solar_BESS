
# src/mpc/mpc_controller.py
# Block-energy MPC controller for a 15-min target with rolling 5-min updates

from __future__ import annotations
from typing import Optional, Dict
import numpy as np

class BlockEnergyMPC:
    def __init__(
        self,
        dt_minutes: int,
        p_discharge_max_kw: float,
        p_charge_max_kw: float,
        eta_charge: float,
        eta_discharge: float,
        ramp_rate_kw_per_step: Optional[float] = None,
    ):
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.p_dis_max = p_discharge_max_kw
        self.p_ch_max = p_charge_max_kw
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.ramp = ramp_rate_kw_per_step

    def _apply_ramp(self, p_des: float, last_p: float) -> float:
        if self.ramp is None:
            return p_des
        return float(np.clip(p_des, last_p - self.ramp, last_p + self.ramp))

    def _apply_power_limits(self, p: float) -> float:
        return float(np.clip(p, -self.p_ch_max, self.p_dis_max))

    def _apply_soc_bounds(
        self,
        E_kwh: float,
        p_kw: float,
        soc_min_kwh: float,
        soc_max_kwh: float
    ) -> float:
        """
        Adjust p_kw minimally to keep next SOC within bounds:
          E_next = E_kwh - dt_h * (p/eta_d) if discharge
          E_next = E_kwh - dt_h * (p*eta_c) if charge (p<0)
        """
        dt_h = self.dt_hours
        if p_kw >= 0: ### discharge
            E_next = E_kwh - dt_h * (p_kw / self.eta_d)
            if E_next < soc_min_kwh:
                # max allowed discharge to stay at soc_min
                p_allowed_dis = max(0.0, (E_kwh - soc_min_kwh) * self.eta_d / dt_h)
                p_kw = min(p_kw, p_allowed_dis)
        else: ### charge
            E_next = E_kwh - dt_h * (p_kw * self.eta_c)
            if E_next > soc_max_kwh:
                # min allowed charging magnitude (negative p) to stay at soc_max
                p_allowed_ch = min(0.0, (E_kwh - soc_max_kwh) / (dt_h * self.eta_c))
                p_kw = max(p_kw, p_allowed_ch)

        # Re-clip to power limits
        return self._apply_power_limits(p_kw)

    def compute_current_setpoint(
        self,
        # Battery state & limits
        E_kwh: float,
        soc_min_kwh: float,
        soc_max_kwh: float,
        last_p_kw: float,
        # Current block data (3 rows for :00, :05, :10)
        block_rows: Dict[str, np.ndarray],
    ) -> float:
        """
        Compute the current BESS power p_kw for the present 5-min step
        to meet the block energy target by the end of the block.

        block_rows keys (numpy arrays aligned by block order):
          - timestamps: array of Timestamps (size 3)
          - substeps: array of {0,1,2}
          - E_target_kwh: scalar repeated (size 3, same value)
          - solar_forecast_kw: array of forecast powers
          - solar_actual_kw: array of actual powers (np.nan if not available)
          - actual_available: bool array, True where actual is present
          - current_index: integer in {0,1,2} indicating current substep
        """
        dt_h = self.dt_hours
        idx_cur = int(block_rows["current_index"])

        ### Do we need to define substeps, can we define by algorithm ?
        substeps = block_rows["substeps"]

        # Past substeps: strictly before current
        past_mask = substeps < substeps[idx_cur]
        future_mask = ~past_mask  # includes current

        # Energy already delivered by solar (use actual if available, else forecast) for elapsed substeps
        solar_past_kw = np.where(
            block_rows["actual_available"] & past_mask,
            np.nan_to_num(block_rows["solar_actual_kw"]),
            block_rows["solar_forecast_kw"]
        )
        E_solar_past_kwh = float(np.sum(solar_past_kw[past_mask]) * dt_h)

        # Energy expected from solar for remaining substeps (including the current)
        E_solar_future_kwh = float(np.sum(block_rows["solar_forecast_kw"][future_mask]) * dt_h)

        # Target energy for the block (same value across rows)
        E_target_kwh = float(block_rows["E_target_kwh"][0])

        # Battery energy required over the remaining substeps to meet the target
        E_bess_needed_kwh = E_target_kwh - (E_solar_past_kwh + E_solar_future_kwh)

        # Remaining number of steps (including current)
        remaining_steps = int(np.sum(future_mask))
        if remaining_steps <= 0:
            # Should not happen; fallback to zero
            return 0.0

        ### Shouldn't be distribute equally ?
        # Distribute evenly across remaining steps (simple heuristic)
        p_des_avg_kw = E_bess_needed_kwh / (remaining_steps * dt_h)

        # Apply ramp and power limits
        p_kw = self._apply_ramp(p_des_avg_kw, last_p_kw)
        p_kw = self._apply_power_limits(p_kw)

        # Enforce SOC bounds
        p_kw = self._apply_soc_bounds(E_kwh, p_kw, soc_min_kwh, soc_max_kwh)

        return float(p_kw)
