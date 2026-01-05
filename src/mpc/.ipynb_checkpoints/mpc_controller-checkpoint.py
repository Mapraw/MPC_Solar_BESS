from __future__ import annotations
from typing import Optional, Dict
import numpy as np

# src/mpc/mpc_controller.py (PATCH)
class BlockEnergyMPC:
    def __init__(
        self,
        dt_minutes: int,
        p_discharge_max_kw: float,
        p_charge_max_kw: float,
        eta_charge: float,
        eta_discharge: float,
        ramp_rate_kw_per_step: Optional[float] = None,
        # NEW: terminal SOC soft guidance
        soc_terminal_kwh: Optional[float] = None,
        terminal_weight: float = 0.0,
    ):
        self.dt_minutes = dt_minutes
        self.dt_hours = dt_minutes / 60.0
        self.p_dis_max = p_discharge_max_kw
        self.p_ch_max = p_charge_max_kw
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.ramp = ramp_rate_kw_per_step

        self.soc_terminal_kwh = soc_terminal_kwh
        self.terminal_weight = max(0.0, float(terminal_weight))

    # ... keep _apply_ramp, _apply_power_limits, _apply_soc_bounds unchanged ...
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
        E_kwh: float,
        soc_min_kwh: float,
        soc_max_kwh: float,
        last_p_kw: float,
        block_rows: Dict[str, np.ndarray],
        # NEW: how many 5-min steps remain in the day (including current)?
        remaining_steps_day: Optional[int] = None,
    ) -> float:
        dt_h = self.dt_hours
        idx_cur = int(block_rows["current_index"])
        substeps = block_rows["substeps"]

        past_mask = substeps < substeps[idx_cur]
        future_mask = ~past_mask  # includes current

        # Past energy from solar (actual if available else forecast)
        solar_past_kw = np.where(
            block_rows["actual_available"] & past_mask,
            np.nan_to_num(block_rows["solar_actual_kw"]),
            block_rows["solar_forecast_kw"]
        )
        E_solar_past_kwh = float(np.sum(solar_past_kw[past_mask]) * dt_h)

        # Future energy from solar (including current) -> forecast
        E_solar_future_kwh = float(np.sum(block_rows["solar_forecast_kw"][future_mask]) * dt_h)

        E_target_kwh = float(block_rows["E_target_kwh"][0])
        E_bess_needed_kwh = E_target_kwh - (E_solar_past_kwh + E_solar_future_kwh)

        remaining_steps_block = int(np.sum(future_mask))
        p_des_avg_kw = E_bess_needed_kwh / (remaining_steps_block * dt_h)

        # ---- Terminal SOC soft bias (gradually steer SOC toward target) ----
        if self.soc_terminal_kwh is not None and self.terminal_weight > 0.0 and remaining_steps_day:
            # Energy error in battery: positive if above desired SOC
            E_soc_err_kwh = E_kwh - self.soc_terminal_kwh
            # Spread correction over the remaining steps in the day (small bias)
            p_soc_bias_kw = self.terminal_weight * (E_soc_err_kwh / (remaining_steps_day * dt_h))
            p_des_avg_kw = p_des_avg_kw + p_soc_bias_kw
        # -------------------------------------------------------------------

        # Ramp, power limits, SOC bounds
        p_kw = self._apply_ramp(p_des_avg_kw, last_p_kw)
        p_kw = self._apply_power_limits(p_kw)
        p_kw = self._apply_soc_bounds(E_kwh, p_kw, soc_min_kwh, soc_max_kwh)
        return float(p_kw)
