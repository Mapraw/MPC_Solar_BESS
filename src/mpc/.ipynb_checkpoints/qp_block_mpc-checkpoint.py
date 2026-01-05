
# src/mpc/qp_block_mpc.py
from __future__ import annotations
import cvxpy as cp
import numpy as np
from typing import Optional, Dict

class QPBlockEnergyMPC:
    def __init__(
        self,
        dt_minutes: int,
        p_discharge_max_kw: float,
        p_charge_max_kw: float,
        eta_charge: float,
        eta_discharge: float,
        ramp_rate_kw_per_step: Optional[float] = None,
        # weights
        w_track: float = 1.0,
        w_mag: float = 1e-4,
        w_smooth: float = 1e-2,
        w_block_energy: float = 10.0,
        w_terminal_soc: float = 0.1,
        soc_terminal_kwh: Optional[float] = None,
    ):
        self.dt_minutes = dt_minutes
        self.dt_h = dt_minutes / 60.0
        self.p_dis_max = p_discharge_max_kw
        self.p_ch_max = p_charge_max_kw
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.ramp = ramp_rate_kw_per_step

        self.w_track = w_track
        self.w_mag = w_mag
        self.w_smooth = w_smooth
        self.w_block_energy = w_block_energy
        self.w_terminal_soc = w_terminal_soc
        self.soc_terminal_kwh = soc_terminal_kwh

    def compute_current_setpoint(
        self,
        E0_kwh: float,
        soc_min_kwh: float,
        soc_max_kwh: float,
        last_p_kw: float,
        # current block info
        block_rows: Dict[str, np.ndarray],
        # global info
        remaining_steps_day: int,
        target_power_kw_block: float,  # E_target_kwh / 0.25
    ) -> float:
        """
        Solve a QP over the remaining steps in the current block.
        Decision variables: p_pos >= 0, p_neg >= 0, p = p_pos - p_neg.

        Objective includes tracking to target power, magnitude, smoothness,
        block energy penalty, optional terminal SOC penalty.
        """
        # Remaining steps (including current)
        # Use only future_mask indices, keep order
        substeps = block_rows["substeps"].astype(int)
        idx_cur = int(block_rows["current_index"])
        future_mask = substeps >= substeps[idx_cur]
        R = int(np.sum(future_mask))  # 1..3

        # Slice arrays for the remaining steps
        solar_fc = block_rows["solar_forecast_kw"][future_mask].astype(float)
        timestamps = block_rows["timestamps"][future_mask]
        # For tracking to power: target power is constant across block
        target_pw = np.full(R, float(target_power_kw_block), dtype=float)

        # Variables
        p_pos = cp.Variable(R, nonneg=True)
        p_neg = cp.Variable(R, nonneg=True)
        p = p_pos - p_neg

        # Power bounds per step (pos/neg)
        constraints = [
            p_pos <= self.p_dis_max,
            p_neg <= self.p_ch_max,
        ]

        # Ramp-rate constraints between consecutive p's (and vs last_p_kw at first step)
        if self.ramp is not None:
            constraints += [
                cp.abs(p[0] - last_p_kw) <= self.ramp
            ]
            for k in range(1, R):
                constraints += [cp.abs(p[k] - p[k-1]) <= self.ramp]

        # SOC dynamics across remaining steps (linear via split variables)
        # E_{k+1} = E_k - dt_h*(p_pos_k/eta_d) + dt_h*(p_neg_k*eta_c)
        E_seq = [E0_kwh]
        for k in range(R):
            E_next = E_seq[-1] - self.dt_h * (p_pos[k] / self.eta_d) + self.dt_h * (p_neg[k] * self.eta_c)
            E_seq.append(E_next)

        # SOC bounds at each step
        for k in range(1, len(E_seq)):  # skip E_seq[0] (initial)
            constraints += [
                E_seq[k] >= soc_min_kwh,
                E_seq[k] <= soc_max_kwh,
            ]

        # Objective
        obj = 0

        # 1) Power tracking: (solar_fc + p - target_pw)^2 over remaining steps
        obj += self.w_track * cp.sum_squares(solar_fc + p - target_pw)

        # 2) Magnitude penalty: small penalty to avoid excessive battery usage
        obj += self.w_mag * cp.sum_squares(p)

        # 3) Smoothness: penalty on changes between consecutive steps
        if R >= 2:
            obj += self.w_smooth * cp.sum_squares(p[1:] - p[:-1])

        # 4) Block energy penalty: make cumulative energy at block end match target E_target_kwh
        # E_block_err = E_target_kwh - (E_solar_past + E_solar_future + dt_h*sum(p over remaining))
        # We approximate with solar forecast (actuals are embedded in past part in simulator)
        dt_h = self.dt_h
        E_batt_future_kwh = dt_h * cp.sum(p)  # battery contribution to grid energy (AC side)
        # The simulator will pass the block energy error term as part of target calculation,
        # so we penalize deviation from the target power; this term refines energy matching:
        # We'll use a soft penalty with weight w_block_energy to favor end-of-block energy match.
        # Need E_target_kwh; we can reconstruct from target power * 0.25 h:
        E_target_kwh = float(target_power_kw_block) * 0.25
        # Solar forecast energy over remaining steps:
        E_solar_future_kwh = float(np.sum(solar_fc) * dt_h)
        # Past solar energy is accounted before calling QP (we can't change it here).
        # We add a penalty for (E_solar_future + E_batt_future - (E_target - E_solar_past)) ≈ 0,
        # but since E_solar_past is fixed, we implicitly push (E_solar_future + E_batt_future) toward E_target.
        obj += self.w_block_energy * cp.square(E_solar_future_kwh + E_batt_future_kwh - E_target_kwh)

        # 5) Terminal SOC penalty at the very last step of the day (optional)
        if (self.soc_terminal_kwh is not None) and (self.w_terminal_soc > 0.0) and (remaining_steps_day is not None):
            # Apply only if this block includes the final steps (simulator will still apply final correction if needed)
            E_last = E_seq[-1]
            obj += self.w_terminal_soc * cp.square(E_last - float(self.soc_terminal_kwh))

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if p.value is None:
            # Fallback to heuristic "do nothing"
            return float(0.0)

        # Apply only first control move
        return float(p.value[0])
