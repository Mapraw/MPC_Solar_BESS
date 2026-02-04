"""
hydro_storage_unit_U1.py — Bi-directional hydro storage (pumped-hydro) unit

Electricity-only abstraction (no thermal timers/costs):
- +MW  = discharge (generation to grid)
- -MW  = charge (pumping from grid)
- Ramp-limited power changes (MW/min) in either direction
- State of Energy (SoE, MWh) with charge/discharge efficiencies

Minimum power behavior (fixed):
- Minimums are enforced ONLY on the requested target (nameplate clip).
- During ramping and SoE clipping, minimums are NOT forced, so the unit can
  pass smoothly through values below the minimum on the way up/down.
- Example: with max=12, min=7, ramp=6 MW/min
  - Ramp-up from 0 to +12: 0 → +6 → +12 (no jump to 7 on the first minute)
  - Ramp-down to 1.0, mutate=False) -> net MWh to grid  - Ramp-down to 0: +12 → +6 → 0  (no hold at 7)
- energy_in_minutes(minutes, mutate=False) -> net MWh to grid
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import copy


class UnitState(Enum):
    IDLE = auto()
    CHARGING = auto()
    DISCHARGING = auto()
    RAMPING = auto()


@dataclass
class HydroStorageConfig:
    # Nameplate limits (MW). Use positive magnitudes.
    max_discharge_mw: float = 12.0
    max_charge_mw: float = 0

    # Minimum non-zero operating magnitudes (MW). Set to 0.0 to disable.
    min_discharge_mw: float = 7.0
    min_charge_mw: float = 0

    # Ramp (MW/min). Split into directional ramps later if needed.
    ramp_mw_per_min: float = 6.0

    # Storage capacity and initial state (MWh).
    capacity_mwh: float = 45.0
    initial_soe_mwh: float = 45.0

    # Efficiencies (0..1). Round-trip ≈ eta_charge * eta_discharge.
    eta_charge: float = 1
    eta_discharge: float = 1

    # Numerical tolerance.
    eps: float = 1e-9



class HydroStorageUnit:
    """
    Hydro storage unit (pumped-hydro) with:
      - ramp limit,
      - SoE-aware clipping (Policy A: SoE clip only),
      - min power enforced on targets only (not on ramp/SoE).

    Sign convention:
      +MW to grid (discharge), -MW from grid (charge).
    """

    def __init__(self, name: str, config: HydroStorageConfig | None = None):
        self.name = name
        self.config = config or HydroStorageConfig()

        # Operational state
        self.state = UnitState.IDLE
        self.setpoint_mw = 0.0
        self.target_mw = 0.0

        # Storage state (MWh)
        self.soe_mwh = max(0.0, min(self.config.capacity_mwh, self.config.initial_soe_mwh))

        # Accounting (MWh)
        self.energy_out_mwh = 0.0   # delivered to grid (+MW)
        self.energy_in_mwh = 0.0    # taken from grid (-MW)

        self._eps = self.config.eps

    # ------------------------- Command API -------------------------
    def request_power(self, mw: float) -> None:
        """
        Set desired power (+ discharge, - charge).
        Nameplate envelope is applied here; minimums are enforced for non-zero targets only.
        """
        mw = float(mw)
        # Clip to overall envelope first
        mw = max(-self.config.max_charge_mw, min(self.config.max_discharge_mw, mw))

        # If effectively zero, keep zero (do not raise to minimum)
        if abs(mw) <= self._eps:
            self.target_mw = 0.0
            return

        # Enforce minimum magnitude by direction for non-zero requests
        if mw > 0.0:
            mw = max(self.config.min_discharge_mw, mw)
        else:
            mw = min(-self.config.min_charge_mw, mw)  # negative

        self.target_mw = mw

    # Back-compat alias with the old turbine API
    def request_output(self, mw: float) -> None:
        self.request_power(mw)

    # ------------------------- Helpers -------------------------
    def _ramp_limit(self, dt_min: float) -> float:
        return max(0.0, self.config.ramp_mw_per_min * float(dt_min))

    def _clip_to_nameplate(self, mw: float) -> float:
        """
        Enforce overall envelope (±max) and minimums only on non-zero targets.
        If target is zero, returns 0 exactly (no min applied).
        """
        mw = max(-self.config.max_charge_mw, min(self.config.max_discharge_mw, mw))
        if abs(mw) <= self._eps:
            return 0.0
        if mw > 0.0:
            return max(self.config.min_discharge_mw, mw)
        else:
            return min(-self.config.min_charge_mw, mw)

    def _clip_to_soe(self, mw: float, dt_min: float) -> float:
        """
        POLICY A (final):
        Clip only by SoE feasibility for this step (no minimums here),
        so ramp can pass smoothly through values below the min on the way up/down.
        """
        if dt_min <= 0.0:
            return self.setpoint_mw
        if abs(mw) <= self._eps:
            return 0.0

        dt_h = dt_min / 60.0
        soe = self.soe_mwh
        cap = self.config.capacity_mwh
        eta_c = self.config.eta_charge
        eta_d = self.config.eta_discharge

        if mw > 0.0:
            # Discharge: energy to grid limited by available SoE
            e_out_max = max(0.0, soe * eta_d)
            p_max_feasible = (e_out_max / dt_h) if dt_h > 0 else 0.0
            return max(0.0, min(mw, p_max_feasible))
        else:
            # Charge: grid energy limited by headroom
            headroom = max(0.0, cap - soe)
            e_in_max_grid = (headroom / eta_c) if eta_c > 0 else 0.0
            p_abs_limit = (e_in_max_grid / dt_h) if dt_h > 0 else 0.0
            return min(0.0, max(mw, -p_abs_limit))

    # ------------------------- Simulation step -------------------------
    def step(self, dt_min: float) -> float:
        """
        Advance by dt_min minutes. Returns actual power (MW) after ramp + SoE clipping.
        """
        dt_min = float(dt_min)
        if dt_min <= 0.0:
            return self.setpoint_mw

        # 1) Desired = nameplate/min applied to the target only
        desired = self._clip_to_nameplate(self.target_mw)

        # 2) Ramp toward desired
        ramp = self._ramp_limit(dt_min)
        if desired > self.setpoint_mw + self._eps:
            p_tmp = min(self.setpoint_mw + ramp, desired)
        elif desired < self.setpoint_mw - self._eps:
            p_tmp = max(self.setpoint_mw - ramp, desired)
        else:
            p_tmp = desired

        # NOTE: No "snap to minimum" here. We keep the ramp value as-is.

        # 3) SoE feasibility (Policy A)
        p = self._clip_to_soe(p_tmp, dt_min)

        # 4) Update SoE & energy accounting
        dt_h = dt_min / 60.0
        if p > 0.0:
            # Delivered to grid; reservoir supplies e_out / eta_d
            e_out = p * dt_h
            e_from_res = e_out / self.config.eta_discharge if self.config.eta_discharge > 0 else 0.0
            self.energy_out_mwh += e_out
            self.soe_mwh = max(0.0, self.soe_mwh - e_from_res)
        elif p < 0.0:
            # Taken from grid; reservoir stores e_in * eta_c
            e_in = (-p) * dt_h
            e_to_res = e_in * self.config.eta_charge
            self.energy_in_mwh += e_in
            self.soe_mwh = min(self.config.capacity_mwh, self.soe_mwh + e_to_res)

        self.setpoint_mw = p

        # 5) Cosmetic state
        if abs(self.setpoint_mw) <= self._eps and abs(self.target_mw) <= self._eps:
            self.state = UnitState.IDLE
        elif abs(self.setpoint_mw - desired) > self._eps:
            self.state = UnitState.RAMPING
        else:
            self.state = UnitState.DISCHARGING if self.setpoint_mw > 0.0 else UnitState.CHARGING

        return self.setpoint_mw

    # ------------------------- Utilities -------------------------
    def is_running(self) -> bool:
        return abs(self.setpoint_mw) > self._eps or abs(self.target_mw) > self._eps

    def status(self) -> tuple[str, str, float, float]:
        """
        Returns (name, state, MW, SoE_MWh)
        """
        return (self.name, self.state.name, self.setpoint_mw, self.soe_mwh)

    # ------------------------- Energy helpers -------------------------
    def energy_over(self, total_minutes: float, step_minutes: float = 1.0, mutate: bool = False) -> float:
        """
        Integrate NET energy to grid (MWh) over the next total_minutes
        using the trapezoidal rule on MW (+ discharge, - charge).
        If mutate=False (default), simulate on a deep copy (safe look-ahead).
        """
        total_minutes = float(total_minutes)
        step_minutes = float(step_minutes)
        if total_minutes <= 0 or step_minutes <= 0:
            return 0.0

        sim = self if mutate else copy.deepcopy(self)
        net_energy = 0.0
        remaining = total_minutes

        p_prev = sim.setpoint_mw
        while remaining > 1e-12:
            dt = min(step_minutes, remaining)
            p_new = sim.step(dt)
            net_energy += (p_prev + p_new) * 0.5 * (dt / 60.0)
            p_prev = p_new
            remaining -= dt

        return net_energy

    def energy_in_minutes(self, minutes: float, mutate: bool = False) -> float:
        return self.energy_over(minutes, step_minutes=1.0, mutate=mutate)


def make_unit(config: HydroStorageConfig | None = None) -> HydroStorageUnit:
    """Factory returning a unit instance named 'U1'."""
    return HydroStorageUnit(name="U1", config=config)

Energy helpers:
