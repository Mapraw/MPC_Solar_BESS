"""
hydro_storage_unit_U1.py — Bi-directional hydro storage (pumped-hydro) unit
Policy A for minimums: try to meet min power but clip to SoE-feasible if energy is insufficient.

+MW = discharge to grid
-MW = charge from grid
"""

### Fix ramp down 12 --> 7 --> 0

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
    capacity_mwh: float = 10000
    initial_soe_mwh: float = 10000

    # Efficiencies (0..1). Round-trip ≈ eta_charge * eta_discharge.
    eta_charge: float = 1
    eta_discharge: float = 1

    # Numerical tolerance.
    eps: float = 1e-9


class HydroStorageUnit:
    """
    Bi-directional hydro storage unit with ramp limits and SoE-aware clipping.
    Implements Policy A for minimums: enforce mins if possible, otherwise clip down
    to what SoE can support in the current step.
    """

    def __init__(self, name: str, config: HydroStorageConfig | None = None):
        self.name = name
        self.config = config or HydroStorageConfig()

        self.state = UnitState.IDLE
        self.setpoint_mw = 0.0       # actual power after ramp/SoE
        self.target_mw = 0.0         # requested (+ discharge, - charge)

        # Storage state (MWh)
        self.soe_mwh = max(
            0.0,
            min(self.config.capacity_mwh, self.config.initial_soe_mwh)
        )

        # Accounting (MWh)
        self.energy_out_mwh = 0.0    # delivered to grid
        self.energy_in_mwh = 0.0     # taken from grid

        self._eps = self.config.eps

    # ------------------------- Command API -------------------------
    def request_power(self, mw: float) -> None:
        """
        Set desired power (+ discharge, - charge).
        NOTE: Small requests are allowed; they'll be processed by nameplate + min enforcement.
        If you prefer "tiny -> zero", add logic here to zero-out |mw| < min_*.
        """
        mw = float(mw)
        self.target_mw = max(-self.config.max_charge_mw,
                             min(self.config.max_discharge_mw, mw))

    # Back-compat alias (from your turbine interface)
    def request_output(self, mw: float) -> None:
        self.request_power(mw)

    # ------------------------- Helpers -------------------------
    def _ramp_limit(self, dt_min: float) -> float:
        return max(0.0, self.config.ramp_mw_per_min * float(dt_min))

    def _clip_to_nameplate(self, mw: float) -> float:
        # Envelope first
        mw = max(-self.config.max_charge_mw, min(self.config.max_discharge_mw, mw))
        # Keep true zero if tiny / idle
        if abs(mw) <= self._eps:
            return 0.0
        # Enforce mins by direction for non-zero targets
        if mw > 0.0:
            return max(self.config.min_discharge_mw, mw)
        else:
            return min(-self.config.min_charge_mw, mw)
            
    def _clip_to_soe(self, mw: float, dt_min: float) -> float:
        """
        POLICY A (final):
        Clip requested power only by SoE feasibility for this step.
        Do NOT enforce minimum levels here, so ramp can pass through values below min.
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
            # Discharge: energy to grid limited by SoE: e_out_max = soe * eta_d
            e_out_max = max(0.0, soe * eta_d)
            p_max_feasible = (e_out_max / dt_h) if dt_h > 0 else 0.0
            return max(0.0, min(mw, p_max_feasible))
        else:
            # Charge: grid energy limited by headroom: e_in_max = headroom / eta_c
            headroom = max(0.0, cap - soe)
            e_in_max_grid = (headroom / eta_c) if eta_c > 0 else 0.0
            p_abs_limit = (e_in_max_grid / dt_h) if dt_h > 0 else 0.0
            return min(0.0, max(mw, -p_abs_limit))

    # ------------------------- Simulation step -------------------------
    def step(self, dt_min: float) -> float:
        dt_min = float(dt_min)
        if dt_min <= 0.0:
            return self.setpoint_mw
    
        # 1) Nameplate + mins on the target (does NOT break ramp)
        desired = self._clip_to_nameplate(self.target_mw)
    
        # 2) Apply ramp limit toward desired
        ramp = self._ramp_limit(dt_min)
        if desired > self.setpoint_mw + self._eps:
            p_tmp = min(self.setpoint_mw + ramp, desired)
        elif desired < self.setpoint_mw - self._eps:
            p_tmp = max(self.setpoint_mw - ramp, desired)
        else:
            p_tmp = desired
    
        # 3) SoE feasibility (Policy A: clip to what energy allows this step)
        p = self._clip_to_soe(p_tmp, dt_min)
    
        # 4) Update SoE & accounting
        dt_h = dt_min / 60.0
        if p > 0.0:
            e_out = p * dt_h
            e_from_res = e_out / self.config.eta_discharge if self.config.eta_discharge > 0 else 0.0
            self.energy_out_mwh += e_out
            self.soe_mwh = max(0.0, self.soe_mwh - e_from_res)
        elif p < 0.0:
            e_in = (-p) * dt_h
            e_to_res = e_in * self.config.eta_charge
            self.energy_in_mwh += e_in
            self.soe_mwh = min(self.config.capacity_mwh, self.soe_mwh + e_to_res)
    
        self.setpoint_mw = p
    
        # Cosmetic state
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
        return (self.name, self.state.name, self.setpoint_mw, self.soe_mwh)

    # ------------------------- Energy helpers -------------------------
    def energy_over(self, total_minutes: float, step_minutes: float = 1.0, mutate: bool = False) -> float:
        """
        Integrate NET energy to grid (MWh) over the next total_minutes using trapezoidal rule.
        Positive = discharge to grid; Negative = charging from grid.
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