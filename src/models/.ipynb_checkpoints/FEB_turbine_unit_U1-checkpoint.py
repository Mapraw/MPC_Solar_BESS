"""
turbine_unit_U1.py — Five-state turbine unit (fixed)
States:
1) START      : timer; zero output
2) RAMP_UP    : increase toward target; obey ramp; >= min_mw when running
3) NORMAL     : hold/adjust within [min_mw, max_mw]; ramp-limited; no drift at equality
4) RAMP_DOWN  : decrease toward 0; obey ramp
5) SHUTDOWN   : timer; zero output

Energy helpers:
- energy_over(total_minutes, step_minutes=1.0, mutate=False) -> MWh (trapezoidal)
- energy_in_minutes(minutes, mutate=False) -> MWh
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
import copy


class UnitState(Enum):
    START = auto()
    RAMP_UP = auto()
    NORMAL = auto()
    RAMP_DOWN = auto()
    SHUTDOWN = auto()


@dataclass
class TurbineUnitConfig:
    min_mw: float = 7.0
    max_mw: float = 12.0
    ramp_mw_per_min: float = 1.0
    startup_time_min: float = 3.0
    shutdown_time_min: float = 5.0
    startup_cost: float = 1000.0


class TurbineUnit:
    """
    Five-state hydro turbine unit with start/stop timers, ramp limits, and energy accounting.
    - Positive power only (0 .. max_mw).
    - Command via request_output(mw).  mw>0 => start/ramp-up as needed; mw=0 => ramp_down then shutdown.
    """
    def __init__(self, name: str, config: TurbineUnitConfig | None = None):
        self.name = name
        self.config = config or TurbineUnitConfig()

        # Start fully shut down
        self.state = UnitState.SHUTDOWN
        self.setpoint_mw = 0.0       # current actual MW
        self.target_mw = 0.0         # external request (>=0)
        self._timer_s = 0.0          # START/SHUTDOWN countdown (seconds)
        self.accum_startup_cost = 0.0
        self.energy_mwh = 0.0

        # Numeric tolerance for "hold" behavior when at target
        self._eps = 1e-9

    # -------------------- Command API --------------------
    def request_output(self, mw: float) -> None:
        """
        Set desired power (>=0 MW). Triggers start/ramp-up or ramp-down/shutdown as needed.
        """
        self.target_mw = max(0.0, float(mw))

        # If fully shut down and someone asks >0, enter START
        if self.state == UnitState.SHUTDOWN and self.target_mw > 0.0 and self._timer_s <= 0.0:
            self._enter_start()

        # If currently ramping up or normal and request goes to 0 -> begin ramp-down
        if self.target_mw == 0.0 and self.state in (UnitState.RAMP_UP, UnitState.NORMAL):
            self._enter_ramp_down()

        # If currently ramping down and request goes >0, switch back to ramp-up
        if self.target_mw > 0.0 and self.state == UnitState.RAMP_DOWN:
            self._enter_ramp_up()

    # -------------------- State helpers --------------------
    def _enter_start(self) -> None:
        self.state = UnitState.START
        self.setpoint_mw = 0.0
        self._timer_s = self.config.startup_time_min * 60.0

    def _enter_ramp_up(self) -> None:
        self.state = UnitState.RAMP_UP
        # ramping starts from current setpoint; if just finished START it's 0.0

    def _enter_normal(self) -> None:
        self.state = UnitState.NORMAL

    def _enter_ramp_down(self) -> None:
        self.state = UnitState.RAMP_DOWN

    def _enter_shutdown(self) -> None:
        self.state = UnitState.SHUTDOWN
        self.setpoint_mw = 0.0
        self._timer_s = self.config.shutdown_time_min * 60.0

    def _finish_start(self) -> None:
        # Charge startup cost once the unit leaves START (becomes able to produce)
        self.accum_startup_cost += self.config.startup_cost
        self._enter_ramp_up()

    def _finish_shutdown(self) -> None:
        # After shutdown timer expires, remain in SHUTDOWN with 0 MW
        self.setpoint_mw = 0.0
        # If there is a pending positive target, allow a fresh start on next step
        # (handled in step/command)

    # -------------------- Physics helpers --------------------
    def _ramp_limit(self, dt_min: float) -> float:
        return max(0.0, self.config.ramp_mw_per_min * float(dt_min))

    def _clip_operating_window(self, mw: float) -> float:
        """
        Enforce physical operating window when running:
        - If target > 0, deliver within [min_mw, max_mw].
        - If target == 0, caller should be in RAMP_DOWN/SHUTDOWN; returning 0 for safety.
        """
        if self.target_mw <= 0.0:
            return 0.0
        return min(max(mw, self.config.min_mw), self.config.max_mw)

    # -------------------- Simulation step --------------------
    def step(self, dt_min: float) -> float:
        """
        Advance the unit by dt_min minutes, updating state and setpoint.
        Returns: actual MW after the step.
        """
        dt_min = float(dt_min)
        dt_h = dt_min / 60.0
        ramp = self._ramp_limit(dt_min)

        # ---- State machine ----
        if self.state == UnitState.START:
            # Zero output during startup timer
            self.setpoint_mw = 0.0
            self._timer_s -= dt_min * 60.0
            if self._timer_s <= 0.0:
                self._finish_start()

        elif self.state == UnitState.RAMP_UP:
            if self.target_mw <= 0.0:
                self._enter_ramp_down()
            else:
                desired = self._clip_operating_window(self.target_mw)
                if desired > self.setpoint_mw + self._eps:
                    self.setpoint_mw = min(self.setpoint_mw + ramp, desired)
                elif desired < self.setpoint_mw - self._eps:
                    # Target decreased while ramping up; ramp down but not below min/desired
                    self.setpoint_mw = max(self.setpoint_mw - ramp, max(self.config.min_mw, desired))
                else:
                    # Within epsilon: settle at desired and enter NORMAL
                    self.setpoint_mw = desired
                    self._enter_normal()

        elif self.state == UnitState.NORMAL:
            if self.target_mw <= 0.0:
                self._enter_ramp_down()
            else:
                desired = self._clip_operating_window(self.target_mw)
                if desired > self.setpoint_mw + self._eps:
                    self.setpoint_mw = min(self.setpoint_mw + ramp, desired)
                elif desired < self.setpoint_mw - self._eps:
                    self.setpoint_mw = max(self.setpoint_mw - ramp, max(self.config.min_mw, desired))
                else:
                    # Hold exactly at desired (no drift)
                    self.setpoint_mw = desired

        elif self.state == UnitState.RAMP_DOWN:
            # Ramp toward zero
            if self.setpoint_mw > 0.0:
                self.setpoint_mw = max(0.0, self.setpoint_mw - ramp)
            if self.setpoint_mw <= 0.0 + self._eps:
                # Enter timed shutdown (zero output maintained during timer)
                self._enter_shutdown()

        elif self.state == UnitState.SHUTDOWN:
            # Zero output; count down shutdown timer if active
            self.setpoint_mw = 0.0
            if self._timer_s > 0.0:
                self._timer_s -= dt_min * 60.0
                if self._timer_s <= 0.0:
                    self._finish_shutdown()
            # If fully shut and target is positive, allow starting again
            if self._timer_s <= 0.0 and self.target_mw > 0.0:
                self._enter_start()

        # ---- Energy accounting ----
        self.energy_mwh += self.setpoint_mw * dt_h
        return self.setpoint_mw

    # -------------------- Utilities --------------------
    def is_running(self) -> bool:
        return self.state in (UnitState.RAMP_UP, UnitState.NORMAL, UnitState.RAMP_DOWN)

    def status(self) -> tuple[str, str, float]:
        return (self.name, self.state.name, self.setpoint_mw)

    # -------------------- Energy helpers --------------------
    def energy_over(self, total_minutes: float, step_minutes: float = 1.0, mutate: bool = False) -> float:
        """
        Integrate energy (MWh) over the next `total_minutes` using the trapezoidal rule.
        If mutate=False (default), simulate on a deep copy (safe look-ahead).
        """
        total_minutes = float(total_minutes)
        step_minutes = float(step_minutes)
        if total_minutes <= 0 or step_minutes <= 0:
            return 0.0

        sim = self if mutate else copy.deepcopy(self)
        energy = 0.0
        remaining = total_minutes

        # Power at the beginning of first sub-step
        p_prev = sim.setpoint_mw

        while remaining > 1e-12:
            dt = min(step_minutes, remaining)
            # Advance simulation by dt -> returns *end* power of this sub-step
            p_new = sim.step(dt)
            # Trapezoid area: average power over the interval
            energy += (p_prev + p_new) * 0.5 * (dt / 60.0)

            # Next sub-step starts at current end power
            p_prev = p_new
            remaining -= dt

        return energy

    def energy_in_minutes(self, minutes: float, mutate: bool = False) -> float:
        return self.energy_over(minutes, step_minutes=1.0, mutate=mutate)


def make_unit(config: TurbineUnitConfig | None = None) -> TurbineUnit:
    """Factory returning a unit instance named 'U1'."""
    return TurbineUnit(name="U1", config=config)