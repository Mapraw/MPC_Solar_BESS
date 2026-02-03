# plants/hydro.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any
from datetime import datetime
from base import PowerPlant, PlantType

class UnitState(Enum):
    OFF = auto()
    STARTING = auto()
    ON = auto()
    SHUTTING_DOWN = auto()

@dataclass
class HydroUnit:
    unit_id: int
    min_mw: float = 7.0
    max_mw: float = 12.0
    ramp_mw_per_min: float = 1.0
    startup_time_s: float = 180.0
    shutdown_time_s: float = 300.0
    state: UnitState = UnitState.OFF
    power_mw: float = 0.0
    timer_s: float = 0.0
    target_mw: float = 0.0

    def step(self, dt_s: float) -> None:
        ramp_mw_per_s = self.ramp_mw_per_min / 60.0
        max_delta = ramp_mw_per_s * dt_s

        if self.state == UnitState.OFF:
            self.power_mw = 0.0
            self.timer_s = 0.0

        elif self.state == UnitState.STARTING:
            self.timer_s += dt_s
            if self.timer_s >= self.startup_time_s:
                self.state = UnitState.ON
                # enter at min
                self.power_mw = max(self.min_mw, self.power_mw)

        elif self.state == UnitState.ON:
            desired = max(self.min_mw, min(self.max_mw, self.target_mw))
            if desired > self.power_mw:
                self.power_mw = min(desired, self.power_mw + max_delta)
            else:
                self.power_mw = max(desired, self.power_mw - max_delta)

        elif self.state == UnitState.SHUTTING_DOWN:
            self.timer_s += dt_s
            self.power_mw = max(0.0, self.power_mw - max_delta)
            if self.timer_s >= self.shutdown_time_s or self.power_mw <= 1e-6:
                self.state = UnitState.OFF
                self.power_mw = 0.0
                self.timer_s = 0.0

    def start(self):
        if self.state == UnitState.OFF:
            self.state = UnitState.STARTING
            self.timer_s = 0.0

    def shutdown(self):
        if self.state in (UnitState.ON, UnitState.STARTING):
            self.state = UnitState.SHUTTING_DOWN
            self.timer_s = 0.0

class HydroTurbineFleet(PowerPlant):
    """
    Fleet of hydraulic turbines (n units). Each unit 7–12 MW, 1 MW/min ramp,
    3 min startup, 5 min shutdown. Command is total MW (discharge only).
    Startup cost accumulates on OFF->STARTING.
    """
    def __init__(
        self,
        name: str,
        n_units: int = 3,
        min_mw: float = 7.0,
        max_mw: float = 12.0,
        ramp_mw_per_min: float = 1.0,
        startup_time_s: float = 180.0,
        shutdown_time_s: float = 300.0,
        startup_cost: float = 1000.0,
        api=None
    ):
        super().__init__(name, PlantType.HYDRO, api)
        self.units: List[HydroUnit] = [
            HydroUnit(
                unit_id=i + 1,
                min_mw=min_mw,
                max_mw=max_mw,
                ramp_mw_per_min=ramp_mw_per_min,
                startup_time_s=startup_time_s,
                shutdown_time_s=shutdown_time_s,
            ) for i in range(n_units)
        ]
        self.startup_cost = float(startup_cost)
        self.total_startup_cost = 0.0
        self._cmd_power_mw: float = 0.0

    def command_power(self, power_mw: float):
        self._cmd_power_mw = max(0.0, float(power_mw))

    # ---- internal helpers ----
    def _on_units(self) -> List[HydroUnit]:
        return [u for u in self.units if u.state == UnitState.ON]

    def _start_one_off_unit(self):
        for u in self.units:
            if u.state == UnitState.OFF:
                u.start()
                self.total_startup_cost += self.startup_cost
                return u
        return None

    def _schedule_shutdown_of_one(self):
        candidates = [u for u in self.units if u.state == UnitState.ON]
        if not candidates:
            candidates = [u for u in self.units if u.state == UnitState.STARTING]
        if candidates:
            u = sorted(candidates, key=lambda x: x.power_mw)[0]
            u.shutdown()
            return u
        return None

    def step(self, now: datetime, dt_s: float) -> float:
        # evolve
        for u in self.units:
            u.step(dt_s)

        request = self._cmd_power_mw
        on_units = self._on_units()
        total_min = sum(u.min_mw for u in on_units)
        total_max = sum(u.max_mw for u in on_units)

        # start units if needed
        while request > total_max and any(u.state == UnitState.OFF for u in self.units):
            self._start_one_off_unit()
            on_units = self._on_units()
            total_min = sum(u.min_mw for u in on_units)
            total_max = sum(u.max_mw for u in on_units)

        # shut down if request < total_min (and there are active units)
        if request < total_min and len(on_units) > 0:
            if request == 0.0 or len(on_units) > 1:
                self._schedule_shutdown_of_one()

        # recompute ON units and assign targets
        on_units = self._on_units()
        n_on = len(on_units)
        targets: Dict[int, float] = {}

        if n_on > 0:
            if request <= n_on * on_units[0].min_mw:
                for u in on_units:
                    targets[u.unit_id] = u.min_mw
            elif request >= n_on * on_units[0].max_mw:
                for u in on_units:
                    targets[u.unit_id] = u.max_mw
            else:
                base = request / n_on
                for u in on_units:
                    targets[u.unit_id] = max(u.min_mw, min(u.max_mw, base))

                # fine adjust to match total as close as possible
                def total_target() -> float:
                    return sum(targets[u.unit_id] for u in on_units)

                # raise if under
                while total_target() < request - 1e-6:
                    deficit = request - total_target()
                    progressed = False
                    for u in on_units:
                        room = u.max_mw - targets[u.unit_id]
                        if room > 0.0:
                            add = min(room, deficit / n_on)
                            targets[u.unit_id] += add
                            progressed = True
                    if not progressed:
                        break
                # lower if over
                while total_target() > request + 1e-6:
                    excess = total_target() - request
                    progressed = False
                    for u in on_units:
                        room = targets[u.unit_id] - u.min_mw
                        if room > 0.0:
                            sub = min(room, excess / n_on)
                            targets[u.unit_id] -= sub
                            progressed = True
                    if not progressed:
                        break

        # apply targets
        for u in self.units:
            if u.state == UnitState.ON:
                u.target_mw = targets.get(u.unit_id, u.min_mw)
            else:
                u.target_mw = 0.0

        self.current_power_mw = sum(u.power_mw for u in self.units)
        return self.current_power_mw

    def telemetry(self) -> Dict[str, Any]:
        return {
            "requested_mw": self._cmd_power_mw,
            "total_startup_cost": self.total_startup_cost,
            "units": [
                {
                    "unit_id": u.unit_id,
                    "state": u.state.name,
                    "power_mw": round(u.power_mw, 4),
                    "target_mw": round(u.target_mw, 4),
                    "timer_s": round(u.timer_s, 2),
                } for u in self.units
            ]
        }
``