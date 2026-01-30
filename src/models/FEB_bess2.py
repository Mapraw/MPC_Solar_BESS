# plants/bess.py
from __future__ import annotations
from typing import Optional, Dict, Any
from datetime import datetime
from base import PowerPlant, PlantType

class BESS(PowerPlant):
    """
    BESS with commanded power (MW). + = discharge to grid; - = charge from grid.
    SOC tracked in MWh. Enforces limits & efficiencies. Optional ramp.
    """
    def __init__(
        self,
        name: str,
        capacity_mwh: float,
        soc_init_mwh: float,
        p_discharge_max_mw: float,
        p_charge_max_mw: float,
        eta_discharge: float = 0.95,
        eta_charge: float = 0.95,
        ramp_mw_per_min: Optional[float] = None,
        api=None
    ):
        super().__init__(name, PlantType.BESS, api)
        self.capacity_mwh = float(capacity_mwh)
        self.soc_mwh = float(soc_init_mwh)
        self.p_discharge_max_mw = float(p_discharge_max_mw)
        self.p_charge_max_mw = float(p_charge_max_mw)
        self.eta_discharge = float(eta_discharge)
        self.eta_charge = float(eta_charge)
        self.ramp_mw_per_s = (ramp_mw_per_min / 60.0) if ramp_mw_per_min else None
        self._cmd_power_mw: float = 0.0

    def command_power(self, power_mw: float):
        self._cmd_power_mw = float(power_mw)

    def step(self, now: datetime, dt_s: float) -> float:
        target = self._cmd_power_mw
        # Ramp:
        if self.ramp_mw_per_s is not None:
            max_delta = self.ramp_mw_per_s * dt_s
            target = max(min(target, self.current_power_mw + max_delta),
                         self.current_power_mw - max_delta)
        # Limits:
        if target >= 0:
            target = min(target, self.p_discharge_max_mw)
        else:
            target = max(target, -self.p_charge_max_mw)

        # SOC:
        dt_h = dt_s / 3600.0
        if target >= 0:
            energy_needed_mwh = (target / self.eta_discharge) * dt_h
            if energy_needed_mwh > self.soc_mwh:
                target = (self.soc_mwh / dt_h) * self.eta_discharge if dt_h > 0 else 0.0
                energy_needed_mwh = (target / self.eta_discharge) * dt_h
            self.soc_mwh -= energy_needed_mwh
        else:
            charge_power = -target
            energy_in_mwh = (charge_power * self.eta_charge) * dt_h
            if self.soc_mwh + energy_in_mwh > self.capacity_mwh:
                energy_room = self.capacity_mwh - self.soc_mwh
                charge_power = (energy_room / (self.eta_charge * dt_h)) if dt_h > 0 else 0.0
                target = -charge_power
                energy_in_mwh = (charge_power * self.eta_charge) * dt_h
            self.soc_mwh += energy_in_mwh

        self.current_power_mw = float(target)
        return self.current_power_mw

    def telemetry(self) -> Dict[str, Any]:
        return {
            "soc_mwh": round(self.soc_mwh, 6),
            "soc_percent": round(100.0 * self.soc_mwh / self.capacity_mwh, 3),
            "p_cmd_mw": self._cmd_power_mw,
        }