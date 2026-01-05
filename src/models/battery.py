
# src/models/battery.py
# Battery parameters and energy state with charge/discharge efficiencies

from dataclasses import dataclass

@dataclass
class BatteryParams:
    energy_capacity_kwh: float
    soc_init_kwh: float
    soc_min_kwh: float
    soc_max_kwh: float
    p_discharge_max_kw: float
    p_charge_max_kw: float
    eta_charge: float
    eta_discharge: float
    soc_terminal_kwh: float

class BatteryState:
    def __init__(self, params: BatteryParams):
        self.params = params
        self.energy_kwh = params.soc_init_kwh
        self.last_p_kw = 0.0

    def step(self, p_kw: float, dt_minutes: int):
        """
        Update battery energy state for one step given AC power at grid side.
        Positive p_kw = discharge to grid; Negative p_kw = charge from grid.
        SOC evolution uses efficiencies (AC-side power command).
        """
        dt_hours = dt_minutes / 60.0
        if p_kw >= 0:
            # Discharging: energy decreases by p / eta_d
            self.energy_kwh -= dt_hours * (p_kw / self.params.eta_discharge)
        else:
            # Charging: energy increases by -p * eta_c (since p<0)
            self.energy_kwh -= dt_hours * (p_kw * self.params.eta_charge)
        self.last_p_kw = p_kw

    def within_bounds(self) -> bool:
        return (self.params.soc_min_kwh - 1e-6) <= self.energy_kwh <= (self.params.soc_max_kwh + 1e-6)
