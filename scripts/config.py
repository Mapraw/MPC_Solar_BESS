
# config.py
# Central configuration for the EMS + MPC

CONFIG = {
    "time": {
        "timezone": "Asia/Bangkok",
 # Simulation day (local timestamps in ISO-8601)
        "day_start": "2026-01-03T00:00:00",
        "day_end": "2026-01-03T23:55:00",

        # Resolutions
        "dt_minutes_rtu": 5,    # Real-time step for MPC and 5-min forecast
        "dt_minutes_day_ahead": 15,  # Target resolution

        # MPC horizon (in 5-min steps). 24 -> 2 hours look-ahead
        "mpc_horizon_steps": 24,

        # Optional ramp-rate constraint between consecutive steps (kW per 5-min step)
        "ramp_rate_kw_per_step": 2000
    },

    "battery": {
        "energy_capacity_kwh": 100000,  # 100 MWh
        "soc_init_kwh": 50000,          # 50% initial SOC
        "soc_min_kwh": 10000,           # 10% min SOC
        "soc_max_kwh": 90000,           # 90% max SOC
        "p_discharge_max_kw": 25000,    # 25 MW discharge limit
        "p_charge_max_kw": 25000,       # 25 MW charge limit (AC side)
        "eta_charge": 0.95,
        "eta_discharge": 0.95
    },

    "tracking": {
        "log_dir": "logs",
        "save_plots": True
    }
}
