
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
        "energy_capacity_kwh": 5000,  # 100 MWh
        "soc_init_kwh": 2500,          # 50% initial SOC
        "soc_min_kwh": 500,           # 10% min SOC
        "soc_max_kwh": 4500,           # 90% max SOC
        "p_discharge_max_kw": 2500,    # 25 MW discharge limit
        "p_charge_max_kw": 2500,       # 25 MW charge limit (AC side)
        "eta_charge": 0.95,
        "eta_discharge": 0.95
    },

    "tracking": {
        "log_dir": "logs",
        "save_plots": True
    },
    
    "hydro": {
        "enabled": True,
    
        # Maximum generation (power out)
        "p_out_max_kw": 18000,      #
        "eta_out": 1,            # generation efficiency
    
        # Maximum pumping (power in)
        "p_in_max_kw": 0,       # 40 MW
        "eta_in": 1,             # pumping efficiency
    
        # Optional ramp rate limit (same as battery style)
        "ramp_rate_kw_per_step": 15000,  # 3 MW per 1-min step
        "on_time_delay": 5, # delay time for ramping 2.5 minutes
        "off_time_delay": 5,
    },

}

### real : Hydro 36 MW; ramp 6000 kW per min, Solar 45 MW, BESS 10 MWH
## simulation = MW /2 /1000 kW