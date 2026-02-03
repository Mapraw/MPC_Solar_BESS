This code have 2 branches
1. master: Prototype of MPC
2. MPC_1sec: 1 second interval thinking

Test system: python online_test.py

In this test, the python will run one period simulation through for loop. We have already prepared data in data folder.
Then, the code run for loop until the end of the day (which is defined in CONFIG).

Noted: 
1. Environment (runner) : src/runtime/realtime_runner.py
2. Power Calculation: MPC_Solar_BESS/src/mpc/qp_block_mpc.py
3. Result after running the code: MPC_Solar_BESS/logs

OBjective function (in FEB_objective_function.ipynb)
Now, it's objective function for calculating in 15 minutes periods. This will used for 1 day ahead planning

Emulator (in models folder)
- Turbine : Should use minute interval in controlling the power plant
- BESS & Solar : are able to use second interval in controlling the power plant

MPC will have 3 models
1. 15 minutes-interval, 2 day ahead optimization: Used for 2 day ahead planning.
2. 5 minutes-interval, 1 day ahead optimization: Used for Decide how much hydro should react.
3. 1 minutes-interval, 15 minutes optimization: Used for decide how much BESS should react.