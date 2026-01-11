Test system: python online_test.py

In this test, the python will run one period simulation through for loop. We have already prepared data in data folder.
Then, the code run for loop until the end of the day (which is defined in CONFIG).

Noted: 
1. Environment (runner) : src/runtime/realtime_runner.py
2. Power Calculation: MPC_Solar_BESS/src/mpc/qp_block_mpc.py
3. Result after running the code: MPC_Solar_BESS/logs
