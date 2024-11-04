# Control Loop

Generate a CSV with timesteps and Zernike coefficients to run a control loop on:

    # 1 ms steps for a total of 0.025 seconds, terms 2 through 24, constant amplitude on term 3
    python3 main.py gen_zernike_time_steps \
        single_term_3_10 0.001 25 2 24 --single-zernike-constant-value 3 1e-9

Do a control loop run:

    # Use a neural network
    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last

    # Use a response matrix
    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
