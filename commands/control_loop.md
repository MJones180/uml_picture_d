# Control Loop

## Generate CSV With Timesteps

1 ms steps for a total of 0.05 seconds, terms 2 through 24, constant amplitude on a single term:

    # Term 3, 10 nm
    python3 main.py gen_zernike_time_steps \
        single_term_3_10 0.001 50 2 24 --single-zernike-constant-value 3 10e-9

    # Term 3, 50 nm
    python3 main.py gen_zernike_time_steps \
        single_term_3_50 0.001 50 2 24 --single-zernike-constant-value 3 50e-9

1 ms steps for a total of 0.05 seconds, terms 2 through 24, constant amplitude on all terms:

    # 50 nm
    python3 main.py gen_zernike_time_steps \
        all_terms_50 0.001 50 2 24 --all-zernikes-constant-value 50e-9

## Perform Control Loop Run

Neural Network:

    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last

Response Matrix:

    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm

    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
