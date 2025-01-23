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

    # 10 nm
    python3 main.py gen_zernike_time_steps \
        all_terms_10 0.001 50 2 24 --all-zernikes-constant-value 10e-9

    # 50 nm
    python3 main.py gen_zernike_time_steps \
        all_terms_50 0.001 50 2 24 --all-zernikes-constant-value 50e-9

## Perform Control Loop Run

Control loop step files:

    # Old best neural network model
    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last
    python3 main.py control_loop_run \
        all_terms_10 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --neural-network inference_speedup_v1_2 last

    # New best neural network model, much better on small aberrations
    python3 main.py control_loop_run \
        all_terms_10 -0.6 -0.2 0 v84_approx 600e-9 --neural-network weighted_aberration_ranges_local_v4 last

    # Most simple response matrix
    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm
    python3 main.py control_loop_run \
        all_terms_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm

    # Most advanced response matrix
    python3 main.py control_loop_run \
        single_term_3_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
    python3 main.py control_loop_run \
        single_term_3_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
    python3 main.py control_loop_run \
        all_terms_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000
    python3 main.py control_loop_run \
        all_terms_50 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_50nm_range_2000

    # PICTURE-C response matrix
    python3 main.py control_loop_run \
        all_terms_10 -0.6 -0.2 0 v84_approx 600e-9 --response-matrix fixed_40nm_positive_and_negative

Many control loops on static wavefronts to see how well the model can capture:

    python3 main_scnp.py control_loop_dataset_capture \
        random_group_500_20_10_just_aberrations 20 1e-3 2e-10 \
        -0.6 -0.2 0.0 v84_approx 600e-9 --neural-network weighted_aberration_ranges_local_v4 last \
        --cores 7

    python3 main_scnp.py control_loop_dataset_capture \
        random_group_500_20_10_just_aberrations 20 1e-3 2e-10 \
        -0.6 -0.2 0.0 v84_approx 600e-9 --response-matrix fixed_40nm_positive_and_negative \
        --cores 7
