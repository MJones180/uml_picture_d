# Commands Lookup

## Data Simulation

A single row with no aberrations (used to compute the difference during preprocessing):

    python3 main_scnp.py sim_data no_aberrations v84 600e-9 --no-aberrations 2 24
    python3 main_scnp.py sim_data no_aberrations_approx v84_approx 600e-9 --no-aberrations 2 24

A single row with aberrations (10 nm) on every term:

    python3 main_scnp.py sim_data all_10nm v84 600e-9 \
        --fixed-amount-per-zernike-all 2 24 10e-9 \
        --save-full-intensity

A single row where different terms have different aberrations:

    # 1 row, 500 nm for Z2-3, 20 nm for Z4-8, 10 nm for Z9-24
    python3 main_scnp.py sim_data fixed_group_500_20_10 v84_approx 600e-9 \
        --fixed-amount-per-zernike-all 2 3 500e-9 4 8 20e-9 9 24 10e-9

The `--save-aberrations-csv` or `--save-aberrations-csv-quit` args can be passed and an additional CSV file will be written out containing the aberrations. This can be handy if just the aberration values are needed, then the script can terminate.

    python3 main.py sim_data just_aberrations_all_10nm v84 600e-9 \
        --fixed-amount-per-zernike-all 2 24 10e-9 \
        --save-aberrations-csv-quit

Data at a fixed RMS error (can be used to create a response matrix):

    # 10 nm
    python3 main_scnp.py sim_data fixed_10nm v84 600e-9 \
        --output-write-batch 10 \
        --fixed-amount-per-zernike 2 24 10e-9 \
        --append-no-aberrations-row \
        --save-full-intensity \
        --cores 4

    # 40 nm
    python3 main_scnp.py sim_data fixed_40nm v84 600e-9 \
        --output-write-batch 10 \
        --fixed-amount-per-zernike 2 24 40e-9 \
        --append-no-aberrations-row \
        --save-full-intensity \
        --cores 4

Data used to create the +/- 40 nm RMS error response matrix:

    python3 main_scnp.py sim_data fixed_pm_40nm v84_approx 600e-9 \
        --output-write-batch 10 \
        --fixed-amount-per-zernike-pm 2 24 40e-9 \
        --append-no-aberrations-row \
        --save-full-intensity \
        --cores 4

Fixed aberrations on a grid for one term at a time in each row across all terms:

    # -50 to 50 nm in 10 nm increments
    python3 main_scnp.py sim_data fixed_50nm_range v84 600e-9 \
        --output-write-batch 50 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --cores 4

    # -50 to 50 nm in 10 nm increments
    python3 main_scnp.py sim_data fixed_50nm_range_approx v84_approx 600e-9 \
        --output-write-batch 50 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --cores 4

    # -50 to 50 nm with 2000 points in between
    python3 main_scnp.py sim_data fixed_50nm_range_2000 v84 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --append-no-aberrations-row \
        --cores 4

    # -50 to 50 nm with 2000 points in between using the approximated VVC simulations
    python3 main_scnp.py sim_data fixed_50nm_range_2000_approx v84_approx 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --append-no-aberrations-row \
        --cores 4

    # -10 to 10 nm with 401 points in between
    python3 main_scnp.py sim_data fixed_10nm_range_401 v84 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -10e-9" 10e-9 401 \
        --append-no-aberrations-row \
        --cores 4

    # -1 to 1 nm with 301 points in between
    python3 main_scnp.py sim_data fixed_1nm_range_301 v84 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 301 \
        --append-no-aberrations-row \
        --cores 4

Random aberrations for every term in each row:

    # -10 to 10 nm, 25,000 rows
    python3 main_scnp.py sim_data random_10nm_med v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 25000 2 24 " -10e-9" 10e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -10 to 10 nm, 100,000 rows
    python3 main_scnp.py sim_data random_10nm_large v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -10e-9" 10e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -20 to 20 nm, 100,000 rows
    python3 main_scnp.py sim_data random_20nm_large v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -20e-9" 20e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -30 to 30 nm, 100,000 rows
    python3 main_scnp.py sim_data random_30nm_large v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -30e-9" 30e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -40 to 40 nm, 100,000 rows
    python3 main_scnp.py sim_data random_40nm_large v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -40e-9" 40e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -50 to 50 nm, 25,000 rows
    python3 main_scnp.py sim_data random_50nm_med v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 25000 2 24 " -50e-9" 50e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -50 to 50 nm, 100,000 rows
    python3 main_scnp.py sim_data random_50nm_large v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -50e-9" 50e-9 \
        --append-no-aberrations-row \
        --cores 4

Random aberrations for every term in each row generated using the approximated VVC simulations:

    # -1 to 1 nm, 100,000 rows
    python3 main_scnp.py sim_data random_1nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -1e-9" 1e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -2 to 2 nm, 100,000 rows
    python3 main_scnp.py sim_data random_2nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -2e-9" 2e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -10 to 10 nm, 25,000 rows
    python3 main_scnp.py sim_data random_10nm_med_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 25000 2 24 " -10e-9" 10e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -10 to 10 nm, 100,000 rows
    python3 main_scnp.py sim_data random_10nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -10e-9" 10e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -20 to 20 nm, 100,000 rows
    python3 main_scnp.py sim_data random_20nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -20e-9" 20e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -30 to 30 nm, 100,000 rows
    python3 main_scnp.py sim_data random_30nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -30e-9" 30e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -40 to 40 nm, 100,000 rows
    python3 main_scnp.py sim_data random_40nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -40e-9" 40e-9 \
        --append-no-aberrations-row \
        --cores 4

    # -50 to 50 nm, 100,000 rows
    python3 main_scnp.py sim_data random_50nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 100000 2 24 " -50e-9" 50e-9 \
        --append-no-aberrations-row \
        --cores 4

If 100k rows of all the previous datasets were already simulated, then we can instantly create ones with less rows:

    python3 main.py random_trim_raw_dataset random_2nm_large_approx random_2nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_2nm_large_approx random_2nm_med_approx 50000

    python3 main.py random_trim_raw_dataset random_10nm_large_approx random_10nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_10nm_large_approx random_10nm_med_approx 50000

    python3 main.py random_trim_raw_dataset random_20nm_large_approx random_20nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_20nm_large_approx random_20nm_med_approx 50000

    python3 main.py random_trim_raw_dataset random_30nm_large_approx random_30nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_30nm_large_approx random_30nm_med_approx 50000

    python3 main.py random_trim_raw_dataset random_40nm_large_approx random_40nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_40nm_large_approx random_40nm_med_approx 50000

    python3 main.py random_trim_raw_dataset random_50nm_large_approx random_50nm_small_approx 20000
    python3 main.py random_trim_raw_dataset random_50nm_large_approx random_50nm_med_approx 50000

Random aberrations where different groups have different ranges:

    # 100,000 rows, 500 nm for Z2-3, 20 nm for Z4-8, 10 nm for Z9-24
    python3 main_scnp.py sim_data random_group_500_20_10 v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --cores 4

    # 100,000 rows, 50 nm for Z2-3, 10 nm for Z4-8, 5 nm for Z9-24
    python3 main_scnp.py sim_data random_group_50_10_5 v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -50e-9" 50e-9 4 8 " -10e-9" 10e-9 9 24 " -5e-9" 5e-9 \
        --cores 4

    # 100,000 rows, 15 nm for Z2-3, 5 nm for Z4-8, 2 nm for Z9-24
    python3 main_scnp.py sim_data random_group_15_5_2 v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -5e-9" 5e-9 9 24 " -2e-9" 2e-9 \
        --cores 4

    # 100,000 rows, 15 nm for Z2-3, 1 nm for Z4-8, 0.5 nm for Z9-24
    python3 main_scnp.py sim_data random_group_15_1_half v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --cores 4

    # 100,000 rows, 10 nm for Z2-3, 2 nm for Z4-8, 1 nm for Z9-24
    python3 main_scnp.py sim_data random_group_10_2_1 v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --cores 4

    # 100,000 rows, 0.5 nm for Z2-3, 0.25 nm for Z4-8, 0.2 nm for Z9-24
    python3 main_scnp.py sim_data random_group_half_quarter_fifth v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike 100000 2 3 " -5e-10" 5e-10 4 8 " -2.5e-10" 2.5e-10 9 24 " -2e-10" 2e-10 \
        --cores 4

More random aberrations but with a random normal distribution (Gaussian):

    # 100,000 rows, sigma=1 nm, normal distribution
    python3 main_scnp.py sim_data random_1nm_large_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 24 0 1e-9 \
        --cores 7

    # 25,000 rows, sigma=5 nm, normal distribution
    python3 main_scnp.py sim_data random_5nm_med_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 25000 2 24 0 5e-9 \
        --cores 7

    # 100,000 rows, sigma=5 nm, normal distribution
    python3 main_scnp.py sim_data random_5nm_large_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 24 0 5e-9 \
        --cores 7

    # 100,000 rows, sigma=(250, 10, 5) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_250_10_5_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 250e-9 4 8 0 10e-9 9 24 0 5e-9 \
        --cores 7

    # 100,000 rows, sigma=(25, 5, 2) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_25_5_2_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 25e-9 4 8 0 5e-9 9 24 0 2e-9 \
        --cores 7

    # 100,000 rows, sigma=(7, 2, 1) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_7_2_1_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 7e-9 4 8 0 2e-9 9 24 0 1e-9 \
        --cores 7

    # 100,000 rows, sigma=(7, 0.5, 0.25) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_7_half_quarter_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 7e-9 4 8 0 5e-10 9 24 0 2.5e-10 \
        --cores 7

    # 100,000 rows, sigma=(5, 1, 0.5) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_5_1_half_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 5e-9 4 8 0 1e-9 9 24 0 5e-10 \
        --cores 7

    # 100,000 rows, sigma=(0.25, 0.125, 0.1) nm for Z(2-3, 4-8, 9-24)
    python3 main_scnp.py sim_data random_group_quarter_eigth_tenth_normal v84_approx 600e-9 \
        --output-write-batch 500 --append-no-aberrations-row \
        --rand-amount-per-zernike-normal 100000 2 3 0 2.5e-10 4 8 0 1.25e-10 9 24 0 0.1e-10 \
        --cores 7

Just the aberrations for groups with different ranges:

    python3 main_scnp.py sim_data random_group_500_20_10_just_aberrations v84_approx 600e-9 \
        --rand-amount-per-zernike 200 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

Random aberration for only one term in each row ranging from -50 to 50 nm:

    python3 main_scnp.py sim_data random_50nm_single_med v84 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike-single 2 24 " -50e-9" 50e-9 25000 \
        --append-no-aberrations-row \
        --cores 4

A base row with random aberrations between -50 and 50 nm for each term. Then, rows with Gaussian perturbations (1 nm std) about the base row.

    python3 main_scnp.py sim_data rows_with_gaussian_pert v84 600e-9 \
        --rand-amount-per-zernike-row-then-gaussian-pert 1e-9 2 24 " -10e-9" 10e-9 25 \
        --save-plots True True True True --cores 4

Random aberration ranging from -50 to 50 nm for only one term in each row and a row for each of the Zernike terms (can be used to create a response matrix):

    python3 main_scnp.py sim_data random_50nm_single_each_large v84 600e-9 \
        --output-write-batch 1000 \
        --rand-amount-per-zernike-single-each 2 24 " -50e-9" 50e-9 5000 \
        --append-no-aberrations-row \
        --cores 4

Just the Zernike wavefront without any propagation:

    # Fixed values at 40 nm
    python3 main_scnp.py sim_data fixed_40nm_zernike_wf no_prop 600e-9 \
        --output-write-batch 10 \
        --fixed-amount-per-zernike 2 24 40e-9 \
        --append-no-aberrations-row \
        --save-full-intensity \
        --cores 4 --use-only-aberration-map

    # 10 nm on all terms at once
    python3 main_scnp.py sim_data all_10nm_zernike_wf no_prop 600e-9 \
        --fixed-amount-per-zernike-all 2 24 10e-9 \
        --save-full-intensity --use-only-aberration-map

    # Fixed grid from -50 to 50 nm with 21 points in between
    python3 main_scnp.py sim_data fixed_50nm_range_zernike_wf no_prop 600e-9 \
        --output-write-batch 50 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --cores 4 --use-only-aberration-map

    # Fixed grid from -50 to 50 nm with 201 points in between
    python3 main_scnp.py sim_data fixed_50nm_range_201_zernike_wf no_prop 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 201 \
        --append-no-aberrations-row \
        --cores 4 --use-only-aberration-map

    # 25,000 rows between -10 and 10 nm
    python3 main_scnp.py sim_data random_10nm_med_zernike_wf no_prop 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 25000 2 24 " -10e-9" 10e-9 \
        --append-no-aberrations-row \
        --cores 4 --use-only-aberration-map

    # 25,000 rows between -50 and 50 nm
    python3 main_scnp.py sim_data random_50nm_med_zernike_wf no_prop 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 25000 2 24 " -50e-9" 50e-9 \
        --append-no-aberrations-row \
        --cores 4 --use-only-aberration-map

## Data Preprocessing

This data can be used to generate the `zernike` plots:

    python3 main.py preprocess_data_bare fixed_50nm_range fixed_50nm_range_processed
    python3 main.py preprocess_data_bare fixed_50nm_range_approx fixed_50nm_range_approx_processed

Can be used for model training/validation:

    python3 main.py preprocess_data_complete \
        random_50nm_med \
        train_random_50nm_gl val_random_50nm_gl test_random_50nm_gl \
        75 10 15 \
        --norm-outputs globally

    python3 main.py preprocess_data_complete \
        random_50nm_large \
        train_ran50nm_gl_lg_diff val_ran50nm_gl_lg_diff test_ran50nm_gl_lg_diff \
        70 20 10 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        random_50nm_single_med \
        train_ran50nm_single_diff val_ran50nm_single_diff test_ran50nm_single_diff \
        70 20 10 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        random_50nm_single_med \
        train_com50nm_gl_diff val_com50nm_gl_diff test_com50nm_gl_diff \
        70 20 10 \
        --norm-outputs globally \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_50nm_large

    python3 main.py preprocess_data_complete \
        random_50nm_single_each_large \
        train_ran50nm_se_diff val_ran50nm_se_diff empty \
        75 25 0 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        random_50nm_single_each_large \
        train_val_ran50nm_se_diff_all empty empty \
        100 0 0 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_val_fixed_50nm_diff empty empty \
        100 0 0 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    # Updated the code to add a row with no aberrations
    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_val_fixed_50nm_diff_v2 empty empty \
        100 0 0 \
        --norm-outputs globally \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_val_fixed_50nm_diff_1nm_supersampled empty empty \
        100 0 0 \
        --norm-outputs globally \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags fixed_1nm_range_301

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_val_fixed_50nm_diff_1nm_supersampled_ones_range empty empty \
        100 0 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags fixed_1nm_range_301

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_val_fixed_50nm_diff_ones_range empty empty \
        100 0 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_fixed_50nm_ones_range val_fixed_50nm_ones_range empty \
        80 20 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        fixed_10nm_range_401 \
        train_val_fixed_10nm_diff_ones_range empty empty \
        100 0 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations

    python3 main.py preprocess_data_complete \
        fixed_50nm_range_2000 \
        train_fixed_2000_and_random_large val_fixed_2000_and_random_large empty \
        75 25 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_50nm_large

    python3 main.py preprocess_data_complete \
        random_50nm_large \
        train_fixed_2000_and_random_large_v2 val_fixed_2000_and_random_large_v2 empty \
        75 25 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000

    python3 main.py preprocess_data_complete \
        random_50nm_large \
        train_fixed_2000_and_random_group_ranges val_fixed_2000_and_random_group_ranges empty \
        75 25 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_10nm_large random_20nm_large random_30nm_large random_40nm_large \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000

    python3 main.py preprocess_data_complete \
        random_50nm_large \
        train_fixed_2000_and_random_group_ranges_v2 val_fixed_2000_and_random_group_ranges_v2 empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_10nm_large random_20nm_large random_30nm_large random_40nm_large \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000

    python3 main.py preprocess_data_complete \
        random_50nm_large_approx \
        train_fixed_2000_and_random_group_ranges_approx val_fixed_2000_and_random_group_ranges_approx empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_10nm_large_approx random_20nm_large_approx random_30nm_large_approx random_40nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_50nm_large_approx \
        train_random_group_ranges_approx val_random_group_ranges_approx empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_10nm_large_approx random_20nm_large_approx random_30nm_large_approx random_40nm_large_approx

    python3 main.py preprocess_data_complete \
        random_50nm_large_approx \
        train_fixed_2000_and_random_group_ranges_approx_v2 val_fixed_2000_and_random_group_ranges_approx_v2 empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_2nm_large_approx random_10nm_large_approx random_20nm_large_approx random_30nm_large_approx random_40nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_50nm_large_approx \
        train_fixed_2000_and_random_group_ranges_approx_v3 val_fixed_2000_and_random_group_ranges_approx_v3 empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_1nm_large_approx random_2nm_large_approx random_10nm_large_approx random_20nm_large_approx random_30nm_large_approx random_40nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_50nm_small_approx \
        train_fixed_2000_and_random_group_ranges_approx_v2_small val_fixed_2000_and_random_group_ranges_approx_v2_small empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_2nm_small_approx random_10nm_small_approx random_20nm_small_approx random_30nm_small_approx random_40nm_small_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_50nm_med_approx \
        train_fixed_2000_and_random_group_ranges_approx_v2_med val_fixed_2000_and_random_group_ranges_approx_v2_med empty \
        85 15 0 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_2nm_med_approx random_10nm_med_approx random_20nm_med_approx random_30nm_med_approx random_40nm_med_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges \
        val_fixed_2000_and_random_weighted_group_ranges \
        test_fixed_2000_and_random_weighted_group_ranges \
        80 15 5 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_2nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_v2 \
        val_fixed_2000_and_random_weighted_group_ranges_v2 \
        test_fixed_2000_and_random_weighted_group_ranges_v2 \
        80 15 5 \
        --norm-outputs globally --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local \
        val_fixed_2000_and_random_weighted_group_ranges_local \
        test_fixed_2000_and_random_weighted_group_ranges_local \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local_v2 \
        val_fixed_2000_and_random_weighted_group_ranges_local_v2 \
        test_fixed_2000_and_random_weighted_group_ranges_local_v2 \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_group_15_1_half random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local_v3 \
        val_fixed_2000_and_random_weighted_group_ranges_local_v3 \
        test_fixed_2000_and_random_weighted_group_ranges_local_v3 \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_group_15_1_half random_1nm_large_approx random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local_v4 \
        val_fixed_2000_and_random_weighted_group_ranges_local_v4 \
        test_fixed_2000_and_random_weighted_group_ranges_local_v4 \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_group_15_1_half random_group_half_quarter_fifth random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local_v4_bdw \
        val_fixed_2000_and_random_weighted_group_ranges_local_v4_bdw \
        test_fixed_2000_and_random_weighted_group_ranges_local_v4_bdw \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations_approx \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_group_15_1_half random_group_half_quarter_fifth random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_250_10_5_normal \
        train_fixed_2000_and_random_weighted_group_ranges_local_v4_normal \
        val_fixed_2000_and_random_weighted_group_ranges_local_v4_normal \
        test_fixed_2000_and_random_weighted_group_ranges_local_v4_normal \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_25_5_2_normal random_group_7_2_1_normal random_group_5_1_half_normal random_group_7_half_quarter_normal random_group_quarter_eigth_tenth_normal random_1nm_large_normal random_5nm_large_normal \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

    python3 main.py preprocess_data_complete \
        random_group_500_20_10 \
        train_fixed_2000_and_random_weighted_group_ranges_local_v4_sum1 \
        val_fixed_2000_and_random_weighted_group_ranges_local_v4_sum1 \
        test_fixed_2000_and_random_weighted_group_ranges_local_v4_sum1 \
        80 15 5 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff no_aberrations \
        --additional-raw-data-tags random_group_50_10_5 random_group_15_5_2 random_group_10_2_1 random_group_15_1_half random_group_half_quarter_fifth random_2nm_large_approx random_10nm_large_approx \
        --additional-raw-data-tags-train-only fixed_50nm_range_2000_approx

Can be used for testing:

    # 1 nm
    python3 main.py preprocess_data_bare random_1nm_large_approx random_1nm_large_processed

    # 10 nm
    python3 main.py preprocess_data_bare random_10nm_med random_10nm_med_processed
    python3 main.py preprocess_data_bare random_10nm_med_approx random_10nm_med_approx_processed

    # 50 nm
    python3 main.py preprocess_data_bare random_50nm_med random_50nm_med_processed

    # sigma = 5 nm
    python3 main.py preprocess_data_bare random_5nm_med_normal random_5nm_med_normal_processed

    # One row with no aberrations
    python3 main.py preprocess_data_bare no_aberrations no_aberrations_processed

If the data does not finish simulating, then the tables will have unequal sizes.
When this happens, each datafile's rows can be trimmed down:

    python3 main.py hdf_file_ops --trim-rows-in-datafile-based-on-table ../data/raw/dataset_tag/0_data.h5 ccd_intensity zernike_coeffs

## Creating a Response Matrix

Response matrix at 40 nm:

    python3 main.py create_response_matrix --simulated-data-tag-single fixed_40nm

Averaged response matrix:

    python3 main.py create_response_matrix \
        --simulated-data-tag-average fixed_pm_40nm

    python3 main.py create_response_matrix \
        --simulated-data-tag-average random_50nm_single_each_large

    python3 main.py create_response_matrix \
        --simulated-data-tag-average fixed_50nm_range_2000

## Running a Response Matrix

`fixed_40nm` response matrix:

    python3 main.py run_response_matrix fixed_40nm \
        fixed_50nm_range_processed --scatter-plot 4 6 2 1e-7 15 --zernike-plots

    python3 main.py run_response_matrix fixed_40nm \
        random_10nm_med_processed --scatter-plot 4 6 2 1e-7 15

    python3 main.py run_response_matrix fixed_40nm \
        test_ran50nm_gl_lg_diff --scatter-plot 4 6 2 1e-7 15 \
        --inputs-need-denorm --inputs-are-diff

    python3 main.py run_response_matrix fixed_40nm \
        test_com50nm_gl_diff --scatter-plot 4 6 2 1e-7 15 \
        --inputs-need-denorm --inputs-are-diff

    python3 main.py run_response_matrix fixed_40nm \
        test_ran50nm_single_diff --scatter-plot 4 6 2 1e-7 15 \
        --inputs-need-denorm --inputs-are-diff

`fixed_pm_40nm` response matrix:

    python3 main.py run_response_matrix fixed_pm_40nm \
        fixed_50nm_range_approx_processed --scatter-plot 4 6 2 1e-7 15 --zernike-plots

`random_50nm_single_each_large` response matrix:

    python3 main.py run_response_matrix random_50nm_single_each_large \
        fixed_50nm_range_processed --scatter-plot 4 6 2 1e-7 15 --zernike-plots

    python3 main.py run_response_matrix random_50nm_single_each_large \
        random_10nm_med_processed --scatter-plot 4 6 2 1e-7 15

    python3 main.py run_response_matrix random_50nm_single_each_large \
       random_50nm_med_processed --scatter-plot 4 6 2 1e-7 15

`fixed_50nm_range_2000` response matrix:

    python3 main.py run_response_matrix fixed_50nm_range_2000 \
        fixed_50nm_range_processed --scatter-plot 4 6 2 1e-7 15 --zernike-plots

    python3 main.py run_response_matrix fixed_50nm_range_2000 \
        random_10nm_med_processed --scatter-plot 4 6 2 1e-7 15

    python3 main.py run_response_matrix fixed_50nm_range_2000 \
       random_50nm_med_processed --scatter-plot 4 6 2 1e-7 15

## Model Training and Testing

Commands for model training and testing can be found in the `model_training_version.txt` file.
This is in a separate file to reduce clutter.
The file has the format of `txt` to reduce lag when opening.

## PICTURE-D Instrument

Commands for the PICTURE-D instrument can be found in the `picture_d_llowfs_cnn.md` file.
This is in a separate file to reduce clutter.
