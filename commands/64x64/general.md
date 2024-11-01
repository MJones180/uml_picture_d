# Commands Lookup

## Data Simulation

A single row with no aberrations (used to compute the difference during preprocessing):

    python3 main_scnp.py sim_data no_aberrations v84 600e-9 --no-aberrations

A single row with aberrations on every term:

    python3 main_scnp.py sim_data all_10nm v84 600e-9 \
        --fixed-amount-per-zernike-all 2 24 10e-9 \
        --save-full-intensity

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

Fixed aberrations on a grid for one term at a time in each row across all terms:

    # -50 to 50 nm in 10 nm increments
    python3 main_scnp.py sim_data fixed_50nm_range v84 600e-9 \
        --output-write-batch 50 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --cores 4

    # -50 to 50 nm with 2000 points in between using the approximated VVC simulations
    python3 main_scnp.py sim_data fixed_50nm_range_2000_approx v84_approx 600e-9 \
        --output-write-batch 1000 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --append-no-aberrations-row \
        --cores 4

Random aberrations for every term in each row generated using the approximated VVC simulations:

    # -10 to 10 nm, 100,000 rows
    python3 main_scnp.py sim_data random_10nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 2 24 " -10e-9" 10e-9 100000 \
        --append-no-aberrations-row \
        --cores 4

    # -20 to 20 nm, 100,000 rows
    python3 main_scnp.py sim_data random_20nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 2 24 " -20e-9" 20e-9 100000 \
        --append-no-aberrations-row \
        --cores 4

    # -30 to 30 nm, 100,000 rows
    python3 main_scnp.py sim_data random_30nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 2 24 " -30e-9" 30e-9 100000 \
        --append-no-aberrations-row \
        --cores 4

    # -40 to 40 nm, 100,000 rows
    python3 main_scnp.py sim_data random_40nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 2 24 " -40e-9" 40e-9 100000 \
        --append-no-aberrations-row \
        --cores 4

    # -50 to 50 nm, 100,000 rows
    python3 main_scnp.py sim_data random_50nm_large_approx v84_approx 600e-9 \
        --output-write-batch 500 \
        --rand-amount-per-zernike 2 24 " -50e-9" 50e-9 100000\
        --append-no-aberrations-row \
        --cores 4

## Data Preprocessing

This data can be used to generate the `zernike` plots:

    python3 main.py preprocess_data_bare fixed_50nm_range fixed_50nm_range_processed

Can be used for model training/validation:

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

Can be used for testing:

    # One row with no aberrations
    python3 main.py preprocess_data_bare no_aberrations no_aberrations_processed

If the data does not finish simulating, then the tables will have unequal sizes.
When this happens, each datafile's rows can be trimmed down:

    python3 main.py hdf_file_ops --trim-rows-in-datafile-based-on-table ../data/raw_simulated/dataset_tag/0_data.h5 ccd_intensity zernike_coeffs

## Creating a Response Matrix

Averaged response matrix:

    python3 main.py create_response_matrix \
        --simulated-data-tag-average fixed_50nm_range_2000_approx

## Running a Response Matrix

`fixed_50nm_range_2000_approx` response matrix:

    python3 main.py run_response_matrix fixed_50nm_range_2000_approx \
        fixed_50nm_range_processed --scatter-plot 5 5 --zernike-plots

    python3 main.py run_response_matrix fixed_50nm_range_2000_approx \
        random_10nm_med_processed --scatter-plot 5 5

    python3 main.py run_response_matrix fixed_50nm_range_2000_approx \
       random_50nm_med_processed --scatter-plot 5 5

## Model Training and Testing

Commands for model training and testing can be found in the `model_training_version.txt` file.
This is in a separate file to reduce clutter.
The file has the format of `txt` to reduce lag when opening.
