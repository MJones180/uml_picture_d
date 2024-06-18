# Commands Lookup

This is a listing of some of the commands that have been used for the various scripts.

## Fixed Grid Data

Simulate data along a fixed grid from -50 to 50 nm in 10 nm increments.
This data is used to generate all the `zernike` plots.

    python3 main_stnp.py sim_data fixed_50nm_range v84 600e-9 \
        --output-write-batch 50 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --cores 4
    python3 main.py preprocess_data_bare fixed_50nm_range fixed_50nm_range_processed

## Response Matrix

Simulate the data and create a response matrix at 40 nm, will have the name `fixed_40nm`.

    python3 main_stnp.py sim_data fixed_40nm v84 600e-9 \
        --output-write-batch 10 \
        --fixed-amount-per-zernike 2 24 40e-9 \
        --cores 4
    python3 main.py create_response_matrix --simulated-data-tag fixed_40nm

## Running the Response Matrix

Calling the response matrix `fixed_40nm` on the `fixed_50nm_range_processed` dataset:

    python3 main.py run_response_matrix \
            fixed_40nm fixed_50nm_range_processed --scatter-plot 5 5 --zernike-plots

## Training and Testing Data

Simulate data to use for model training, validation, and testing.

Random Zernike aberrations for each term ranging from -50 to 50 nm:

    # 25,000 row dataset
    python3 main_stnp.py sim_data random_50nm_med v84 600e-9 \
                --output-write-batch 500 \
                --rand-amount-per-zernike 2 24 " -50e-9" 50e-9 25000 \
                --cores 4
    python3 main.py preprocess_data_complete \
                random_50nm_med \
                train_random_50nm_gl val_random_50nm_gl test_random_50nm_gl \
                75 10 15 \
                --norm-outputs globally

    # 100,000 row dataset
    python3 main_stnp.py sim_data random_50nm_large v84 600e-9 \
                --output-write-batch 500 \
                --rand-amount-per-zernike 2 24 " -50e-9" 50e-9 100000\
                --cores 4
    python3 main.py preprocess_data_complete \
                random_50nm_large \
                train_ran50nm_gl_lg_diff val_ran50nm_gl_lg_diff test_ran50nm_gl_lg_diff \
                70 20 10 \
                --norm-outputs globally \
                --use-field-diff

Random Zernike aberrations for each term ranging from -10 to 10 nm:

    python3 main_stnp.py sim_data random_10nm_med v84 600e-9 \
                --output-write-batch 500 \
                --rand-amount-per-zernike 2 24 " -10e-9" 10e-9 25000 \
                --cores 4
    python3 main.py preprocess_data_bare random_10nm_med random_10nm_med_processed

Random Zernike aberration for only one term in each row ranging from -50 to 50 nm:

    python3 main_stnp.py sim_data random_50nm_single_med v84 600e-9 \
        --output-write-batch 10 \
        --rand-amount-per-zernike-single 2 24 " -50e-9" 50e-9 25000 \
        --cores 4

Preprocessed dataset consisting of two of the above raw datasets (`random_50nm_single_med` and `random_50nm_large`, 125,000 rows).
The base field is subtracted off for each row so that they consist of only the differences:

    python3 main.py preprocess_data_complete \
        random_50nm_single_med \
        train_com50nm_gl_diff val_com50nm_gl_diff test_com50nm_gl_diff \
        70 20 10 \
        --norm-outputs globally \
        --use-field-diff \
        --additional-raw-data-tags random_50nm_large

## Model Training

Based off of the `train_ran50nm_gl_lg_diff` dataset, will have tags `ran50nm_gl_lg_diff_`:

    python3 main_stnp.py batch_model_train \
                train_ran50nm_gl_lg_diff val_ran50nm_gl_lg_diff \
                ran50nm_gl_lg_diff_ 500 \
                --networks dfc1 dfc2 sc1 btbc1 btbc2 \
                --losses mse --optimizers adam \
                --lrs 1e-3 5e-4 1e-4 \
                --batch-sizes 64 128 \
                --overwrite-existing --only-best-epoch --early-stopping 10 \
                --max-threads 4

Based off of the `train_com50nm_gl_diff` dataset, will have tags `com50nm_gl_diff_`:

    python3 main_stnp.py batch_model_train \
        train_com50nm_gl_diff val_com50nm_gl_diff \
        com50nm_gl_diff_ 500 \
        --networks dfc1 dfc2 dfc3 btbc2 btbc3 \
        --losses mse --optimizers adam \
        --lrs 6e-4 1e-4 6e-5 \
        --batch-sizes 64 128 \
        --overwrite-existing --only-best-epoch --early-stopping 10 \
        --max-threads 4

Based off of the `train_com50nm_gl_diff` dataset, will have tags `com50nm_gl_diff_v2_`:

    python3 main_stnp.py batch_model_train \
        train_com50nm_gl_diff val_com50nm_gl_diff \
        com50nm_gl_diff_v2_ 500 \
        --networks dfc2v2 dfc3v2 btbc4 \
        --losses mse --optimizers adam nadam \
        --lrs 1e-4 6e-5 \
        --batch-sizes 64 \
        --overwrite-existing --only-best-epoch --early-stopping 10 \
        --max-threads 4

## Model Testing

Testing for models with `ran50nm_gl_lg_diff_` tags:

    python3 main.py batch_model_test \
        test_ran50nm_gl_lg_diff --scatter-plot 5 5 \
        --epoch-and-tag-range last ran50nm_gl_lg_diff_ 1 30
    python3 main.py run_response_matrix fixed_40nm test_ran50nm_gl_lg_diff \
        --scatter-plot 5 5 --inputs-need-denorm --inputs-are-diff
    python3 main.py rank_analysis_dir test_ran50nm_gl_lg_diff --first 5

    python3 main.py batch_model_test \
        fixed_50nm_range_processed  \
        --inputs-need-norm --inputs-need-diff \
        --scatter-plot 5 5 --zernike-plots \
        --epoch-and-tag-range last ran50nm_gl_lg_diff_ 1 30
    python3 main.py rank_analysis_dir fixed_50nm_range_processed \
        --ds-on-fixed-grid --r-min-filter 0.4 --filter ran50nm_gl_lg_diff --first 5

Testing for models with `com50nm_gl_diff_` tags:

    python3 main.py batch_model_test \
        test_com50nm_gl_diff --scatter-plot 5 5 \
        --epoch-and-tag-range last com50nm_gl_diff_ 1 30
    python3 main.py run_response_matrix fixed_40nm test_com50nm_gl_diff \
        --scatter-plot 5 5 --inputs-need-denorm --inputs-are-diff
    python3 main.py rank_analysis_dir test_com50nm_gl_diff --first 5

    python3 main.py batch_model_test \
        fixed_50nm_range_processed  \
        --inputs-need-norm --inputs-need-diff \
        --scatter-plot 5 5 --zernike-plots \
        --epoch-and-tag-range last com50nm_gl_diff_ 1 30
    python3 main.py rank_analysis_dir fixed_50nm_range_processed \
        --ds-on-fixed-grid --r-min-filter 0.4 --filter com50nm_gl_diff --first 5

Testing for models with `com50nm_gl_diff_v2_` tags:

    python3 main.py batch_model_test \
        test_com50nm_gl_diff --scatter-plot 5 5 \
        --epoch-and-tag-range last com50nm_gl_diff_v2_ 1 12
    python3 main.py run_response_matrix fixed_40nm test_com50nm_gl_diff \
        --scatter-plot 5 5 --inputs-need-denorm --inputs-are-diff
    python3 main.py rank_analysis_dir test_com50nm_gl_diff --first 5

    python3 main.py batch_model_test \
        fixed_50nm_range_processed  \
        --inputs-need-norm --inputs-need-diff \
        --scatter-plot 5 5 --zernike-plots \
        --epoch-and-tag-range last com50nm_gl_diff_v2_ 1 12
    python3 main.py rank_analysis_dir fixed_50nm_range_processed \
        --ds-on-fixed-grid --r-min-filter 0.4 --filter com50nm_gl_diff_v2 --first 5

## Random Commands

Print out the layers, trainable neurons, and time it takes to run the `dfc3` network:

    python3 main.py network_info dfc3 --benchmark 1000

Output information on the `train_com50nm_gl_diff` dataset:

    python3 main.py dataset_info \
        train_com50nm_gl_diff \
        --verify-network-compatability dfc3 \
        --plot-example-images --plot-outputs-hist

Plot the training vs validation loss for all epochs in `com50nm_gl_diff_v2_1`:

    python3 main.py plot_model_loss com50nm_gl_diff_v2_1

Prune the `tag_lookup.json` file from older models:

    python3 main.py prune_tag_lookup
