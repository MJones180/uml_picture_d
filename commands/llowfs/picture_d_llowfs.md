These are all the commands needed to obtain data on the PICTURE-D instrument and to create the associated RM and CNN.
Most of the commands are adapted from the ones meant for simulated data which are taken from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error (technically ±39.995 nm RMS error).
The CNN is based off the simulated [V49] `weighted_aberration_ranges_local_v4` model.
Based on instrument data obtained on April 1, 2025.

TABLE OF CONTENTS:
    SEC1 - INPUT ABERRATION CSV FILES
    SEC2 - CSV TO BINARY FILES
    SEC3 - MOVE FITS FILES
    SEC4 - FITS TO HDF FILES
    SEC5 - BASE FIELD NOTES
    SEC6 - PREPROCESS DATAFILES
    SEC7 - CNN TRAINING AND TESTING
    SEC8 - RM CREATION AND TESTING

SEC1 - INPUT ABERRATION CSV FILES ++++++++++++++++++++++++++++++++++++++++++++++
The CSV files containing the aberrations that will be run on the instrument.
Every datafile produced will contain 100k rows (may potentially change).
For Groups 9 and 10 which have 46k and 25k rows respectively, each datafile will
be padded with aberration free rows to make 100k rows.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- 1 ----
    # 100,000 rows, 500 nm for Z2-3, 20 nm for Z4-8, 10 nm for Z9-24
    # Based on simulated dataset: random_group_500_20_10
    python3 main_scnp.py sim_data picture_d_aberrations_group_1 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- 2 ----
    # 100,000 rows, 50 nm for Z2-3, 10 nm for Z4-8, 5 nm for Z9-24
    # Based on simulated dataset: random_group_50_10_5
    python3 main_scnp.py sim_data picture_d_aberrations_group_2 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -50e-9" 50e-9 4 8 " -10e-9" 10e-9 9 24 " -5e-9" 5e-9 \
        --save-aberrations-csv-quit

    # ---- 3 ----
    # 100,000 rows, 15 nm for Z2-3, 5 nm for Z4-8, 2 nm for Z9-24
    # Based on simulated dataset: random_group_15_5_2
    python3 main_scnp.py sim_data picture_d_aberrations_group_3 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -5e-9" 5e-9 9 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ---- 4 ----
    # 100,000 rows, 15 nm for Z2-3, 1 nm for Z4-8, 0.5 nm for Z9-24
    # Based on simulated dataset: random_group_15_1_half
    python3 main_scnp.py sim_data picture_d_aberrations_group_4 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit

    # ---- 5 ----
    # 100,000 rows, 10 nm for Z2-3, 2 nm for Z4-8, 1 nm for Z9-24
    # Based on simulated dataset: random_group_10_2_1
    python3 main_scnp.py sim_data picture_d_aberrations_group_5 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit

    # ---- 6 ----
    # 100,000 rows, 0.5 nm for Z2-3, 0.25 nm for Z4-8, 0.2 nm for Z9-24
    # Based on simulated dataset: random_group_half_quarter_fifth
    python3 main_scnp.py sim_data picture_d_aberrations_group_6 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -5e-10" 5e-10 4 8 " -2.5e-10" 2.5e-10 9 24 " -2e-10" 2e-10 \
        --save-aberrations-csv-quit

    # ---- 7 ----
    # 100,000 rows, 2 nm for Z2-24
    # Based on simulated dataset: random_2nm_large_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_7 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ---- 8 ----
    # 100,000 rows, 10 nm for Z2-24
    # Based on simulated dataset: random_10nm_large_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_8 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- 9 ----
    # 46,000 rows, -50 to 50 nm with 2000 points in between
    # Based on simulated dataset: fixed_50nm_range_2000_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_9 v84_approx 600e-9 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --save-aberrations-csv-quit

    # ---- 10 ----
    # 25,000 rows, 10 nm for Z2-Z24
    # Based on simulated dataset: random_10nm_med
    python3 main_scnp.py sim_data picture_d_aberrations_group_10 v84_approx 600e-9 \
        --rand-amount-per-zernike 25000 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

SEC2 - CSV TO BINARY FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++
Export the CSV files to binary so that they can be read by the flight code.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_1 \
        --simulated-data-tags picture_d_aberrations_group_1
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_2 \
        --simulated-data-tags picture_d_aberrations_group_2
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_3 \
        --simulated-data-tags picture_d_aberrations_group_3
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_4 \
        --simulated-data-tags picture_d_aberrations_group_4
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_5 \
        --simulated-data-tags picture_d_aberrations_group_5
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_6 \
        --simulated-data-tags picture_d_aberrations_group_6
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_7 \
        --simulated-data-tags picture_d_aberrations_group_7
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_8 \
        --simulated-data-tags picture_d_aberrations_group_8
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_9 \
        --simulated-data-tags picture_d_aberrations_group_9
    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_10 \
        --simulated-data-tags picture_d_aberrations_group_10

SEC3 - MOVE FITS FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The FITS datafiles that were obtained on the instrument should be moved to the
`data/raw` directory with new folder and file names.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # New path:
    #     lyt_100k_chunks/[IDX]_data.fits
    # IDX  Data Aberration Group           Filename
    # 0    picture_d_aberrations_group_1   lyt_alp_train_lac_20250401_152943_caldata.fits
    # 1    picture_d_aberrations_group_2   lyt_alp_train_lac_20250401_154408_caldata.fits
    # 2    picture_d_aberrations_group_3   lyt_alp_train_lac_20250401_155025_caldata.fits
    # 3    picture_d_aberrations_group_4   lyt_alp_train_lac_20250401_162221_caldata.fits
    # 4    picture_d_aberrations_group_5   lyt_alp_train_lac_20250401_161757_caldata.fits
    # 5    picture_d_aberrations_group_6   lyt_alp_train_lac_20250401_161330_caldata.fits
    # 6    picture_d_aberrations_group_7   lyt_alp_train_lac_20250401_160856_caldata.fits
    # 7    picture_d_aberrations_group_8   lyt_alp_train_lac_20250401_160418_caldata.fits

    # New path:
    #     lyt_single_zernikes/0_data.fits
    # Data Aberration Group           Filename
    # picture_d_aberrations_group_9   lyt_alp_train_lac_20250401_155940_caldata.fits

    # New path:
    #     lyt_10nm_testing/0_data.fits
    # Data Aberration Group           Filename
    # picture_d_aberrations_group_10  lyt_alp_train_lac_20250401_155512_caldata.fits

SEC4 - FITS TO HDF FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++
These FITS datafiles should be converted to HDF files. The format of the HDF
datafiles should be the same as the raw simulation datafiles.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py convert_picd_instrument_data picd_instrument_data_100k_groups 2 24 \
        --fits-data-tags lyt_100k_chunks
    python3 main.py convert_picd_instrument_data picd_instrument_data_25k_10nm 2 24 \
        --fits-data-tags lyt_10nm_testing --first-n-rows 25000
    python3 main.py convert_picd_instrument_data picd_instrument_data_no_aberrations 2 24 \
        --fits-data-tags lyt_10nm_testing --base-field-data 70000
    python3 main.py convert_picd_instrument_data picd_instrument_data_single_zernikes 2 24 \
        --fits-data-tags lyt_single_zernikes --first-n-rows 46000
    # Technically ±39.995 nm RMS error
    python3 main.py convert_picd_instrument_data picd_instrument_data_single_zernikes_pm40 2 24 \
        --fits-data-tags lyt_single_zernikes --first-n-rows 46000 --slice-row-ranges 4600 4623 41377 41400

SEC5 - BASE FIELD NOTES ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
All of the FITS datafiles contain a `PRIMARY` table which will either be empty
or contain 400 wavefronts. Originally, these 400 rows were averaged over to
create the base field. However, the base field produced from these 400 rows does
not seem to be the best, so these wavefronts must contain some aberration.
Therefore, the aberration free rows used to pad the dataset to 100k rows from
either Group 9 or 10 (see SEC1) should be used to create the base field.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

SEC6 - PREPROCESS DATAFILES ++++++++++++++++++++++++++++++++++++++++++++++++++++
The newly converted HDF datafiles should be preprocessed.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Preprocess the data for training, validation, and testing
    python3 main.py preprocess_data_complete \
        picd_instrument_data_100k_groups \
        train_picd_data val_picd_data test_picd_data \
        80 15 5 \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff picd_instrument_data_no_aberrations \
        --additional-raw-data-tags-train-only picd_instrument_data_single_zernikes

    # Preprocess the testing data
    python3 main.py preprocess_data_bare picd_instrument_data_single_zernikes \
        picd_instrument_data_single_zernikes_raw_processed
    python3 main.py preprocess_data_bare picd_instrument_data_25k_10nm \
        picd_instrument_data_25k_10nm_raw_processed

SEC7 - CNN TRAINING AND TESTING ++++++++++++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN model created from the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main_scnp.py model_train picd_cnn_round_one \
        train_picd_data val_picd_data \
        best32_1_smaller_10 mae adam 1e-5 1000 --batch-size 256 \
        --overwrite-existing --only-best-epoch --early-stopping 15
    python3 main_scnp.py model_train picd_cnn \
        train_picd_data val_picd_data \
        best32_1_smaller_10 mae adam 1e-8 200 --batch-size 128 \
        --overwrite-existing --only-best-epoch --early-stopping 15 \
        --init-weights picd_cnn_round_one last

    python3 main.py model_test picd_cnn last \
        test_picd_data --scatter-plot 4 6 2 0 15
    python3 main.py model_test picd_cnn last \
        picd_instrument_data_25k_10nm_raw_processed \
        --scatter-plot 4 6 2 1e-7 15 --inputs-need-norm --inputs-need-diff
    python3 main.py model_test picd_cnn last \
        picd_instrument_data_single_zernikes_raw_processed \
        --zernike-plots --inputs-need-norm --inputs-need-diff

    # Export the model so that it can be used in the `pytorch_model_in_c` repo
    # via ONNX runtime.
    python3 main.py export_model picd_cnn last val_picd_data --benchmark 5000

SEC8 - RM CREATION AND TESTING +++++++++++++++++++++++++++++++++++++++++++++++++
Create and test the RM model on the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py create_response_matrix \
        --simulated-data-tag-average picd_instrument_data_single_zernikes_pm40 \
        --base-field-tag picd_instrument_data_no_aberrations

    python3 main.py run_response_matrix picd_instrument_data_single_zernikes_pm40 \
        test_picd_data \
        --scatter-plot 4 6 2 0 15 --inputs-need-denorm --inputs-are-diff
    python3 main.py run_response_matrix picd_instrument_data_single_zernikes_pm40 \
        picd_instrument_data_25k_10nm_raw_processed \
        --scatter-plot 4 6 2 1e-8 15
    python3 main.py run_response_matrix picd_instrument_data_single_zernikes_pm40 \
        picd_instrument_data_single_zernikes_raw_processed --zernike-plots
