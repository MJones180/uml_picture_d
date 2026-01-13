....................................................................................................
NOTES AND CHANGES COMPARED TO `picture_d_llowfs_v3.md`:
- Turns out the CNN trained on `piccsim` data actually does not perform the best. Due to this, more
  data was obtained on the PICTURE-D instrument which was used to train the CNN in this file.
- Used the corresponding base field for every datafile to improve the differential wavefronts.
....................................................................................................

These are the commands needed to obtain data on the PICTURE-D instrument and to create the associated models.
Most of the commands are based on the ones used to simulate data in this repo; the commands are
taken/referenced from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error (technically ±39.995 nm RMS error).
The CNN is based off the [V55a] `sum1_scaling_faster_model` model.

TABLE OF CONTENTS:
    SEC1 - INPUT ABERRATION CSV FILES
    SEC2 - CSV TO BINARY FILES
    SEC3 - MOVE FITS FILES
    SEC4 - PREMERGE SOME FITS FILES
    SEC5 - FITS TO HDF FILES
    SEC6 - PREPROCESS DATAFILES
    SEC7 - [V55a] CNN TRAINING AND TESTING
    SEC8 - RM CREATION AND TESTING
    SEC9 - (EXTRA) [V55b] CNN TRAINING AND TESTING

SEC1 - INPUT ABERRATION CSV FILES ++++++++++++++++++++++++++++++++++++++++++++++
The CSV files containing the aberrations that will be run on the instrument.
Every datafile produced will contain 100k rows (may potentially change).
For Groups 8 and 9 which have 46k and 25k rows respectively.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- 0 ----
    # 100,000 rows, 500 nm for Z2-3, 20 nm for Z4-8, 10 nm for Z9-24
    # Based on simulated dataset: random_group_500_20_10
    python3 main_scnp.py sim_data picture_d_aberrations_group_0 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- 1 ----
    # 100,000 rows, 50 nm for Z2-3, 10 nm for Z4-8, 5 nm for Z9-24
    # Based on simulated dataset: random_group_50_10_5
    python3 main_scnp.py sim_data picture_d_aberrations_group_1 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -50e-9" 50e-9 4 8 " -10e-9" 10e-9 9 24 " -5e-9" 5e-9 \
        --save-aberrations-csv-quit

    # ---- 2 ----
    # 100,000 rows, 15 nm for Z2-3, 5 nm for Z4-8, 2 nm for Z9-24
    # Based on simulated dataset: random_group_15_5_2
    python3 main_scnp.py sim_data picture_d_aberrations_group_2 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -5e-9" 5e-9 9 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ---- 3 ----
    # 100,000 rows, 15 nm for Z2-3, 1 nm for Z4-8, 0.5 nm for Z9-24
    # Based on simulated dataset: random_group_15_1_half
    python3 main_scnp.py sim_data picture_d_aberrations_group_3 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit

    # ---- 4 ----
    # 100,000 rows, 10 nm for Z2-3, 2 nm for Z4-8, 1 nm for Z9-24
    # Based on simulated dataset: random_group_10_2_1
    python3 main_scnp.py sim_data picture_d_aberrations_group_4 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit

    # ---- 5 ----
    # 100,000 rows, 0.5 nm for Z2-3, 0.25 nm for Z4-8, 0.2 nm for Z9-24
    # Based on simulated dataset: random_group_half_quarter_fifth
    python3 main_scnp.py sim_data picture_d_aberrations_group_5 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 3 " -5e-10" 5e-10 4 8 " -2.5e-10" 2.5e-10 9 24 " -2e-10" 2e-10 \
        --save-aberrations-csv-quit

    # ---- 6 ----
    # 100,000 rows, 2 nm for Z2-24
    # Based on simulated dataset: random_2nm_large_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_6 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ---- 7 ----
    # 100,000 rows, 10 nm for Z2-24
    # Based on simulated dataset: random_10nm_large_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_7 v84_approx 600e-9 \
        --rand-amount-per-zernike 100000 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- 8 ----
    # 46,000 rows, -50 to 50 nm with 2000 points in between
    # Based on simulated dataset: fixed_50nm_range_2000_approx
    python3 main_scnp.py sim_data picture_d_aberrations_group_8 v84_approx 600e-9 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --save-aberrations-csv-quit

    # ---- 9 ----
    # 25,000 rows, 10 nm for Z2-Z24
    # Based on simulated dataset: random_10nm_med
    python3 main_scnp.py sim_data picture_d_aberrations_group_9 v84_approx 600e-9 \
        --rand-amount-per-zernike 25000 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

SEC2 - CSV TO BINARY FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++
Export the CSV files to binary so that they can be read by the flight code.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py export_zernike_inputs_to_binary picture_d_aberrations_group_0 \
        --simulated-data-tags picture_d_aberrations_group_0
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

SEC3 - MOVE FITS FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The FITS datafiles that were obtained on the instrument should be moved to the
`data/raw` directory with new folder and file names.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    mkdir lyt_100k_chunks_v4
    # picture_d_aberrations_group_0, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_144138_caldata.fits lyt_100k_chunks_v4/0_data.fits
    mv lyt_alp_train_lac_20250731_144547_caldata.fits lyt_100k_chunks_v4/1_data.fits
    mv lyt_alp_train_lac_20250731_145113_caldata.fits lyt_100k_chunks_v4/2_data.fits
    mv lyt_alp_train_lac_20250731_145508_caldata.fits lyt_100k_chunks_v4/3_data.fits
    # picture_d_aberrations_group_1, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_145937_caldata.fits lyt_100k_chunks_v4/4_data.fits
    mv lyt_alp_train_lac_20250731_150411_caldata.fits lyt_100k_chunks_v4/5_data.fits
    mv lyt_alp_train_lac_20250731_150820_caldata.fits lyt_100k_chunks_v4/6_data.fits
    mv lyt_alp_train_lac_20250731_151623_caldata.fits lyt_100k_chunks_v4/7_data.fits
    # picture_d_aberrations_group_2, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_152146_caldata.fits lyt_100k_chunks_v4/8_data.fits
    mv lyt_alp_train_lac_20250731_153135_caldata.fits lyt_100k_chunks_v4/9_data.fits
    mv lyt_alp_train_lac_20250731_154039_caldata.fits lyt_100k_chunks_v4/10_data.fits
    mv lyt_alp_train_lac_20250731_154441_caldata.fits lyt_100k_chunks_v4/11_data.fits
    # picture_d_aberrations_group_3, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_154842_caldata.fits lyt_100k_chunks_v4/12_data.fits
    mv lyt_alp_train_lac_20250731_155730_caldata.fits lyt_100k_chunks_v4/13_data.fits
    mv lyt_alp_train_lac_20250731_160138_caldata.fits lyt_100k_chunks_v4/14_data.fits
    mv lyt_alp_train_lac_20250731_160538_caldata.fits lyt_100k_chunks_v4/15_data.fits
    # picture_d_aberrations_group_4, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_161411_caldata.fits lyt_100k_chunks_v4/16_data.fits
    mv lyt_alp_train_lac_20250731_161813_caldata.fits lyt_100k_chunks_v4/17_data.fits
    mv lyt_alp_train_lac_20250731_162310_caldata.fits lyt_100k_chunks_v4/18_data.fits
    mv lyt_alp_train_lac_20250731_162727_caldata.fits lyt_100k_chunks_v4/19_data.fits
    # picture_d_aberrations_group_5, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_163123_caldata.fits lyt_100k_chunks_v4/20_data.fits
    mv lyt_alp_train_lac_20250731_163732_caldata.fits lyt_100k_chunks_v4/21_data.fits
    mv lyt_alp_train_lac_20250731_165014_caldata.fits lyt_100k_chunks_v4/22_data.fits
    mv lyt_alp_train_lac_20250731_165443_caldata.fits lyt_100k_chunks_v4/23_data.fits
    # picture_d_aberrations_group_6, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_170019_caldata.fits lyt_100k_chunks_v4/24_data.fits
    mv lyt_alp_train_lac_20250731_170457_caldata.fits lyt_100k_chunks_v4/25_data.fits
    mv lyt_alp_train_lac_20250731_170930_caldata.fits lyt_100k_chunks_v4/26_data.fits
    mv lyt_alp_train_lac_20250731_171411_caldata.fits lyt_100k_chunks_v4/27_data.fits
    # picture_d_aberrations_group_7, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_171923_caldata.fits lyt_100k_chunks_v4/28_data.fits
    mv lyt_alp_train_lac_20250731_172729_caldata.fits lyt_100k_chunks_v4/29_data.fits
    mv lyt_alp_train_lac_20250731_173218_caldata.fits lyt_100k_chunks_v4/30_data.fits
    mv lyt_alp_train_lac_20250731_173617_caldata.fits lyt_100k_chunks_v4/31_data.fits

    mkdir lyt_single_zernikes_v4
    # picture_d_aberrations_group_8, 2 chunks of 25k rows and 21k rows
    mv lyt_alp_train_lac_20250731_174007_caldata.fits lyt_single_zernikes_v4/0_data.fits
    mv lyt_alp_train_lac_20250731_174613_caldata.fits lyt_single_zernikes_v4/1_data.fits

    mkdir lyt_10nm_testing_v4
    # picture_d_aberrations_group_9, 1 chunk of 25k rows
    mv lyt_alp_train_lac_20250731_175044_caldata.fits lyt_10nm_testing_v4/0_data.fits

    # The base field (no aberrations) for each chunk
    mkdir lyt_no_aberrations_v4
    # picture_d_aberrations_group_0, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_144138_caldata_extra.fits lyt_no_aberrations_v4/0_data.fits
    mv lyt_alp_train_lac_20250731_144547_caldata_extra.fits lyt_no_aberrations_v4/1_data.fits
    mv lyt_alp_train_lac_20250731_145113_caldata_extra.fits lyt_no_aberrations_v4/2_data.fits
    mv lyt_alp_train_lac_20250731_145508_caldata_extra.fits lyt_no_aberrations_v4/3_data.fits
    # picture_d_aberrations_group_1, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_145937_caldata_extra.fits lyt_no_aberrations_v4/4_data.fits
    mv lyt_alp_train_lac_20250731_150411_caldata_extra.fits lyt_no_aberrations_v4/5_data.fits
    mv lyt_alp_train_lac_20250731_150820_caldata_extra.fits lyt_no_aberrations_v4/6_data.fits
    mv lyt_alp_train_lac_20250731_151623_caldata_extra.fits lyt_no_aberrations_v4/7_data.fits
    # picture_d_aberrations_group_2, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_152146_caldata_extra.fits lyt_no_aberrations_v4/8_data.fits
    mv lyt_alp_train_lac_20250731_153135_caldata_extra.fits lyt_no_aberrations_v4/9_data.fits
    mv lyt_alp_train_lac_20250731_154039_caldata_extra.fits lyt_no_aberrations_v4/10_data.fits
    mv lyt_alp_train_lac_20250731_154441_caldata_extra.fits lyt_no_aberrations_v4/11_data.fits
    # picture_d_aberrations_group_3, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_154842_caldata_extra.fits lyt_no_aberrations_v4/12_data.fits
    mv lyt_alp_train_lac_20250731_155730_caldata_extra.fits lyt_no_aberrations_v4/13_data.fits
    mv lyt_alp_train_lac_20250731_160138_caldata_extra.fits lyt_no_aberrations_v4/14_data.fits
    mv lyt_alp_train_lac_20250731_160538_caldata_extra.fits lyt_no_aberrations_v4/15_data.fits
    # picture_d_aberrations_group_4, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_161411_caldata_extra.fits lyt_no_aberrations_v4/16_data.fits
    mv lyt_alp_train_lac_20250731_161813_caldata_extra.fits lyt_no_aberrations_v4/17_data.fits
    mv lyt_alp_train_lac_20250731_162310_caldata_extra.fits lyt_no_aberrations_v4/18_data.fits
    mv lyt_alp_train_lac_20250731_162727_caldata_extra.fits lyt_no_aberrations_v4/19_data.fits
    # picture_d_aberrations_group_5, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_163123_caldata_extra.fits lyt_no_aberrations_v4/20_data.fits
    mv lyt_alp_train_lac_20250731_163732_caldata_extra.fits lyt_no_aberrations_v4/21_data.fits
    mv lyt_alp_train_lac_20250731_165014_caldata_extra.fits lyt_no_aberrations_v4/22_data.fits
    mv lyt_alp_train_lac_20250731_165443_caldata_extra.fits lyt_no_aberrations_v4/23_data.fits
    # picture_d_aberrations_group_6, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_170019_caldata_extra.fits lyt_no_aberrations_v4/24_data.fits
    mv lyt_alp_train_lac_20250731_170457_caldata_extra.fits lyt_no_aberrations_v4/25_data.fits
    mv lyt_alp_train_lac_20250731_170930_caldata_extra.fits lyt_no_aberrations_v4/26_data.fits
    mv lyt_alp_train_lac_20250731_171411_caldata_extra.fits lyt_no_aberrations_v4/27_data.fits
    # picture_d_aberrations_group_7, 4 chunks of 25k rows each
    mv lyt_alp_train_lac_20250731_171923_caldata_extra.fits lyt_no_aberrations_v4/28_data.fits
    mv lyt_alp_train_lac_20250731_172729_caldata_extra.fits lyt_no_aberrations_v4/29_data.fits
    mv lyt_alp_train_lac_20250731_173218_caldata_extra.fits lyt_no_aberrations_v4/30_data.fits
    mv lyt_alp_train_lac_20250731_173617_caldata_extra.fits lyt_no_aberrations_v4/31_data.fits
    # picture_d_aberrations_group_8, 2 chunks of 25k rows and 21k rows
    mv lyt_alp_train_lac_20250731_174007_caldata_extra.fits lyt_no_aberrations_v4/32_data.fits
    mv lyt_alp_train_lac_20250731_174613_caldata_extra.fits lyt_no_aberrations_v4/33_data.fits
    # picture_d_aberrations_group_9, 1 chunk of 25k rows
    mv lyt_alp_train_lac_20250731_175044_caldata_extra.fits lyt_no_aberrations_v4/34_data.fits

SEC4 - PREMERGE SOME FITS FILES ++++++++++++++++++++++++++++++++++++++++++++++++
For the datafiles in the `lyt_single_zernikes_v4` directory, there should be
only a single FITS datafile, not two.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    cd lyt_single_zernikes_v4
    
    # --In `python3` environment ---------------------
    from astropy.io import fits
    import numpy as np

    with fits.open('0_data.fits') as hdul:
        image_data_0 = hdul['IMAGE'].data
        zernike_data_0 = hdul['ZCMD'].data
        print(hdul['PRIMARY'].data)

    with fits.open('1_data.fits') as hdul:
        image_data_1 = hdul['IMAGE'].data
        zernike_data_1 = hdul['ZCMD'].data

    image_data = np.vstack((image_data_0, image_data_1))
    zernike_data = np.vstack((zernike_data_0, zernike_data_1))
    primary_hdu = fits.PrimaryHDU(data=np.array([]))
    image_hdu = fits.ImageHDU(data=image_data, name='IMAGE')
    zernike_hdu = fits.ImageHDU(data=zernike_data, name='ZCMD')
    hdul = fits.HDUList([primary_hdu, image_hdu, zernike_hdu])
    hdul.writeto('2_data.fits')
    # ------------------------------------------------

    rm lyt_single_zernikes_v4/0_data.fits
    rm lyt_single_zernikes_v4/1_data.fits
    mv lyt_single_zernikes_v4/2_data.fits lyt_single_zernikes_v4/0_data.fits

SEC5 - FITS TO HDF FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++
These FITS datafiles should be converted to HDF files. The format of the HDF
datafiles should be the same as the raw simulation datafiles.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py convert_picd_instrument_data picd_instrument_data_100k_groups_v4 2 24 \
        --fits-data-tags lyt_100k_chunks_v4
    python3 main.py convert_picd_instrument_data picd_instrument_data_25k_10nm_v4 2 24 \
        --fits-data-tags lyt_10nm_testing_v4
    python3 main.py convert_picd_instrument_data picd_instrument_data_no_aberrations_v4 2 24 \
        --fits-data-tags lyt_no_aberrations_v4 --base-field-data 0
    python3 main.py convert_picd_instrument_data picd_instrument_data_single_zernikes_v4 2 24 \
        --fits-data-tags lyt_single_zernikes_v4
    # Technically ±39.995 nm RMS error
    python3 main.py convert_picd_instrument_data picd_instrument_data_single_zernikes_pm40_v4 2 24 \
        --fits-data-tags lyt_single_zernikes_v4 --slice-row-ranges 4600 4623 41377 41400

SEC6 - PREPROCESS DATAFILES ++++++++++++++++++++++++++++++++++++++++++++++++++++
The newly converted HDF datafiles should be preprocessed.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Preprocess the data for training, validation, and testing
    python3 main.py preprocess_data_complete \
        picd_instrument_data_100k_groups_v4 \
        train_picd_data_v4 val_picd_data_v4 test_picd_data_v4 \
        80 15 5 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff picd_instrument_data_no_aberrations_v4 \
        --use-field-diff-mapping 0 0 25000 1 25000 50000 2 50000 75000 \
            3 75000 100000 4 100000 125000 5 125000 150000 6 150000 175000 \
            7 175000 200000 8 200000 225000 9 225000 250000 10 250000 275000 \
            11 275000 300000 12 300000 325000 13 325000 350000 14 350000 375000 \
            15 375000 400000 16 400000 425000 17 425000 450000 18 450000 475000 \
            19 475000 500000 20 500000 525000 21 525000 550000 22 550000 575000 \
            23 575000 600000 24 600000 625000 25 625000 650000 26 650000 675000 \
            27 675000 700000 28 700000 725000 29 725000 750000 30 750000 775000 \
            31 775000 800000 32 800000 825000 33 825000 846000 \
        --additional-raw-data-tags-train-only picd_instrument_data_single_zernikes_v4

    # Preprocess the testing data
    python3 main.py preprocess_data_bare picd_instrument_data_single_zernikes_v4 \
        picd_instrument_data_single_zernikes_raw_processed_v4
    python3 main.py preprocess_data_bare picd_instrument_data_25k_10nm_v4 \
        picd_instrument_data_25k_10nm_raw_processed_v4

SEC7 - [V55a] CNN TRAINING AND TESTING +++++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN model created from the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main_scnp.py model_train picd_cnn_v4_round_one \
        train_picd_data_v4 val_picd_data_v4 \
        speedup_6 mae adam 1e-5 1000 --batch-size 256 \
        --overwrite-existing --only-best-epoch --early-stopping 15
    python3 main_scnp.py model_train picd_cnn_v4 \
        train_picd_data_v4 val_picd_data_v4 \
        speedup_6 mae adam 1e-8 200 --batch-size 128 \
        --overwrite-existing --only-best-epoch --early-stopping 15 \
        --init-weights picd_cnn_v4_round_one last

    python3 main.py model_test picd_cnn_v4 last \
        test_picd_data_v4 --scatter-plot 4 6 2 0 15
    python3 main.py model_test picd_cnn_v4 last \
        picd_instrument_data_25k_10nm_raw_processed_v4 \
        --scatter-plot 4 6 2 1e-7 15 --inputs-need-norm --inputs-need-diff \
        --change-base-field picd_instrument_data_no_aberrations_v4 31 0 25000
    python3 main.py model_test picd_cnn_v4 last \
        picd_instrument_data_single_zernikes_raw_processed_v4 \
        --zernike-plots --inputs-need-norm --inputs-need-diff \
        --change-base-field picd_instrument_data_no_aberrations_v4 32 0 25000 33 25000 46000

    # Export the model so that it can be used in the `pytorch_model_in_c` repo
    # via ONNX runtime.
    python3 main.py export_model picd_cnn_v4 last val_picd_data_v4 --benchmark 5000

    # The following commands, when run in the newly exported model's directory,
    # will prep the files to be run in the flight software.
    mv example_data/first_input_row_norm.txt example_data/input_line.txt
    mv example_data/first_output_row_truth.txt example_data/output_line.txt
    mv example_data/first_output_row_onnx.txt example_data/model_output_line.txt
    rm example_data/first_output_row_norm_onnx.txt
    rm example_data/first_output_row_norm_truth.txt
    rm example_data/first_output_row_norm_ts.txt
    rm example_data/first_output_row_ts.txt
    rm model.pt
    rm README.txt
    # Keep only the last two lines of the normalization data file.
    tail -n 2 norm_data.txt > temp.txt && mv temp.txt norm_data.txt

SEC8 - RM CREATION AND TESTING +++++++++++++++++++++++++++++++++++++++++++++++++
Create and test the RM model on the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py create_response_matrix \
        --simulated-data-tag-average picd_instrument_data_single_zernikes_pm40_v4 \
        --base-field-tag picd_instrument_data_no_aberrations_v4 \
        --wfs-sum-to-one --base-field-mapping 32 0 23 33 23 46

    python3 main.py run_response_matrix picd_instrument_data_single_zernikes_pm40_v4 \
        picd_instrument_data_25k_10nm_raw_processed_v4 \
        --scatter-plot 4 6 2 1e-8 15 --wfs-need-sum-to-one \
        --change-base-field picd_instrument_data_no_aberrations_v4 31 0 25000
    python3 main.py run_response_matrix picd_instrument_data_single_zernikes_pm40_v4 \
        picd_instrument_data_single_zernikes_raw_processed_v4 --zernike-plots \
        --wfs-need-sum-to-one \
        --change-base-field picd_instrument_data_no_aberrations_v4 32 0 25000 33 25000 46000

SEC9 - (EXTRA) [V55b] CNN TRAINING AND TESTING +++++++++++++++++++++++++++++++++
The commands to train and test the [V55b] CNN model.
This CNN is too slow to run on the older, slower flight computer.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main_scnp.py model_train picd_cnn_v4_v55b_round_one \
        train_picd_data_v4 val_picd_data_v4 \
        best32_1_smaller_10 mae adam 1e-5 1000 --batch-size 256 \
        --overwrite-existing --only-best-epoch --early-stopping 15
    python3 main_scnp.py model_train picd_cnn_v4_v55b \
        train_picd_data_v4 val_picd_data_v4 \
        best32_1_smaller_10 mae adam 1e-8 200 --batch-size 128 \
        --overwrite-existing --only-best-epoch --early-stopping 15 \
        --init-weights picd_cnn_v4_v55b_round_one last

    python3 main.py model_test picd_cnn_v4_v55b last \
        test_picd_data_v4 --scatter-plot 4 6 2 0 15
    python3 main.py model_test picd_cnn_v4_v55b last \
        picd_instrument_data_25k_10nm_raw_processed_v4 \
        --scatter-plot 4 6 2 1e-7 15 --inputs-need-norm --inputs-need-diff \
        --change-base-field picd_instrument_data_no_aberrations_v4 31 0 25000
    python3 main.py model_test picd_cnn_v4_v55b last \
        picd_instrument_data_single_zernikes_raw_processed_v4 \
        --zernike-plots --inputs-need-norm --inputs-need-diff \
        --change-base-field picd_instrument_data_no_aberrations_v4 32 0 25000 33 25000 46000
