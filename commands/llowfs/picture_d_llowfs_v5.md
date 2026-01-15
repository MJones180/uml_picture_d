....................................................................................................
NOTES AND CHANGES COMPARED TO `picture_d_llowfs_v4.md`:
- In simulation, switched to a two CNN approach: a Capture CNN (large aberrations) and a
  Stabilization CNN (small aberrations). Additionally, created versions of both which are trained
  on wavefronts with simulated camera noise.
- The CNNs now use the `llowfs_cnn_4` architecture.
- Obtained data on the PICTURE-D instrument and used it to extend train the CNN models with noise.
....................................................................................................

These are the commands needed to obtain data on PICTURE-D and to create the associated models.
Most of the commands are based on the ones used to simulate data in this repo; the commands are
taken/referenced from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error.
The CNNs are based on:
    - Capture CNN: [V70a] `wavefront_capture_sim_cam_v4`
    - Stabilization CNN: [V70b] The `wavefront_stabilization_sim_cam_v4`.

TABLE OF CONTENTS:
    SEC1 - INPUT ABERRATION CSV FILES
    SEC3 - MOVE FITS FILES
    SEC4 - PREMERGE SOME FITS FILES
    SEC5 - FITS TO HDF FILES
    SEC6 - PREPROCESS DATAFILES
    SEC7 - CNN TRAINING AND TESTING
    SEC8 - RM CREATION AND TESTING

SEC1 - INPUT ABERRATION CSV FILES ++++++++++++++++++++++++++++++++++++++++++++++
The CSV files containing the aberrations that will be run on the instrument.
Base fields will automatically be saved along with the output FITS datafiles.
The aberrations for each dataset are given by a tuple where the coefficients
correspond to (Z2-3, Z4-8, Z9-24); all coefficients are uniformly random between
Each dataset is based on a simulated dataset which was used to train the
simulated CNNs. Each of the random datasets contains the same number of rows.
Each random dataset has the same number of rows, as specified by the `NUMB_ROWS`
argument -- this must be set. Additionally, based on the number of rows, the
corresponding fixed datasets must also be created.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==== RM Data ====

    # Based on `fixed_pm_40nm` - +/- 40 nm (47 total)
    python3 main.py sim_data f_pm_40 no_prop 0 \
        --fixed-amount-per-zernike-pm 2 24 40e-9 \
        --append-no-aberrations-row --save-aberrations-csv-quit

    # ==== CNN Random Data ====

    # Set the number of rows in each random dataset
    # SET VALUE: Small - 1000, Med - 10000
    export NUMB_ROWS=1000

    # The tag in the dataset name is just divided by 1k
    export NUMB_ROWS_TAG=$(($NUMB_ROWS/1000))

    # ---- Shared Dataset ----
    # Based on `random_group_10_2_1` - (10, 2, 1)
    python3 main.py sim_data r_10_2_1_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -10e-9" 10e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit

    # ---- Capture CNN Datasets ----
    # Based on `random_group_500_20_10` - (500, 20, 10)
    python3 main.py sim_data r_500_20_10_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_50_10_5` - (50, 10, 5)
    python3 main.py sim_data r_50_10_5_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -50e-9" 50e-9 4 8 " -10e-9" 10e-9 9 24 " -5e-9" 5e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_5_2` - (15, 5, 2)
    python3 main.py sim_data r_15_5_2_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -15e-9" 15e-9 4 8 " -5e-9" 5e-9 9 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit
    # Based on `random_10nm_large_approx` - (10, 10, 10)
    python3 main.py sim_data r_10_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- Stabilization CNN Datasets ----
    # Based on `random_group_25_1_half` - (25, 1, 0.5)
    python3 main.py sim_data r_group_25_1_half_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -25e-9" 25e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_2_1` - (15, 2, 1)
    python3 main.py sim_data r_group_15_2_1_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -15e-9" 15e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_1_half` - (15, 1, 0.5)
    python3 main.py sim_data r_15_1_half_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -15e-9" 15e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_10_1_half` - (10, 1, 0.5)
    python3 main.py sim_data r_10_1_half_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -10e-9" 10e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_10_half_quarter` - (10, 0.5, 0.25)
    python3 main.py sim_data r_10_half_quarter_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -10e-9" 10e-9 4 8 " -5e-10" 5e-10 9 24 " -2.5e-10" 2.5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_half_quarter_fifth` - (0.5, 0.25, 0.2)
    python3 main.py sim_data r_half_quarter_fifth_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -5e-10" 5e-10 4 8 " -2.5e-10" 2.5e-10 9 24 " -2e-10" 2e-10 \
        --save-aberrations-csv-quit
    # Based on `random_1nm_large_approx` - (1, 1, 1)
    python3 main.py sim_data r_1_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit
    # Based on `random_2nm_large_approx` - (2, 2, 2)
    python3 main.py sim_data r_2_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ==== CNN Fixed Data ====

    # Based on `fixed_50nm_range_2000_approx` - used for the Capture CNN
    # Small - [-50, 50] with 151 points (3473 total)
    python3 main.py sim_data f_50_151 no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 151 \
        --save-aberrations-csv-quit
    # Med - [-50, 50] with 501 points (11523 total)
    python3 main.py sim_data f_50_501 no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 501 \
        --save-aberrations-csv-quit

    # Based on `fixed_1nm_range_301_approx` - used for the Stabilization CNN
    # Small - [-1, 1] with 51 points (1173 total)
    python3 main.py sim_data f_1_51 no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 51 \
        --save-aberrations-csv-quit
    # Med - [-1, 1] with 301 points (6923 total)
    python3 main.py sim_data f_1_301 no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 301 \
        --save-aberrations-csv-quit

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
    
    # -- In `python3` environment --------------------
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

SEC7 - CNN TRAINING AND TESTING ++++++++++++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN models created from the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ---- Capture CNN ----

    python3 main_scnp.py model_train instrument_llowfs_capture_v5a \
        train_instrument_llowfs_capture_v5 \
        val_instrument_llowfs_capture_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch

    python3 main_scnp.py model_train instrument_llowfs_capture_v5b \
        train_instrument_llowfs_capture_v5 \
        val_instrument_llowfs_capture_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers dense_block1 out_layer

    python3 main_scnp.py model_train instrument_llowfs_capture_v5c \
        train_instrument_llowfs_capture_v5 \
        val_instrument_llowfs_capture_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers dense_block1 out_layer \
        --transfer-learning-batchnorm

    python3 main_scnp.py model_train instrument_llowfs_capture_v5d \
        train_instrument_llowfs_capture_v5 \
        val_instrument_llowfs_capture_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers conv_block7 conv_block8 dense_block1 out_layer

    python3 main_scnp.py model_train instrument_llowfs_capture_v5e \
        train_instrument_llowfs_capture_v5 \
        val_instrument_llowfs_capture_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers conv_block7 conv_block8 dense_block1 out_layer \
        --transfer-learning-batchnorm

    ---- Stabilization CNN ----

    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5a \
        train_instrument_llowfs_stabilization_v5 \
        val_instrument_llowfs_stabilization_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch

    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5b \
        train_instrument_llowfs_stabilization_v5 \
        val_instrument_llowfs_stabilization_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers dense_block1 out_layer

    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5c \
        train_instrument_llowfs_stabilization_v5 \
        val_instrument_llowfs_stabilization_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers dense_block1 out_layer \
        --transfer-learning-batchnorm

    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5d \
        train_instrument_llowfs_stabilization_v5 \
        val_instrument_llowfs_stabilization_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers conv_block7 conv_block8 dense_block1 out_layer

    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5e \
        train_instrument_llowfs_stabilization_v5 \
        val_instrument_llowfs_stabilization_v5 \
        llowfs_cnn_4 mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v4 last \
        --overwrite-existing --only-best-epoch \
        --transfer-learning-train-layers conv_block7 conv_block8 dense_block1 out_layer \
        --transfer-learning-batchnorm



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
