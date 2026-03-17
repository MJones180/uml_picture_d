....................................................................................................
NOTES AND CHANGES COMPARED TO `picture_d_llowfs_v4.md`:
- In simulation, switched to a two CNN approach: a Capture CNN (large aberrations) and a
  Stabilization CNN (small aberrations). Additionally, created versions of both which are trained
  on wavefronts with simulated camera noise.
- CNNs are trained with transfer learning on the simulated models with camera noise.
- The CNNs now use the `llowfs_cnn_4_no_dropout` architecture.
- Obtained new PICTURE-D instrument data on March 11, 2026.
    - ZIP of the data stored at `data/raw/llowfs_training_03_11_2026.zip`.
    - Instrument did not have the 540-660 nm band pass on when data was collected;
      instead, unfiltered white light was used.
    - The data contains two frames for each row, but only the second frame should be used.
    - The coefficients written to the datafiles are incorrect, so the original coefficients
      from the CSV files must be used.
    - All the images need to be flipped horizontally to align with the simulated data.
....................................................................................................

These are the commands needed to obtain data on PICTURE-D and to create the associated models.
Most of the commands are based on the ones used to simulate data in this repo; the commands are
taken/referenced from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error.
The CNNs are based on:
    - Capture CNN: [V72a] `wavefront_capture_sim_cam_v5`
    - Stabilization CNN: [V72b] The `wavefront_stabilization_sim_cam_v5`.

TABLE OF CONTENTS:
    SEC1 - INPUT ABERRATION CSV FILES
    SEC2 - COPY CSV COEFFICIENTS
    SEC3 - MOVE FITS FILES
    SEC4 - FITS TO HDF FILES
    SEC5 - PREPROCESS DATAFILES
    SEC6 - CNN TRANSFER TRAINING AND TESTING
    SEC7 - RM CREATION AND TESTING

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
    python3 main.py sim_data f_pm_40_coeffs no_prop 0 \
        --fixed-amount-per-zernike-pm 2 24 40e-9 \
        --append-no-aberrations-row --save-aberrations-csv-quit

    # ==== CNN Random Data ====

    # Set the number of rows in each random dataset
    # SET VALUE: Small - 1000, Med - 10000, Full - 100000
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
    python3 main.py sim_data r_25_1_half_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
        --rand-amount-per-zernike $NUMB_ROWS 2 3 " -25e-9" 25e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_2_1` - (15, 2, 1)
    python3 main.py sim_data r_15_2_1_coeffs_${NUMB_ROWS_TAG}k no_prop 0 \
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
    python3 main.py sim_data f_50_151_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 151 \
        --save-aberrations-csv-quit
    # Med - [-50, 50] with 501 points (11523 total)
    python3 main.py sim_data f_50_501_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 501 \
        --save-aberrations-csv-quit
    # Full - [-50, 50] with 2000 points (46000 total)
    python3 main.py sim_data f_50_2000_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --save-aberrations-csv-quit

    # Based on `fixed_1nm_range_301_approx` - used for the Stabilization CNN
    # Small - [-1, 1] with 51 points (1173 total)
    python3 main.py sim_data f_1_51_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 51 \
        --save-aberrations-csv-quit
    # Med / Full - [-1, 1] with 301 points (6923 total)
    python3 main.py sim_data f_1_301_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 301 \
        --save-aberrations-csv-quit

SEC2 - COPY CSV COEFFICIENTS +++++++++++++++++++++++++++++++++++++++++++++++++++
The CSV coefficients must be copied to the `data/raw/` folder so that they can
be used when converting the data from FITS to HDF. The reason for this is that
the incorrect coefficients were written out when obtaining the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    cd data/raw
    mkdir picd_inst_coeffs_v5
    cd picd_inst_coeffs_v5

    mkdir rm_data
    mv f_pm_40_coeffs/aberrations.csv rm_data/f_pm_40_coeffs.csv
    rm -rf f_pm_40_coeffs

    mkdir sm_data
    mv r_10_2_1_coeffs_1k/aberrations.csv sm_data/r_10_2_1_coeffs_1k.csv
    mv r_500_20_10_coeffs_1k/aberrations.csv sm_data/r_500_20_10_coeffs_1k.csv
    mv r_50_10_5_coeffs_1k/aberrations.csv sm_data/r_50_10_5_coeffs_1k.csv
    mv r_15_5_2_coeffs_1k/aberrations.csv sm_data/r_15_5_2_coeffs_1k.csv
    mv r_10_coeffs_1k/aberrations.csv sm_data/r_10_coeffs_1k.csv
    mv r_25_1_half_coeffs_1k/aberrations.csv sm_data/r_25_1_half_coeffs_1k.csv
    mv r_15_2_1_coeffs_1k/aberrations.csv sm_data/r_15_2_1_coeffs_1k.csv
    mv r_15_1_half_coeffs_1k/aberrations.csv sm_data/r_15_1_half_coeffs_1k.csv
    mv r_10_1_half_coeffs_1k/aberrations.csv sm_data/r_10_1_half_coeffs_1k.csv
    mv r_10_half_quarter_coeffs_1k/aberrations.csv sm_data/r_10_half_quarter_coeffs_1k.csv
    mv r_half_quarter_fifth_coeffs_1k/aberrations.csv sm_data/r_half_quarter_fifth_coeffs_1k.csv
    mv r_1_coeffs_1k/aberrations.csv sm_data/r_1_coeffs_1k.csv
    mv r_2_coeffs_1k/aberrations.csv sm_data/r_2_coeffs_1k.csv
    mv f_50_151_coeffs/aberrations.csv sm_data/f_50_151_coeffs.csv
    mv f_1_51_coeffs/aberrations.csv sm_data/f_1_51_coeffs.csv
    rm -rf r_500_20_10_coeffs_1k
    rm -rf r_50_10_5_coeffs_1k
    rm -rf r_15_5_2_coeffs_1k
    rm -rf r_10_coeffs_1k
    rm -rf r_25_1_half_coeffs_1k
    rm -rf r_15_2_1_coeffs_1k
    rm -rf r_15_1_half_coeffs_1k
    rm -rf r_10_1_half_coeffs_1k
    rm -rf r_10_half_quarter_coeffs_1k
    rm -rf r_half_quarter_fifth_coeffs_1k
    rm -rf r_1_coeffs_1k
    rm -rf r_2_coeffs_1k
    rm -rf f_50_151_coeffs
    rm -rf f_1_51_coeffs

    mkdir md_data
    mv r_10_2_1_coeffs_10k/aberrations.csv md_data/r_10_2_1_coeffs_10k.csv
    mv r_500_20_10_coeffs_10k/aberrations.csv md_data/r_500_20_10_coeffs_10k.csv
    mv r_50_10_5_coeffs_10k/aberrations.csv md_data/r_50_10_5_coeffs_10k.csv
    mv r_15_5_2_coeffs_10k/aberrations.csv md_data/r_15_5_2_coeffs_10k.csv
    mv r_10_coeffs_10k/aberrations.csv md_data/r_10_coeffs_10k.csv
    mv r_25_1_half_coeffs_10k/aberrations.csv md_data/r_25_1_half_coeffs_10k.csv
    mv r_15_2_1_coeffs_10k/aberrations.csv md_data/r_15_2_1_coeffs_10k.csv
    mv r_15_1_half_coeffs_10k/aberrations.csv md_data/r_15_1_half_coeffs_10k.csv
    mv r_10_1_half_coeffs_10k/aberrations.csv md_data/r_10_1_half_coeffs_10k.csv
    mv r_10_half_quarter_coeffs_10k/aberrations.csv md_data/r_10_half_quarter_coeffs_10k.csv
    mv r_half_quarter_fifth_coeffs_10k/aberrations.csv md_data/r_half_quarter_fifth_coeffs_10k.csv
    mv r_1_coeffs_10k/aberrations.csv md_data/r_1_coeffs_10k.csv
    mv r_2_coeffs_10k/aberrations.csv md_data/r_2_coeffs_10k.csv
    mv f_50_501_coeffs/aberrations.csv md_data/f_50_501_coeffs.csv
    mv f_1_301_coeffs/aberrations.csv md_data/f_1_301_coeffs.csv
    rm -rf r_10_2_1_coeffs_10k
    rm -rf r_500_20_10_coeffs_10k
    rm -rf r_50_10_5_coeffs_10k
    rm -rf r_15_5_2_coeffs_10k
    rm -rf r_10_coeffs_10k
    rm -rf r_25_1_half_coeffs_10k
    rm -rf r_15_2_1_coeffs_10k
    rm -rf r_15_1_half_coeffs_10k
    rm -rf r_10_1_half_coeffs_10k
    rm -rf r_10_half_quarter_coeffs_10k
    rm -rf r_half_quarter_fifth_coeffs_10k
    rm -rf r_1_coeffs_10k
    rm -rf r_2_coeffs_10k
    rm -rf f_50_501_coeffs
    rm -rf f_1_301_coeffs

SEC3 - MOVE FITS FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The FITS datafiles containing instrument data should be moved to the `data/raw`
directory with new folder and file names. Datafiles for the Small and Medium
datasets can be found in the ZIP at `data/raw/llowfs_training_03_11_2026.zip`.
All the `*_extra` datafiles contain the base fields and go in the corresponding
`*_bf_*` folders. The shared data is duplicated in both the Capture and
Stabilization datasets to make preprocessing easier.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- RM ----
    mkdir inst_llowfs_v5_rm
    # f_pm_40_coeffs
    mv lyt_alp_train_lac_20260311_162642_caldata.fits       inst_llowfs_v5_rm/0_data.fits
    # This row is already baked into the dataset
    rm lyt_alp_train_lac_20260311_162642_caldata_extra.fits

    # ---- Random 2nm Testing ----
    mkdir inst_llowfs_v5_tst_rnd_2nm
    mkdir inst_llowfs_v5_tst_rnd_2nm_bf
    # r_2_coeffs_1k
    cp lyt_alp_train_lac_20260311_163212_caldata.fits       inst_llowfs_v5_tst_rnd_2nm/0_data.fits
    cp lyt_alp_train_lac_20260311_163212_caldata_extra.fits inst_llowfs_v5_tst_rnd_2nm_bf/0_data.fits

    # ---- Fixed [-1, 1] nm Testing ----
    mkdir inst_llowfs_v5_tst_fix_1nm
    mkdir inst_llowfs_v5_tst_fix_1nm_bf
    # f_1_51_coeffs
    cp lyt_alp_train_lac_20260311_162726_caldata.fits       inst_llowfs_v5_tst_fix_1nm/0_data.fits
    cp lyt_alp_train_lac_20260311_162726_caldata_extra.fits inst_llowfs_v5_tst_fix_1nm_bf/0_data.fits

    # ---- Fixed [-50, 50] nm Testing ----
    mkdir inst_llowfs_v5_tst_fix_50nm
    mkdir inst_llowfs_v5_tst_fix_50nm_bf
    # f_50_151_coeffs
    cp lyt_alp_train_lac_20260311_162755_caldata.fits       inst_llowfs_v5_tst_fix_50nm/0_data.fits
    cp lyt_alp_train_lac_20260311_162755_caldata_extra.fits inst_llowfs_v5_tst_fix_50nm_bf/0_data.fits

    # ---- Capture Data - Small ----
    mkdir inst_llowfs_v5_cap_sm
    mkdir inst_llowfs_v5_cap_bf_sm
    # r_500_20_10_coeffs_1k
    mv lyt_alp_train_lac_20260311_163233_caldata.fits       inst_llowfs_v5_cap_sm/0_data.fits
    mv lyt_alp_train_lac_20260311_163233_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/0_data.fits
    # r_50_10_5_coeffs_1k
    mv lyt_alp_train_lac_20260311_163304_caldata.fits       inst_llowfs_v5_cap_sm/1_data.fits
    mv lyt_alp_train_lac_20260311_163304_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/1_data.fits
    # r_15_5_2_coeffs_1k
    mv lyt_alp_train_lac_20260311_163123_caldata.fits       inst_llowfs_v5_cap_sm/2_data.fits
    mv lyt_alp_train_lac_20260311_163123_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/2_data.fits
    # r_10_coeffs_1k
    mv lyt_alp_train_lac_20260311_162942_caldata.fits       inst_llowfs_v5_cap_sm/3_data.fits
    mv lyt_alp_train_lac_20260311_162942_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/3_data.fits
    # f_50_151_coeffs
    mv lyt_alp_train_lac_20260311_162755_caldata.fits       inst_llowfs_v5_cap_sm/4_data.fits
    mv lyt_alp_train_lac_20260311_162755_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/4_data.fits
    # r_10_2_1_coeffs_1k
    cp lyt_alp_train_lac_20260311_162907_caldata.fits       inst_llowfs_v5_cap_sm/5_data.fits
    cp lyt_alp_train_lac_20260311_162907_caldata_extra.fits inst_llowfs_v5_cap_bf_sm/5_data.fits

    # ---- Stabilization Data - Small ----
    mkdir inst_llowfs_v5_sta_sm
    mkdir inst_llowfs_v5_sta_bf_sm
    # r_25_1_half_coeffs_1k
    mv lyt_alp_train_lac_20260311_163346_caldata.fits       inst_llowfs_v5_sta_sm/0_data.fits
    mv lyt_alp_train_lac_20260311_163346_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/0_data.fits
    # r_15_2_1_coeffs_1k
    mv lyt_alp_train_lac_20260311_163326_caldata.fits       inst_llowfs_v5_sta_sm/1_data.fits
    mv lyt_alp_train_lac_20260311_163326_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/1_data.fits
    # r_15_1_half_coeffs_1k
    mv lyt_alp_train_lac_20260311_163058_caldata.fits       inst_llowfs_v5_sta_sm/2_data.fits
    mv lyt_alp_train_lac_20260311_163058_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/2_data.fits
    # r_10_1_half_coeffs_1k
    mv lyt_alp_train_lac_20260311_162837_caldata.fits       inst_llowfs_v5_sta_sm/3_data.fits
    mv lyt_alp_train_lac_20260311_162837_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/3_data.fits
    # r_10_half_quarter_coeffs_1k
    mv lyt_alp_train_lac_20260311_163034_caldata.fits       inst_llowfs_v5_sta_sm/4_data.fits
    mv lyt_alp_train_lac_20260311_163034_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/4_data.fits
    # r_half_quarter_fifth_coeffs_1k
    mv lyt_alp_train_lac_20260311_163418_caldata.fits       inst_llowfs_v5_sta_sm/5_data.fits
    mv lyt_alp_train_lac_20260311_163418_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/5_data.fits
    # r_1_coeffs_1k
    mv lyt_alp_train_lac_20260311_163149_caldata.fits       inst_llowfs_v5_sta_sm/6_data.fits
    mv lyt_alp_train_lac_20260311_163149_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/6_data.fits
    # r_2_coeffs_1k
    mv lyt_alp_train_lac_20260311_163212_caldata.fits       inst_llowfs_v5_sta_sm/7_data.fits
    mv lyt_alp_train_lac_20260311_163212_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/7_data.fits
    # f_1_51_coeffs
    mv lyt_alp_train_lac_20260311_162726_caldata.fits       inst_llowfs_v5_sta_sm/8_data.fits
    mv lyt_alp_train_lac_20260311_162726_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/8_data.fits
    # r_10_2_1_coeffs_1k
    mv lyt_alp_train_lac_20260311_162907_caldata.fits       inst_llowfs_v5_sta_sm/9_data.fits
    mv lyt_alp_train_lac_20260311_162907_caldata_extra.fits inst_llowfs_v5_sta_bf_sm/9_data.fits

    # ---- Capture Data - Medium ----
    mkdir inst_llowfs_v5_cap_md
    mkdir inst_llowfs_v5_cap_bf_md
    # r_500_20_10_coeffs_10k
    mv lyt_alp_train_lac_20260311_161641_caldata.fits       inst_llowfs_v5_cap_md/0_data.fits
    mv lyt_alp_train_lac_20260311_161641_caldata_extra.fits inst_llowfs_v5_cap_bf_md/0_data.fits
    # r_50_10_5_coeffs_10k
    mv lyt_alp_train_lac_20260311_162115_caldata.fits       inst_llowfs_v5_cap_md/1_data.fits
    mv lyt_alp_train_lac_20260311_162115_caldata_extra.fits inst_llowfs_v5_cap_bf_md/1_data.fits
    # r_15_5_2_coeffs_10k
    mv lyt_alp_train_lac_20260311_161216_caldata.fits       inst_llowfs_v5_cap_md/2_data.fits
    mv lyt_alp_train_lac_20260311_161216_caldata_extra.fits inst_llowfs_v5_cap_bf_md/2_data.fits
    # r_10_coeffs_10k
    mv lyt_alp_train_lac_20260311_160754_caldata.fits       inst_llowfs_v5_cap_md/3_data.fits
    mv lyt_alp_train_lac_20260311_160754_caldata_extra.fits inst_llowfs_v5_cap_bf_md/3_data.fits
    # f_50_501_coeffs
    mv lyt_alp_train_lac_20260311_160314_caldata.fits       inst_llowfs_v5_cap_md/4_data.fits
    mv lyt_alp_train_lac_20260311_160314_caldata_extra.fits inst_llowfs_v5_cap_bf_md/4_data.fits
    # r_10_2_1_coeffs_10k
    cp lyt_alp_train_lac_20260311_160615_caldata.fits       inst_llowfs_v5_cap_md/5_data.fits
    cp lyt_alp_train_lac_20260311_160615_caldata_extra.fits inst_llowfs_v5_cap_bf_md/5_data.fits

    # ---- Stabilization Data - Medium ----
    mkdir inst_llowfs_v5_sta_md
    mkdir inst_llowfs_v5_sta_bf_md
    # r_25_1_half_coeffs_10k
    mv lyt_alp_train_lac_20260311_162400_caldata.fits       inst_llowfs_v5_sta_md/0_data.fits
    mv lyt_alp_train_lac_20260311_162400_caldata_extra.fits inst_llowfs_v5_sta_bf_md/0_data.fits
    # r_15_2_1_coeffs_10k
    mv lyt_alp_train_lac_20260311_162240_caldata.fits       inst_llowfs_v5_sta_md/1_data.fits
    mv lyt_alp_train_lac_20260311_162240_caldata_extra.fits inst_llowfs_v5_sta_bf_md/1_data.fits
    # r_15_1_half_coeffs_10k
    mv lyt_alp_train_lac_20260311_161026_caldata.fits       inst_llowfs_v5_sta_md/2_data.fits
    mv lyt_alp_train_lac_20260311_161026_caldata_extra.fits inst_llowfs_v5_sta_bf_md/2_data.fits
    # r_10_1_half_coeffs_10k
    mv lyt_alp_train_lac_20260311_160447_caldata.fits       inst_llowfs_v5_sta_md/3_data.fits
    mv lyt_alp_train_lac_20260311_160447_caldata_extra.fits inst_llowfs_v5_sta_bf_md/3_data.fits
    # r_10_half_quarter_coeffs_10k
    mv lyt_alp_train_lac_20260311_160908_caldata.fits       inst_llowfs_v5_sta_md/4_data.fits
    mv lyt_alp_train_lac_20260311_160908_caldata_extra.fits inst_llowfs_v5_sta_bf_md/4_data.fits
    # r_half_quarter_fifth_coeffs_10k
    mv lyt_alp_train_lac_20260311_162518_caldata.fits       inst_llowfs_v5_sta_md/5_data.fits
    mv lyt_alp_train_lac_20260311_162518_caldata_extra.fits inst_llowfs_v5_sta_bf_md/5_data.fits
    # r_1_coeffs_10k
    mv lyt_alp_train_lac_20260311_161325_caldata.fits       inst_llowfs_v5_sta_md/6_data.fits
    mv lyt_alp_train_lac_20260311_161325_caldata_extra.fits inst_llowfs_v5_sta_bf_md/6_data.fits
    # r_2_coeffs_10k
    mv lyt_alp_train_lac_20260311_161525_caldata.fits       inst_llowfs_v5_sta_md/7_data.fits
    mv lyt_alp_train_lac_20260311_161525_caldata_extra.fits inst_llowfs_v5_sta_bf_md/7_data.fits
    # f_1_301_coeffs
    mv lyt_alp_train_lac_20260311_160209_caldata.fits       inst_llowfs_v5_sta_md/8_data.fits
    mv lyt_alp_train_lac_20260311_160209_caldata_extra.fits inst_llowfs_v5_sta_bf_md/8_data.fits
    # r_10_2_1_coeffs_10k
    mv lyt_alp_train_lac_20260311_160615_caldata.fits       inst_llowfs_v5_sta_md/9_data.fits
    mv lyt_alp_train_lac_20260311_160615_caldata_extra.fits inst_llowfs_v5_sta_bf_md/9_data.fits

SEC4 - FITS TO HDF FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++
These FITS datafiles should be converted to HDF files. The format of the HDF
datafiles should be the same as the raw simulation datafiles. When converting
these datafiles, the duplicate rows are removed -- this is only for the actual
data, not the base field data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- RM - 47 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_rm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_rm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/rm_data f_pm_40_coeffs

    # ---- Random 2nm Testing - 1,000 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_rnd_2nm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_rnd_2nm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/sm_data r_2_coeffs_1k
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_rnd_2nm_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_rnd_2nm_bf --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Fixed [-1, 1] nm Testing - 1,173 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_fix_1nm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_fix_1nm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/sm_data f_1_51_coeffs
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_fix_1nm_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_fix_1nm_bf --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Fixed [-50, 50] nm Testing - 3,473 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_fix_50nm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_fix_50nm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/sm_data f_50_151_coeffs
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_tst_fix_50nm_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_tst_fix_50nm_bf --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Capture Data - Small - 8,473 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_cap_sm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_cap_sm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/sm_data r_500_20_10_coeffs_1k r_50_10_5_coeffs_1k \
            r_15_5_2_coeffs_1k r_10_coeffs_1k f_50_151_coeffs r_10_2_1_coeffs_1k
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_cap_bf_sm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_cap_bf_sm --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Stabilization Data - Small - 10,173 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_sta_sm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_sta_sm --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/sm_data r_25_1_half_coeffs_1k r_15_2_1_coeffs_1k \
            r_15_1_half_coeffs_1k r_10_1_half_coeffs_1k r_10_half_quarter_coeffs_1k \
            r_half_quarter_fifth_coeffs_1k r_1_coeffs_1k r_2_coeffs_1k f_1_51_coeffs r_10_2_1_coeffs_1k
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_sta_bf_sm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_sta_bf_sm --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Capture Data - Medium - 61,523 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_cap_md_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_cap_md --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/md_data r_500_20_10_coeffs_10k r_50_10_5_coeffs_10k \
            r_15_5_2_coeffs_10k r_10_coeffs_10k f_50_501_coeffs r_10_2_1_coeffs_10k
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_cap_bf_md_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_cap_bf_md --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

    # ---- Stabilization Data - Medium - 96,923 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_sta_md_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_sta_md --take-every-n-rows 2 1 --flip-images-horizontally \
        --use-coeffs-from-csv picd_inst_coeffs_v5/md_data r_25_1_half_coeffs_10k r_15_2_1_coeffs_10k \
            r_15_1_half_coeffs_10k r_10_1_half_coeffs_10k r_10_half_quarter_coeffs_10k \
            r_half_quarter_fifth_coeffs_10k r_1_coeffs_10k r_2_coeffs_10k f_1_301_coeffs r_10_2_1_coeffs_10k
    python3 main.py convert_picd_instrument_data inst_llowfs_v5_sta_bf_md_hdf 2 24 \
        --fits-data-tags inst_llowfs_v5_sta_bf_md --base-field-data 0 --n-base-field-rows 350 \
        --flip-images-horizontally

SEC5 - PREPROCESS DATAFILES ++++++++++++++++++++++++++++++++++++++++++++++++++++
The newly converted HDF datafiles should be preprocessed.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- Training and Validation Datasets ----
    python3 main.py preprocess_data_complete \
        inst_llowfs_v5_cap_md_hdf \
        train_picd_data_v5_cap val_picd_data_v5_cap none 90 10 0 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff inst_llowfs_v5_cap_bf_md_hdf \
        --use-field-diff-mapping 0 0     10000 1 10000 20000 2 20000 30000 \
                                 3 30000 40000 4 40000 51523 5 51523 61523 \
        --fix-seed 314
    python3 main.py preprocess_data_complete \
        inst_llowfs_v5_sta_md_hdf \
        train_picd_data_v5_sta val_picd_data_v5_sta none 90 10 0 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --use-field-diff inst_llowfs_v5_sta_bf_md_hdf \
        --use-field-diff-mapping 0 0     10000 1 10000 20000 2 20000 30000 3 30000 40000 \
                                 4 40000 50000 5 50000 60000 6 60000 70000 7 70000 80000 \
                                 8 80000 86923 9 86923 96923 \
        --fix-seed 314

    # ---- Testing Datasets ----
    python3 main.py preprocess_data_bare inst_llowfs_v5_cap_sm_hdf \
        inst_llowfs_v5_cap_sm_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v5_sta_sm_hdf \
        inst_llowfs_v5_sta_sm_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v5_tst_rnd_2nm_hdf \
        inst_llowfs_v5_tst_rnd_2nm_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v5_tst_fix_1nm_hdf \
        inst_llowfs_v5_tst_fix_1nm_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v5_tst_fix_50nm_hdf \
        inst_llowfs_v5_tst_fix_50nm_hdf_proc

SEC6 - CNN TRANSFER TRAINING AND TESTING +++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN models created from the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- Capture CNN Training ----
    python3 main_scnp.py model_train instrument_llowfs_capture_v5_scratch \
        train_picd_data_v5_cap val_picd_data_v5_cap \
        llowfs_cnn_4_no_dropout mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --overwrite-existing --only-best-epoch --fix-seed 314
    python3 main_scnp.py model_train instrument_llowfs_capture_v5_ext \
        train_picd_data_v5_cap val_picd_data_v5_cap \
        llowfs_cnn_4_no_dropout mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_capture_sim_cam_v5 last \
        --overwrite-existing --only-best-epoch --fix-seed 314

    # ---- Stabilization CNN Training ----
    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5_scratch \
        train_picd_data_v5_sta val_picd_data_v5_sta \
        llowfs_cnn_4_no_dropout mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --overwrite-existing --only-best-epoch --fix-seed 314
    python3 main_scnp.py model_train instrument_llowfs_stabilization_v5_ext \
        train_picd_data_v5_sta val_picd_data_v5_sta \
        llowfs_cnn_4_no_dropout mae adam 10e-5 1000 --batch-size 256 \
        --lr-auto-annealing 10e-7 10 --early-stopping 15 \
        --init-weights wavefront_stabilization_sim_cam_v5 last \
        --overwrite-existing --only-best-epoch --fix-seed 314

    # ---- Capture CNN Testing ----
    python3 main.py model_test instrument_llowfs_capture_v5_scratch last \
        inst_llowfs_v5_cap_sm_hdf_proc \
        --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_cap_bf_sm_hdf 0 0    1000 1 1000 2000 \
                                                         2 2000 3000 3 3000 4000 \
                                                         4 4000 7473 5 7473 8473 \
    python3 main.py model_test instrument_llowfs_capture_v5_ext last \
        inst_llowfs_v5_cap_sm_hdf_proc \
        --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_cap_bf_sm_hdf 0 0    1000 1 1000 2000 \
                                                         2 2000 3000 3 3000 4000 \
                                                         4 4000 7473 5 7473 8473 \
    python3 main.py model_test instrument_llowfs_capture_v5_scratch last \
        inst_llowfs_v5_tst_fix_50nm_hdf_proc \
        --zernike-plots --enable-paper-plots 1 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_fix_50nm_bf_hdf 0 0 3473
    python3 main.py model_test instrument_llowfs_capture_v5_ext last \
        inst_llowfs_v5_tst_fix_50nm_hdf_proc \
        --zernike-plots --enable-paper-plots 1 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_fix_50nm_bf_hdf 0 0 3473

    # ---- Stabilization CNN Testing ----
    python3 main.py model_test instrument_llowfs_stabilization_v5_scratch last \
        inst_llowfs_v5_sta_sm_hdf_proc \
        --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_sta_bf_sm_hdf 0 0    1000 1 1000 2000 \
                                                         2 2000 3000 3 3000 4000 \
                                                         4 4000 5000 5 5000 6000 \
                                                         6 6000 7000 7 7000 8000 \
                                                         8 8000 9173 9 9173 10173
    python3 main.py model_test instrument_llowfs_stabilization_v5_ext last \
        inst_llowfs_v5_sta_sm_hdf_proc \
        --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_sta_bf_sm_hdf 0 0    1000 1 1000 2000 \
                                                         2 2000 3000 3 3000 4000 \
                                                         4 4000 5000 5 5000 6000 \
                                                         6 6000 7000 7 7000 8000 \
                                                         8 8000 9173 9 9173 10173
    python3 main.py model_test instrument_llowfs_stabilization_v5_scratch last \
        inst_llowfs_v5_tst_fix_1nm_hdf_proc \
        --zernike-plots --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_fix_1nm_bf_hdf 0 0 1173
    python3 main.py model_test instrument_llowfs_stabilization_v5_ext last \
        inst_llowfs_v5_tst_fix_1nm_hdf_proc \
        --zernike-plots --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_fix_1nm_bf_hdf 0 0 1173
    python3 main.py model_test instrument_llowfs_stabilization_v5_scratch last \
        inst_llowfs_v5_tst_rnd_2nm_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_rnd_2nm_bf_hdf 0 0 1000
    python3 main.py model_test instrument_llowfs_stabilization_v5_ext last \
        inst_llowfs_v5_tst_rnd_2nm_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field inst_llowfs_v5_tst_rnd_2nm_bf_hdf 0 0 1000

SEC7 - RM CREATION AND TESTING +++++++++++++++++++++++++++++++++++++++++++++++++
Create and test the RM model on the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py create_response_matrix \
        --simulated-data-tag-average inst_llowfs_v5_rm_hdf --wfs-sum-to-one
    python3 main.py run_response_matrix inst_llowfs_v5_rm_hdf \
        inst_llowfs_v5_tst_fix_50nm_hdf_proc \
        --zernike-plots --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field inst_llowfs_v5_tst_fix_50nm_bf_hdf 0 0 3473
    python3 main.py run_response_matrix inst_llowfs_v5_rm_hdf \
        inst_llowfs_v5_tst_fix_1nm_hdf_proc \
        --zernike-plots --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field inst_llowfs_v5_tst_fix_1nm_bf_hdf 0 0 1173
    python3 main.py run_response_matrix inst_llowfs_v5_rm_hdf \
        inst_llowfs_v5_tst_rnd_2nm_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field inst_llowfs_v5_tst_rnd_2nm_bf_hdf 0 0 1000
