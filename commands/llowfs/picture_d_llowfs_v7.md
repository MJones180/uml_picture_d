....................................................................................................
NOTES AND CHANGES COMPARED TO `picture_d_llowfs_v6.md`:
- Obtained new PICTURE-D instrument data on March 26, 2026.
    - Data stored on the Samsung SSD under `llowfs_instrument_data/v7/llowfs_training_03_26_2026`.
    - Using filtered white light: bandpass of 540-660nm.
    - The purge tank was turned off while collecting data.
    - Each row of data has a dedicated reference image associated with it.
    - There are two frames for any given row and the second frame should be used in each case.
      For a given row of Zernike coefficients, there are really four rows in total:
        reference img, reference img (use this one), aberrated img, aberrated img (use this one).
....................................................................................................

These are the commands needed to obtain data on PICTURE-D and to create the associated models.
Most of the commands are based on the ones used to simulate data in this repo; the commands are
taken/referenced from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error.
The CNNs are based on:
    - Capture CNN: [V74a] `wavefront_capture_sim_cam_v6`
    - Stabilization CNN: [V74b] The `wavefront_stabilization_sim_cam_v6`.

TABLE OF CONTENTS:
    SEC1 - INPUT ABERRATION CSV FILE CREATION
    SEC2 - INPUT ABERRATION CSV FILE PREP
    SEC3 - MOVE FITS FILES
    SEC4 - FITS TO HDF FILES
    SEC5 - PREPROCESS DATAFILES
    SEC6 - CNN TRAINING AND TESTING
    SEC7 - RM CREATION AND TESTING

SEC1 - INPUT ABERRATION CSV FILE CREATION ++++++++++++++++++++++++++++++++++++++
The CSV files containing the aberrations that will be run on the instrument.
Base fields will automatically be saved along with the output FITS datafiles.
The aberrations for each dataset are given by a tuple where the coefficients
correspond to (Z2-3, Z4-8, Z9-24); all random coefficients are uniformly random.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ==== RM Data ====

    # Based on `fixed_pm_40nm` - +/- 40 nm (47 total)
    python3 main.py sim_data f_pm_40_coeffs no_prop 0 \
        --fixed-amount-per-zernike-pm 2 24 40e-9 \
        --append-no-aberrations-row --save-aberrations-csv-quit

    # ==== CNN Random Data ====

    # ---- Shared Datasets ----
    # Based on `random_group_10_2_1` - (10, 2, 1)
    python3 main.py sim_data r_10_2_1_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit
    # Based on `random_2nm_med_approx` - (2, 2, 2) - Used for testing
    python3 main.py sim_data r_2_coeffs_50k no_prop 0 \
        --rand-amount-per-zernike 50000 2 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ---- Capture CNN Datasets ----
    # Based on `random_group_500_20_10` - (500, 20, 10)
    python3 main.py sim_data r_500_20_10_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -500e-9" 500e-9 4 8 " -20e-9" 20e-9 9 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_50_10_5` - (50, 10, 5)
    python3 main.py sim_data r_50_10_5_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -50e-9" 50e-9 4 8 " -10e-9" 10e-9 9 24 " -5e-9" 5e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_5_2` - (15, 5, 2)
    python3 main.py sim_data r_15_5_2_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -5e-9" 5e-9 9 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit
    # Based on `random_10nm_large_approx` - (10, 10, 10)
    python3 main.py sim_data r_10_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 24 " -10e-9" 10e-9 \
        --save-aberrations-csv-quit

    # ---- Stabilization CNN Datasets ----
    # Based on `random_group_25_1_half` - (25, 1, 0.5)
    python3 main.py sim_data r_25_1_half_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -25e-9" 25e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_2_1` - (15, 2, 1)
    python3 main.py sim_data r_15_2_1_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -2e-9" 2e-9 9 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit
    # Based on `random_group_15_1_half` - (15, 1, 0.5)
    python3 main.py sim_data r_15_1_half_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -15e-9" 15e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_10_1_half` - (10, 1, 0.5)
    python3 main.py sim_data r_10_1_half_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -1e-9" 1e-9 9 24 " -5e-10" 5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_10_half_quarter` - (10, 0.5, 0.25)
    python3 main.py sim_data r_10_half_quarter_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -10e-9" 10e-9 4 8 " -5e-10" 5e-10 9 24 " -2.5e-10" 2.5e-10 \
        --save-aberrations-csv-quit
    # Based on `random_group_half_quarter_fifth` - (0.5, 0.25, 0.2)
    python3 main.py sim_data r_half_quarter_fifth_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 3 " -5e-10" 5e-10 4 8 " -2.5e-10" 2.5e-10 9 24 " -2e-10" 2e-10 \
        --save-aberrations-csv-quit
    # Based on `random_1nm_large_approx` - (1, 1, 1)
    python3 main.py sim_data r_1_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 24 " -1e-9" 1e-9 \
        --save-aberrations-csv-quit
    # Based on `random_2nm_large_approx` - (2, 2, 2)
    python3 main.py sim_data r_2_coeffs_100k no_prop 0 \
        --rand-amount-per-zernike 100000 2 24 " -2e-9" 2e-9 \
        --save-aberrations-csv-quit

    # ==== CNN Fixed Data ====

    # ---- Capture CNN Datasets ----
    # Based on `fixed_50nm_range_2000_approx`
    # Training - [-50, 50] with 2000 points (46000 total)
    python3 main.py sim_data f_50_2000_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 2000 \
        --save-aberrations-csv-quit
    # Testing - [-50, 50] with 21 points (483 total)
    python3 main.py sim_data f_50_21_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21 \
        --save-aberrations-csv-quit

    # ---- Stabilization CNN Datasets ----
    # Based on `fixed_1nm_range_301_approx`
    # Training - [-1, 1] with 301 points (6923 total)
    python3 main.py sim_data f_1_301_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 301 \
        --save-aberrations-csv-quit
    # Testing - [-1, 1] with 21 points (483 total)
    python3 main.py sim_data f_1_21_coeffs no_prop 0 \
        --fixed-amount-per-zernike-range 2 24 " -1e-9" 1e-9 21 \
        --save-aberrations-csv-quit

SEC2 - INPUT ABERRATION CSV FILE PREP ++++++++++++++++++++++++++++++++++++++++++
The CSV files need to be moved around for easier access. The below commands must
be run from within the `data/raw/` directory.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    mkdir picd_coeffs_v7

    mv f_pm_40_coeffs/aberrations.csv picd_coeffs_v7/f_pm_40_coeffs.csv
    mv r_10_2_1_coeffs_100k/aberrations.csv picd_coeffs_v7/r_10_2_1_coeffs_100k.csv
    mv r_2_coeffs_50k/aberrations.csv picd_coeffs_v7/r_2_coeffs_50k.csv
    mv r_500_20_10_coeffs_100k/aberrations.csv picd_coeffs_v7/r_500_20_10_coeffs_100k.csv
    mv r_50_10_5_coeffs_100k/aberrations.csv picd_coeffs_v7/r_50_10_5_coeffs_100k.csv
    mv r_15_5_2_coeffs_100k/aberrations.csv picd_coeffs_v7/r_15_5_2_coeffs_100k.csv
    mv r_10_coeffs_100k/aberrations.csv picd_coeffs_v7/r_10_coeffs_100k.csv
    mv r_25_1_half_coeffs_100k/aberrations.csv picd_coeffs_v7/r_25_1_half_coeffs_100k.csv
    mv r_15_2_1_coeffs_100k/aberrations.csv picd_coeffs_v7/r_15_2_1_coeffs_100k.csv
    mv r_15_1_half_coeffs_100k/aberrations.csv picd_coeffs_v7/r_15_1_half_coeffs_100k.csv
    mv r_10_1_half_coeffs_100k/aberrations.csv picd_coeffs_v7/r_10_1_half_coeffs_100k.csv
    mv r_10_half_quarter_coeffs_100k/aberrations.csv picd_coeffs_v7/r_10_half_quarter_coeffs_100k.csv
    mv r_half_quarter_fifth_coeffs_100k/aberrations.csv picd_coeffs_v7/r_half_quarter_fifth_coeffs_100k.csv
    mv r_1_coeffs_100k/aberrations.csv picd_coeffs_v7/r_1_coeffs_100k.csv
    mv r_2_coeffs_100k/aberrations.csv picd_coeffs_v7/r_2_coeffs_100k.csv
    mv f_50_2000_coeffs/aberrations.csv picd_coeffs_v7/f_50_2000_coeffs.csv
    mv f_50_21_coeffs/aberrations.csv picd_coeffs_v7/f_50_21_coeffs.csv
    mv f_1_301_coeffs/aberrations.csv picd_coeffs_v7/f_1_301_coeffs.csv
    mv f_1_21_coeffs/aberrations.csv picd_coeffs_v7/f_1_21_coeffs.csv

    rm -rf f_pm_40_coeffs
    rm -rf r_10_2_1_coeffs_100k
    rm -rf r_2_coeffs_50k
    rm -rf r_500_20_10_coeffs_100k
    rm -rf r_50_10_5_coeffs_100k
    rm -rf r_15_5_2_coeffs_100k
    rm -rf r_10_coeffs_100k
    rm -rf r_25_1_half_coeffs_100k
    rm -rf r_15_2_1_coeffs_100k
    rm -rf r_15_1_half_coeffs_100k
    rm -rf r_10_1_half_coeffs_100k
    rm -rf r_10_half_quarter_coeffs_100k
    rm -rf r_half_quarter_fifth_coeffs_100k
    rm -rf r_1_coeffs_100k
    rm -rf r_2_coeffs_100k
    rm -rf f_50_2000_coeffs
    rm -rf f_50_21_coeffs
    rm -rf f_1_301_coeffs
    rm -rf f_1_21_coeffs

SEC3 - MOVE FITS FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The FITS datafiles containing instrument data should be moved to the `data/raw/`
directory with new folder and file names; this is where commands must be run.
The reference images are mixed in with the aberrated image data. The shared data
is duplicated across datasets to make preprocessing easier.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Copy all the FITS datafiles so they can be moved
    cp llowfs_training_03_26_2026/*.fits .

    # ---- RM ----
    mkdir inst_llowfs_v7_rm
    # f_pm_40_coeffs
    mv lyt_alp_train_lac_20260324_175028_caldata.fits       inst_llowfs_v7_rm/0_data.fits

    # ---- Capture Data ----
    mkdir inst_llowfs_v7_cap
    # r_10_2_1_coeffs_100k
    cp lyt_alp_train_lac_20260324_175919_caldata.fits       inst_llowfs_v7_cap/0_data.fits
    # r_500_20_10_coeffs_100k
    mv lyt_alp_train_lac_20260324_192925_caldata.fits       inst_llowfs_v7_cap/1_data.fits
    # r_50_10_5_coeffs_100k
    mv lyt_alp_train_lac_20260324_193759_caldata.fits       inst_llowfs_v7_cap/2_data.fits
    # r_15_5_2_coeffs_100k
    mv lyt_alp_train_lac_20260324_185044_caldata.fits       inst_llowfs_v7_cap/3_data.fits
    # r_10_coeffs_100k
    mv lyt_alp_train_lac_20260324_180754_caldata.fits       inst_llowfs_v7_cap/4_data.fits
    # f_50_2000_coeffs
    mv lyt_alp_train_lac_20260324_174150_caldata.fits       inst_llowfs_v7_cap/5_data.fits

    # ---- Stabilization Data ----
    mkdir inst_llowfs_v7_sta
    # r_10_2_1_coeffs_100k
    mv lyt_alp_train_lac_20260324_175919_caldata.fits       inst_llowfs_v7_sta/0_data.fits
    # r_25_1_half_coeffs_100k
    mv lyt_alp_train_lac_20260324_190752_caldata.fits       inst_llowfs_v7_sta/1_data.fits
    # r_15_2_1_coeffs_100k
    mv lyt_alp_train_lac_20260324_183337_caldata.fits       inst_llowfs_v7_sta/2_data.fits
    # r_15_1_half_coeffs_100k
    mv lyt_alp_train_lac_20260324_182502_caldata.fits       inst_llowfs_v7_sta/3_data.fits
    # r_10_1_half_coeffs_100k
    mv lyt_alp_train_lac_20260324_175045_caldata.fits       inst_llowfs_v7_sta/4_data.fits
    # r_10_half_quarter_coeffs_100k
    mv lyt_alp_train_lac_20260324_181628_caldata.fits       inst_llowfs_v7_sta/5_data.fits
    # r_half_quarter_fifth_coeffs_100k
    mv lyt_alp_train_lac_20260324_194634_caldata.fits       inst_llowfs_v7_sta/6_data.fits
    # r_1_coeffs_100k
    mv lyt_alp_train_lac_20260324_185918_caldata.fits       inst_llowfs_v7_sta/7_data.fits
    # r_2_coeffs_100k
    mv lyt_alp_train_lac_20260324_191627_caldata.fits       inst_llowfs_v7_sta/8_data.fits
    # f_1_301_coeffs
    mv lyt_alp_train_lac_20260324_174101_caldata.fits       inst_llowfs_v7_sta/9_data.fits

    # ---- Random 2nm Testing ----
    mkdir inst_llowfs_v7_tst_2nm_rnd
    # r_2_coeffs_50k
    mv lyt_alp_train_lac_20260324_192501_caldata.fits       inst_llowfs_v7_tst_2nm_rnd/0_data.fits

    # ---- Fixed [-50, 50] nm Testing ----
    mkdir inst_llowfs_v7_tst_50nm_fix
    # f_50_21_coeffs
    mv lyt_alp_train_lac_20260324_175011_caldata.fits       inst_llowfs_v7_tst_50nm_fix/0_data.fits

    # ---- Fixed [-1, 1] nm Testing ----
    mkdir inst_llowfs_v7_tst_1nm_fix
    # f_1_21_coeffs
    mv lyt_alp_train_lac_20260324_174044_caldata.fits       inst_llowfs_v7_tst_1nm_fix/0_data.fits

SEC4 - FITS TO HDF FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++
These FITS datafiles should be converted to HDF files. The format of the HDF
datafiles should be the same as the raw simulation datafiles. When converting
these datafiles, the duplicate rows are removed. The reference images are
followed by the aberrated images in each group.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- RM - 46 Rows ----
    # Taking out the last row in the data which gives the base field
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_rm_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_rm --take-every-n-rows 4 3 --slice-row-ranges 0 184
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_rm_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_rm --take-every-n-rows 4 1 --slice-row-ranges 0 184

    # ---- Capture Data - 546,000 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_cap_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_cap --take-every-n-rows 4 3
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_cap_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_cap --take-every-n-rows 4 1

    # ---- Stabilization Data - 906,923 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_sta_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_sta --take-every-n-rows 4 3
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_sta_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_sta --take-every-n-rows 4 1

    # ---- Random 2nm Testing - 50,000 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_2nm_rnd_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_2nm_rnd --take-every-n-rows 4 3
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_2nm_rnd_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_2nm_rnd --take-every-n-rows 4 1

    # ---- Fixed [-50, 50] nm Testing - 483 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_50nm_fix_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_50nm_fix --take-every-n-rows 4 3
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_50nm_fix_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_50nm_fix --take-every-n-rows 4 1

    # ---- Fixed [-1, 1] nm Testing - 483 Rows ----
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_1nm_fix_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_1nm_fix --take-every-n-rows 4 3
    python3 main.py convert_picd_instrument_data inst_llowfs_v7_tst_1nm_fix_bf_hdf 2 24 \
        --fits-data-tags inst_llowfs_v7_tst_1nm_fix --take-every-n-rows 4 1

SEC5 - PREPROCESS DATAFILES ++++++++++++++++++++++++++++++++++++++++++++++++++++
The newly converted HDF datafiles should be preprocessed.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- Training and Validation Datasets ----
    python3 main.py preprocess_data_complete \
        inst_llowfs_v7_cap_hdf \
        train_picd_data_v7_cap val_picd_data_v7_cap none 90 10 0 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --additional-raw-data-tags-train-only inst_llowfs_v7_cap_to_hdf \
        --use-field-diff inst_llowfs_v7_cap_bf_hdf \
        --use-field-diff-corresponding \
        --set-slices-train-only 500000 546000 \
        --fix-seed 314
    python3 main.py preprocess_data_complete \
        inst_llowfs_v7_sta_hdf \
        train_picd_data_v7_sta val_picd_data_v7_sta none 90 10 0 \
        --disable-norm-inputs --inputs-sum-to-one \
        --norm-outputs individually --norm-range-ones \
        --additional-raw-data-tags-train-only inst_llowfs_v7_sta_to_hdf \
        --use-field-diff inst_llowfs_v7_sta_bf_hdf \
        --use-field-diff-corresponding \
        --set-slices-train-only 900000 906923 \
        --fix-seed 314

    # ---- Testing Datasets ----
    python3 main.py preprocess_data_bare inst_llowfs_v7_tst_2nm_rnd_hdf \
        inst_llowfs_v7_tst_2nm_rnd_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v7_tst_50nm_fix_hdf \
        inst_llowfs_v7_tst_50nm_fix_hdf_proc
    python3 main.py preprocess_data_bare inst_llowfs_v7_tst_1nm_fix_hdf \
        inst_llowfs_v7_tst_1nm_fix_hdf_proc

SEC6 - CNN TRAINING AND TESTING ++++++++++++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN models created from the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # ---- Capture CNN Training ----
    python3 main_scnp.py model_train instrument_llowfs_capture_v7 \
        train_picd_data_v7_cap val_picd_data_v7_cap \
        llowfs_cnn_4_no_dropout mae adamw 1e-3 400 --batch-size 512 \
        --use-cosine-annealing-lr-scheduler 30 1e-6 5e-8 --clip-gradient-norm 10 \
        --overwrite-existing --only-best-epoch --disable-tag-lookup --fix-seed 314

    # ---- Capture CNN Testing ----
    python3 main.py model_test instrument_llowfs_capture_v7 last \
        inst_llowfs_v7_tst_2nm_rnd_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 1 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_2nm_rnd_bf_hdf
    python3 main.py model_test instrument_llowfs_capture_v7 last \
        inst_llowfs_v7_tst_50nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 1 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_50nm_fix_bf_hdf
    python3 main.py model_test instrument_llowfs_capture_v7 last \
        inst_llowfs_v7_tst_1nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 1 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_1nm_fix_bf_hdf

    # ---- Stabilization CNN Training ----
    python3 main_scnp.py model_train instrument_llowfs_stabilization_v7 \
        train_picd_data_v7_sta val_picd_data_v7_sta \
        llowfs_cnn_4_no_dropout mae adamw 1e-3 400 --batch-size 512 \
        --use-cosine-annealing-lr-scheduler 30 1e-6 5e-8 --clip-gradient-norm 10 \
        --overwrite-existing --only-best-epoch --disable-tag-lookup --fix-seed 314

    # ---- Stabilization CNN Testing ----
    python3 main.py model_test instrument_llowfs_stabilization_v7 last \
        inst_llowfs_v7_tst_2nm_rnd_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_2nm_rnd_bf_hdf
    python3 main.py model_test instrument_llowfs_stabilization_v7 last \
        inst_llowfs_v7_tst_50nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_50nm_fix_bf_hdf
    python3 main.py model_test instrument_llowfs_stabilization_v7 last \
        inst_llowfs_v7_tst_1nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 2 --inputs-need-norm --inputs-need-diff \
        --change-base-field-corresponding inst_llowfs_v7_tst_1nm_fix_bf_hdf

SEC7 - RM CREATION AND TESTING +++++++++++++++++++++++++++++++++++++++++++++++++
Create and test the RM model on the instrument data.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py create_response_matrix \
        --simulated-data-tag-average inst_llowfs_v7_rm_hdf --wfs-sum-to-one
        --base-field-tag inst_llowfs_v7_rm_bf_hdf --base-field-mapping-corresponding
    python3 main.py run_response_matrix inst_llowfs_v7_rm_hdf \
        inst_llowfs_v7_tst_2nm_rnd_hdf_proc \
        --scatter-plot 4 6 2 1e-7 15 --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field-corresponding inst_llowfs_v7_tst_2nm_rnd_bf_hdf
    python3 main.py run_response_matrix inst_llowfs_v7_rm_hdf \
        inst_llowfs_v7_tst_50nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field-corresponding inst_llowfs_v7_tst_50nm_fix_bf_hdf
    python3 main.py run_response_matrix inst_llowfs_v7_rm_hdf \
        inst_llowfs_v7_tst_1nm_fix_hdf_proc \
        --zernike-plots --enable-paper-plots 0 --wfs-need-sum-to-one \
        --change-base-field-corresponding inst_llowfs_v7_tst_1nm_fix_bf_hdf
