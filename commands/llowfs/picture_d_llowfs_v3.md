....................................................................................................
NOTES AND CHANGES COMPARED TO `picture_d_llowfs.md`:
- All the results in this file are based on simulations done in the `piccsim` library. The reason
  for this is that data obtained on the instrument drifts for Z2 and Z3, so to prevent this
  simulations are done instead. The simulations should align very well to the actual instrument.
- Updated the normalization so that the sum of all the input pixels are equal to one. This is
  instead of scaling the input values between [-1, 1] based on the global min and max.
- The PICTURE-D flight computer has to use the older flight stack developed for PICTURE-C instead of
  the newer one. Due to this, there is much less computational ability, so the previous CNN
  developed, [V55b], will be to slow. Therefore, CNN model [V55a] will be used instead.
....................................................................................................

These are the commands needed to obtain data in `piccsim` and to create the associated RM and CNN.
Most of the commands are based on the ones used to simulate data in this repo; the commands are
taken/referenced from the `general.md` and `model_training_versions.txt` files.
The RM is for ±40 nm RMS error (technically ±39.995 nm RMS error).
The CNN is based off the [V55a] `sum1_scaling_faster_model` model.

TABLE OF CONTENTS:
    SEC1 - SIMULATED DATA
    SEC2 - FITS TO HDF FILES
    SEC3 - PREPROCESS DATAFILES
    SEC4 - [V55a] CNN TRAINING AND TESTING
    SEC5 - RM CREATION AND TESTING
    SEC6 - (EXTRA) [V55b] CNN TRAINING AND TESTING

SEC1 - SIMULATED DATA ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
The data is simulated in `piccsim` (IDL) using the `batch.llowfsnn.pro` script.
Each simulation will be output in its own FITS datafile.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # All coefficients are in nm RMS of wavefront error.
    # `BASED ON` refers to the corresponding dataset simulated using this repo.
    # Number 10 is a testing dataset; Number 11 is the base field.

    # IDX    N_ROWS   Z2-3    Z4-8    Z9-24    BASED ON
    # 1      100000    500      20       10    random_group_500_20_10
    # 2      100000     50      10        5    random_group_50_10_5
    # 3      100000     15       5        2    random_group_15_5_2
    # 4      100000     15       1      1/2    random_group_15_1_half
    # 5      100000     10       2        1    random_group_10_2_1
    # 6      100000    1/2     1/4      1/5    random_group_half_quarter_fifth
    # 7      100000      2       2        2    random_2nm_large_approx
    # 8      100000     10      10       10    random_10nm_large_approx
    # 9       46000  {±50 each Z, 2000 steps}  fixed_50nm_range_2000_approx
    # 10      25000     10      10       10    random_10nm_med_approx
    # 11          0      0       0        0    no_aberrations_approx

    # The `piccsim/batch.llowfsnn.pro` script was used for simulations.
    # Each chunk of data was simulated using its own core on the workstation.
    # For some reason, as each process ran, the read/write speeds slowly
    # decreased which led to simulations taking way longer than expected.
    # In order to fix this, the `piccsim/output/rx_picture_d_llowfs/*`
    # directories were each put into memory - this means all the temporary
    # files written/deleted for each simulation are stored in memory instead
    # of being written out to the SSD. The command to do this for a directory is
    #   sudo mount -t tmpfs -o size=1G,rw,nodev,nosuid,uid=$(id -u),gid=$(id -g),mode=1700,noatime,nodiratime tmpfs PATH_TO_DIR

SEC2 - FITS TO HDF FILES +++++++++++++++++++++++++++++++++++++++++++++++++++++++
These FITS datafiles should be converted to HDF files. The format of the HDF
datafiles should be the same as the raw simulation datafiles. On the workstation
these FITS datafile directories are located under:
    `/home/michael-jones/Documents/piccsim_sim_data/`.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_1 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_1 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_2 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_2 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_3 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_3 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_4 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_4 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_5 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_5 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_6 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_6 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_7 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_7 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_8 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_8 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_9 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_9 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --rows-per-chunk 10000
    # Slice out the rows for the RM
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_9b \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_9 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24 \
        --slice-row-ranges 4600 4623 41377 41400
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_10 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_10 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24
    python3 main.py convert_piccsim_fits_data piccsim_llowfs_v1_11 \
        /home/michael-jones/Documents/piccsim_sim_data/llowfsnn_11 \
        --fits-file-globs 'dm_*' 'intensity_*' \
        --fits-table-names zernike_coeffs ccd_intensity \
        --add-dummy-tables ccd_sampling --add-zernikes 2 24

SEC3 - PREPROCESS DATAFILES ++++++++++++++++++++++++++++++++++++++++++++++++++++
The newly converted HDF datafiles should be preprocessed.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Preprocess the data for training, validation, and testing
    python3 main.py preprocess_data_complete \
        piccsim_llowfs_v1_1 \
        train_picd_data_v3 val_picd_data_v3 test_picd_data_v3 \
        80 15 5 \
        --disable-norm-inputs --inputs-sum-to-one --norm-range-ones \
        --keep-only-unique-rows --norm-outputs individually \
        --outputs-in-surface-error --outputs-scaling-factor 1e-9 \
        --additional-raw-data-tags piccsim_llowfs_v1_2 piccsim_llowfs_v1_3 \
               piccsim_llowfs_v1_4 piccsim_llowfs_v1_5 piccsim_llowfs_v1_6 \
               piccsim_llowfs_v1_7 piccsim_llowfs_v1_8 \
        --additional-raw-data-tags-train-only piccsim_llowfs_v1_9 \
        --use-field-diff piccsim_llowfs_v1_11

    # Preprocess the testing data
    python3 main.py preprocess_data_bare piccsim_llowfs_v1_9 \
        piccsim_llowfs_v1_9_processed \
        --outputs-in-surface-error --outputs-scaling-factor 1e-9
    python3 main.py preprocess_data_bare piccsim_llowfs_v1_10 \
        piccsim_llowfs_v1_10_processed \
        --outputs-in-surface-error --outputs-scaling-factor 1e-9

SEC4 - [V55a] CNN TRAINING AND TESTING +++++++++++++++++++++++++++++++++++++++++
Train, test, and export the CNN model.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main_scnp.py model_train picd_cnn_v3_round_one \
        train_picd_data_v3 val_picd_data_v3 \
        speedup_6 mae adam 1e-5 1000 --batch-size 256 \
        --overwrite-existing --only-best-epoch --early-stopping 15
    python3 main_scnp.py model_train picd_cnn_v3 \
        train_picd_data_v3 val_picd_data_v3 \
        speedup_6 mae adam 1e-8 200 --batch-size 128 \
        --overwrite-existing --only-best-epoch --early-stopping 15 \
        --init-weights picd_cnn_v3_round_one last

    python3 main.py model_test picd_cnn_v3 last \
        test_picd_data_v3 --scatter-plot 4 6 2 0 15
    python3 main.py model_test picd_cnn_v3 last \
        piccsim_llowfs_v1_9_processed \
        --zernike-plots --inputs-need-norm --inputs-need-diff
    python3 main.py model_test picd_cnn_v3 last \
        piccsim_llowfs_v1_10_processed \
        --scatter-plot 4 6 2 1e-7 15 --inputs-need-norm --inputs-need-diff

    # Export the model so that it can be used in the `pytorch_model_in_c` repo
    # via ONNX runtime.
    python3 main.py export_model picd_cnn_v3 last val_picd_data_v3 --benchmark 5000

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

SEC5 - RM CREATION AND TESTING +++++++++++++++++++++++++++++++++++++++++++++++++
Create and test the RM model.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main.py create_response_matrix \
        --simulated-data-tag-average piccsim_llowfs_v1_9b \
        --base-field-tag piccsim_llowfs_v1_11 \
        --outputs-in-surface-error --outputs-scaling-factor 1e-9

    python3 main.py run_response_matrix piccsim_llowfs_v1_9b \
        piccsim_llowfs_v1_9_processed --zernike-plots
    python3 main.py run_response_matrix piccsim_llowfs_v1_9b \
        piccsim_llowfs_v1_10_processed --scatter-plot 4 6 2 1e-8 15

SEC6 - (EXTRA) [V55b] CNN TRAINING AND TESTING +++++++++++++++++++++++++++++++++
The commands to train and test the [V55b] CNN model.
This CNN is too slow to run on the older, slower flight computer.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    python3 main_scnp.py model_train picd_cnn_v3_v55b_round_one \
        train_picd_data_v3 val_picd_data_v3 \
        best32_1_smaller_10 mae adam 1e-5 1000 --batch-size 256 \
        --overwrite-existing --only-best-epoch --early-stopping 15
    python3 main_scnp.py model_train picd_cnn_v3_v55b \
        train_picd_data_v3 val_picd_data_v3 \
        best32_1_smaller_10 mae adam 1e-8 200 --batch-size 128 \
        --overwrite-existing --only-best-epoch --early-stopping 15 \
        --init-weights picd_cnn_v3_v55b_round_one last

    python3 main.py model_test picd_cnn_v3_v55b last \
        test_picd_data_v3 --scatter-plot 4 6 2 0 15
    python3 main.py model_test picd_cnn_v3_v55b last \
        piccsim_llowfs_v1_9_processed \
        --zernike-plots --inputs-need-norm --inputs-need-diff
    python3 main.py model_test picd_cnn_v3_v55b last \
        piccsim_llowfs_v1_10_processed \
        --scatter-plot 4 6 2 1e-7 15 --inputs-need-norm --inputs-need-diff
