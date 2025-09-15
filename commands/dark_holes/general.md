# Commands Lookup

## Data Conversion

Convert FITS datafiles from the `piccsim` library to HDF.

Science camera masks for the dark hole:

    python3 main.py convert_piccsim_fits_data darkhole_mask \
        /home/picture/code/picture/piccsim/plots/rx_picture_d_efcnn/sim_system_dmcmd \
        --sci-cam-mask-file sim_system_dmcmd_rx_picture_d_efcnn_pol0_sci_dhmask.fits

    # ==============================
    # Create the half dark hole mask
    # ==============================
    # Start in the `data/raw/` folder
    cp -R darkhole_mask darkhole_mask_half
    cd darkhole_mask_half
    python3
        from h5py import File
        mask = File('0_data.h5')['dark_zone_mask'][:]
        mask[:, :51] = 0
        with File('1_data.h5', 'w') as file:
            file['dark_zone_mask'] = mask
        exit()
    rm 0_data.h5; mv 1_data.h5 0_data.h5


Perfect dark hole (no aberrations or DM commands):

    python3 main.py convert_piccsim_fits_data dh_perfect \
        /home/picture/code/picture/piccsim/all_sim_data/dmcmd_perfect \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r

First HODM only:

    # 20nm pokes on every actuator on HODM 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_pokes \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_single_actuator_pokes_20nm \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r

    # 1nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_1nm_100k \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_1nm_100k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000

    # 20nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch1 \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch2 \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch3 \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch3 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch4 \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch4 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    # Same as the above, just less data to make model prototyping quicker and easier
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_100k_train_and_val \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_20k_test \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 20000

Both HODMs:

    # 20nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_84k \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_84k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 84000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_100k_ch1 \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_100k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_100k_ch2 \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_100k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_200k_ch1 \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_200k_ch2 \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000

The DM SVD modes from the inverted matrix:

    python3 main.py convert_piccsim_fits_data hodm1_756_modes \
        /home/picture/code/picture/piccsim/output/svd_modes/rx_picture_d_efcnn_dm1 \
        --fits-file-globs 'rx_picture_d_efcnn_sci_dm1_mode_*' \
        --fits-table-names dm1_modes
    python3 main.py convert_piccsim_fits_data hodm2_756_modes \
        /home/picture/code/picture/piccsim/output/svd_modes/rx_picture_d_efcnn_dm2 \
        --fits-file-globs 'rx_picture_d_efcnn_sci_dm2_mode_*' \
        --fits-table-names dm2_modes

## Data Preprocessing

Preprocess the datasets:

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_20nm val_dh_20nm test_dh_20nm 70 15 15 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_100k_ch1 \
        train_dh_double_prototype_756_norm val_dh_double_prototype_756_norm test_dh_double_prototype_756_norm 80 10 10 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_20nm_xl val_dh_20nm_xl test_dh_20nm_xl 70 15 15 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_20nm_100k_ch1 dh_both_hodms_20nm_100k_ch2

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_both_svd_300_norm val_dh_both_svd_300_norm test_dh_both_svd_300_norm 84 8 8 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_20nm_100k_ch1 dh_both_hodms_20nm_100k_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_both_norm_xl val_dh_both_norm_xl test_dh_both_norm_xl 84 8 8 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_20nm_100k_ch1 dh_both_hodms_20nm_100k_ch2 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_200k_ch1 \
        train_dh_both_norm_xl val_dh_both_norm_xl test_dh_both_norm_xl 84 8 8 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_20nm_200k_ch2 \
        --norm-inputs --norm-outputs --extend-existing-preprocessed-data

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_xl val_dh_single_xl test_dh_single_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_3ch_xl val_dh_single_3ch_xl test_dh_single_3ch_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --add-total-intensity

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_wp_xl val_dh_single_wp_xl test_dh_single_wp_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --additional-raw-data-tags-train-only dh_first_hodm_20nm_pokes

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_svd_xl val_dh_single_svd_xl test_dh_single_svd_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 756

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_svd_300_xl val_dh_single_svd_300_xl test_dh_single_svd_300_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_svd_300_norm_xl val_dh_single_svd_300_norm_xl test_dh_single_svd_300_norm_xl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_first_hodm_1nm_100k \
        train_dh_single_svd_300_norm_xxl val_dh_single_svd_300_norm_xxl test_dh_single_svd_300_norm_xxl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch1 dh_first_hodm_20nm_200k_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_200k_ch3 \
        train_dh_single_svd_300_norm_xxl val_dh_single_svd_300_norm_xxl test_dh_single_svd_300_norm_xxl 84 8 8 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --additional-raw-data-tags dh_first_hodm_20nm_200k_ch4 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --norm-inputs --norm-outputs --extend-existing-preprocessed-data

    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_100k_train_and_val \
        train_dh_single_prototype_svd_300_norm val_dh_single_prototype_svd_300_norm empty 90 10 0 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_20k_test \
        empty empty test_dh_single_prototype_svd_300_norm 0 0 100 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --disable-shuffle

    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_100k_train_and_val \
        train_dh_single_prototype_svd_300_norm_3ch val_dh_single_prototype_svd_300_norm_3ch empty 90 10 0 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --norm-inputs --norm-outputs --add-total-intensity
    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_20k_test \
        empty empty test_dh_single_prototype_svd_300_norm_3ch 0 0 100 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 \
        --disable-shuffle --add-total-intensity

    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_100k_train_and_val \
        train_dh_single_prototype_svd_756_norm val_dh_single_prototype_svd_756_norm empty 90 10 0 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 756 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_20k_test \
        empty empty test_dh_single_prototype_svd_756_norm 0 0 100 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 756 \
        --disable-shuffle

    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_100k_train_and_val \
        train_dh_single_prototype_756_norm val_dh_single_prototype_756_norm empty 90 10 0 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_first_hodm_20nm_20k_test \
        empty empty test_dh_single_prototype_756_norm 0 0 100 \
        --dm-tables dm1 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask_half --remove-dark-zone-padding \
        --disable-shuffle

## Analysis Conversion

Analysis results for models trained on SVD basis outputs can be converted to actuator heights:

    # Half dark hole; one HODM
    python3 main.py convert_analysis_outputs_from_svd_basis \
        dh_v9_2 last test_dh_single_prototype_svd_300_norm \
        --svd-modes-tags hodm1_756_modes \
        --svd-modes-table-names dm1_modes \
        --svd-modes-count 300
    # Full dark hole; two HODMs
    python3 main.py convert_analysis_outputs_from_svd_basis \
        dh_v13 last test_dh_both_svd_300_norm \
        --svd-modes-tags hodm1_756_modes hodm2_756_modes \
        --svd-modes-table-names dm1_modes dm2_modes \
        --svd-modes-count 300
