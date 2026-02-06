# Commands Lookup

## Data Conversion

Convert FITS datafiles from the `piccsim` library to HDF.

Science camera masks for the dark hole:

    python3 main.py convert_piccsim_fits_data darkhole_mask \
        /home/michael-jones/Documents/piccsim/plots/rx_picture_d_efcnn/sim_system_dmcmd \
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
        /home/michael-jones/Documents/piccsim_sim_data/dmcmd_perfect \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r

First HODM only:

    # 20nm pokes on every actuator on HODM 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_pokes \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_single_actuator_pokes_20nm \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r

    # 1nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_1nm_100k \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_1nm_100k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000

    # 20nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch3 \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch3 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_200k_ch4 \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch4 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000
    # Same as the above, just less data to make model prototyping quicker and easier
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_100k_train_and_val \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_first_hodm_20nm_20k_test \
        /home/michael-jones/Documents/piccsim_sim_data/first_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 20000

Both HODMs:

    # 20nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_84k \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_20nm_84k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 84000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_100k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_20nm_100k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_100k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_20nm_100k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_200k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_20nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_200k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_20nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000

    # 10nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_10nm_200k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_10nm_200k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 25000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_10nm_200k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_10nm_200k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 25000

    # 1nm * Gaussian between -1 and 1
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_1nm_37k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_1nm_37k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 20000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_1nm_37k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_1nm_37k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 20000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_1nm_42k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_1nm_42k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 20000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_both_pol_1nm_42k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_both_pol_1nm_42k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i_pol0' 'sci*r_pol0' 'sci_*i_pol1' 'sci*r_pol1' \
        --fits-table-names dm1 dm2 sci_i_pol0 sci_r_pol0 sci_i_pol1 sci_r_pol1 \
        --rows-per-chunk 20000

    # 12 iterations of EFC using the RM; saves the inital EF and the final DH DM command
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_final_dh_14k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_efc_final_dh_14k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_final_dh_14k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_efc_final_dh_14k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_final_dh_9k_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_efc_final_dh_9k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_final_dh_9k_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/both_hodm_efc_final_dh_9k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r

    # 12 iterations of EFC using the RM; saves first 8 iterations with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_8iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 8 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_8iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 8 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first 6 iterations with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_6iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 6 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_6iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 6 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first 4 iterations with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_4iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 4 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_4iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 4 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first 3 iterations with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_3iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 3 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_3iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 3 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first iteration with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_1iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 1 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_30k_1iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_30k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 1 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first 6 iterations with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_27k_6iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_27k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 6 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_27k_6iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_27k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 6 dm1 dm2 \
        --rows-per-chunk 24000

    # 12 iterations of EFC using the RM; saves first iteration with correct DM differentials
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_27k_1iter_ch1 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_27k_ch1 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 1 dm1 dm2 \
        --rows-per-chunk 24000
    python3 main.py convert_piccsim_fits_data dh_both_hodms_efc_27k_1iter_ch2 \
        /home/michael-jones/Documents/piccsim_sim_data/dh_both_hodms_efc_27k_ch2 \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --save-difference-only 12 11 1 dm1 dm2 \
        --rows-per-chunk 24000

The DM SVD modes from the inverted matrix:

    python3 main.py convert_piccsim_fits_data hodm1_756_modes \
        /home/michael-jones/Documents/piccsim/output/svd_modes/rx_picture_d_efcnn_dm1 \
        --fits-file-globs 'rx_picture_d_efcnn_sci_dm1_mode_*' \
        --fits-table-names dm1_modes
    python3 main.py convert_piccsim_fits_data hodm2_756_modes \
        /home/michael-jones/Documents/piccsim/output/svd_modes/rx_picture_d_efcnn_dm2 \
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

    python3 main.py preprocess_data_dark_hole dh_both_hodms_both_pol_1nm_37k_ch1  \
        train_dh_both_1nm_norm_sm val_dh_both_1nm_norm_sm test_dh_both_1nm_norm_sm 84 8 8 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r_pol0 sci_i_pol0 \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_both_pol_1nm_37k_ch2 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_both_pol_1nm_37k_ch1  \
        train_dh_both_1nm_10nm_norm_lg val_dh_both_1nm_10nm_norm_lg test_dh_both_1nm_10nm_norm_lg 84 8 8 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r_pol0 sci_i_pol0 \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_both_pol_1nm_37k_ch2 \
            dh_both_hodms_both_pol_1nm_42k_ch1 dh_both_hodms_both_pol_1nm_42k_ch2 \
            dh_both_hodms_both_pol_10nm_200k_ch1 \
        --norm-inputs --norm-outputs

    # Realized these datasets were using `both_hodm_efc_final_dh_14k_ch2` so the data was not being loaded in
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1   \
        train_dh_both_hodms_efc_final_dh_sm val_dh_both_hodms_efc_final_dh_sm test_dh_both_hodms_efc_final_dh_sm 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags both_hodm_efc_final_dh_14k_ch2  \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_sm_svd_300 val_dh_both_hodms_efc_final_dh_sm_svd_300 test_dh_both_hodms_efc_final_dh_sm_svd_300 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --additional-raw-data-tags both_hodm_efc_final_dh_14k_ch2  \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_med val_dh_both_hodms_efc_final_dh_med test_dh_both_hodms_efc_final_dh_med 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_xlg_6iter val_dh_both_hodms_efc_final_dh_xlg_6iter \
        test_dh_both_hodms_efc_final_dh_xlg_6iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_27k_6iter_ch1 \
        train_dh_both_hodms_efc_final_dh_xlg_6iter val_dh_both_hodms_efc_final_dh_xlg_6iter \
        test_dh_both_hodms_efc_final_dh_xlg_6iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_27k_6iter_ch2 \
        --extend-existing-preprocessed-data --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm val_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm \
        test_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs-globally
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_27k_6iter_ch1 \
        train_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm val_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm \
        test_dh_both_hodms_efc_final_dh_xlg_6iter_global_norm 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_27k_6iter_ch2 \
        --extend-existing-preprocessed-data --norm-inputs --norm-outputs-globally

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter val_dh_both_hodms_efc_final_dh_lg_6iter \
        test_dh_both_hodms_efc_final_dh_lg_6iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_global_out val_dh_both_hodms_efc_final_dh_lg_6iter_global_out \
        test_dh_both_hodms_efc_final_dh_lg_6iter_global_out 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs-globally

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_30k_6iter_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_log10 val_dh_both_hodms_efc_final_dh_lg_6iter_log10 \
        test_dh_both_hodms_efc_final_dh_lg_6iter_log10 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs-globally --input-modified-log --output-modified-log
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_log10 val_dh_both_hodms_efc_final_dh_lg_6iter_log10 \
        test_dh_both_hodms_efc_final_dh_lg_6iter_log10 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
        --norm-inputs --norm-outputs-globally --input-modified-log --output-modified-log \
        --extend-existing-preprocessed-data

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_30k_6iter_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_log10_out val_dh_both_hodms_efc_final_dh_lg_6iter_log10_out \
        test_dh_both_hodms_efc_final_dh_lg_6iter_log10_out 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs-globally --output-modified-log
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_log10_out val_dh_both_hodms_efc_final_dh_lg_6iter_log10_out \
        test_dh_both_hodms_efc_final_dh_lg_6iter_log10_out 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
        --norm-inputs --norm-outputs-globally --output-modified-log \
        --extend-existing-preprocessed-data

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_new_norm val_dh_both_hodms_efc_final_dh_lg_6iter_new_norm \
        test_dh_both_hodms_efc_final_dh_lg_6iter_new_norm 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs-ones --norm-outputs-globally

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_2d_local val_dh_both_hodms_efc_final_dh_lg_6iter_2d_local \
        test_dh_both_hodms_efc_final_dh_lg_6iter_2d_local 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs-ones --norm-outputs --do-not-flatten-output

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_2d_global val_dh_both_hodms_efc_final_dh_lg_6iter_2d_global \
        test_dh_both_hodms_efc_final_dh_lg_6iter_2d_global 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs-ones --norm-outputs-globally --do-not-flatten-output

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_2d_global_v2 val_dh_both_hodms_efc_final_dh_lg_6iter_2d_global_v2 \
        test_dh_both_hodms_efc_final_dh_lg_6iter_2d_global_v2 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --norm-inputs --norm-outputs-globally --do-not-flatten-output

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_half val_dh_both_hodms_efc_final_dh_lg_6iter_half \
        test_dh_both_hodms_efc_final_dh_lg_6iter_half 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_30k_6iter_ch1 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_fl val_dh_both_hodms_efc_final_dh_lg_6iter_fl \
        test_dh_both_hodms_efc_final_dh_lg_6iter_fl 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --flatten-input --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_6iter_fl_no_norm \
        val_dh_both_hodms_efc_final_dh_lg_6iter_fl_no_norm \
        test_dh_both_hodms_efc_final_dh_lg_6iter_fl_no_norm 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_6iter_ch1 dh_both_hodms_efc_30k_6iter_ch2 \
        --flatten-input

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_8iter val_dh_both_hodms_efc_final_dh_lg_8iter \
        test_dh_both_hodms_efc_final_dh_lg_8iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_8iter_ch1 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_30k_8iter_ch2 \
        train_dh_both_hodms_efc_final_dh_lg_8iter val_dh_both_hodms_efc_final_dh_lg_8iter \
        test_dh_both_hodms_efc_final_dh_lg_8iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --norm-inputs --norm-outputs --extend-existing-preprocessed-data

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_8iter_svd val_dh_both_hodms_efc_final_dh_lg_8iter_svd \
        test_dh_both_hodms_efc_final_dh_lg_8iter_svd 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_8iter_ch1 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --norm-inputs --norm-outputs
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_30k_8iter_ch2  \
        train_dh_both_hodms_efc_final_dh_lg_8iter_svd val_dh_both_hodms_efc_final_dh_lg_8iter_svd \
        test_dh_both_hodms_efc_final_dh_lg_8iter_svd 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_3iter val_dh_both_hodms_efc_final_dh_lg_3iter \
        test_dh_both_hodms_efc_final_dh_lg_3iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_3iter_ch1 dh_both_hodms_efc_30k_3iter_ch2 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_1iter val_dh_both_hodms_efc_final_dh_lg_1iter \
        test_dh_both_hodms_efc_final_dh_lg_1iter 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_1iter_ch1 dh_both_hodms_efc_30k_1iter_ch2 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_1iter_svd val_dh_both_hodms_efc_final_dh_lg_1iter_svd \
        test_dh_both_hodms_efc_final_dh_lg_1iter_svd 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_1iter_ch1 dh_both_hodms_efc_30k_1iter_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_xlg_1iter_svd \
        val_dh_both_hodms_efc_final_dh_xlg_1iter_svd \
        test_dh_both_hodms_efc_final_dh_xlg_1iter_svd 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_1iter_ch1 dh_both_hodms_efc_30k_1iter_ch2 \
            dh_both_hodms_efc_27k_1iter_ch1 dh_both_hodms_efc_27k_1iter_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 300 dm2 hodm2_756_modes dm2_modes 300 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_1iter_svd_500 \
        val_dh_both_hodms_efc_final_dh_lg_1iter_svd_500 \
        test_dh_both_hodms_efc_final_dh_lg_1iter_svd_500 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_1iter_ch1 dh_both_hodms_efc_30k_1iter_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 500 dm2 hodm2_756_modes dm2_modes 500 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_1iter_svd_all \
        val_dh_both_hodms_efc_final_dh_lg_1iter_svd_all \
        test_dh_both_hodms_efc_final_dh_lg_1iter_svd_all 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_1iter_ch1 dh_both_hodms_efc_30k_1iter_ch2 \
        --use-dm-svd-basis dm1 hodm1_756_modes dm1_modes 756 dm2 hodm2_756_modes dm2_modes 756 \
        --norm-inputs --norm-outputs

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_4iter_residuals \
        val_dh_both_hodms_efc_final_dh_lg_4iter_residuals \
        test_dh_both_hodms_efc_final_dh_lg_4iter_residuals 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_4iter_ch1 dh_both_hodms_efc_30k_4iter_ch2 \
        --norm-inputs --norm-outputs --use-rm-residuals dh_dm1_dm2

    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_final_dh_14k_ch1 \
        train_dh_both_hodms_efc_final_dh_lg_8iter_residuals \
        val_dh_both_hodms_efc_final_dh_lg_8iter_residuals \
        test_dh_both_hodms_efc_final_dh_lg_8iter_residuals 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_efc_final_dh_14k_ch2 \
            dh_both_hodms_efc_final_dh_9k_ch1 dh_both_hodms_efc_final_dh_9k_ch2 \
            dh_both_hodms_efc_30k_8iter_ch1 \
        --norm-inputs --norm-outputs --use-rm-residuals dh_dm1_dm2
    python3 main.py preprocess_data_dark_hole dh_both_hodms_efc_30k_8iter_ch2 \
        train_dh_both_hodms_efc_final_dh_lg_8iter_residuals \
        val_dh_both_hodms_efc_final_dh_lg_8iter_residuals \
        test_dh_both_hodms_efc_final_dh_lg_8iter_residuals 88 6 6 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --norm-inputs --norm-outputs --use-rm-residuals dh_dm1_dm2 \
        --extend-existing-preprocessed-data

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

## Random Commands

Export a model so it can be run in `piccsim`:

    python3 main.py export_model dh_v29_1 last val_dh_both_hodms_efc_final_dh_lg_6iter \
        --benchmark 50 --no-base-field --no-save-txt-files

Run EF reconstruction:

    python3 piccsim_pipelines/sim_dh_ef.py dh_v25_6 150 val_dh_both_hodms_efc_final_dh_lg_6iter --full-dh

Plot the DM comparison:

    python3 main.py dm_comparison dh_v36 127 \
        test_dh_both_hodms_efc_final_dh_xlg_6iter 10 --dm-size 34

## Response Matrix

Convert a `piccsim` RM to HDF:

    python3 main.py convert_dh_rm \
        dh_dm1_dm2 /home/michael-jones/Documents/piccsim/output/rx_picture_d_efcnn_sim_system_dm1_dm2_RM.fits

Run a RM:

    python3 main.py run_response_matrix dh_dm1_dm2 \
        test_dh_both_hodms_efc_final_dh_xlg_6iter --inputs-need-denorm --dh-rm

Plot the DM comparison:

    python3 main.py dm_comparison dh_dm1_dm2 0 \
        test_dh_both_hodms_efc_final_dh_xlg_6iter 10 --dm-size 34 --rm-not-nn

A RM can also be added to the weights of a layer in a NN:

    # The NN is just the RM - the datasets don't matter here, they just have to exist
    python3 main_scnp.py model_train \
        efc_rm_as_nn train_dh_both_hodms_efc_final_dh_lg_6iter_fl \
        val_dh_both_hodms_efc_final_dh_lg_6iter_fl \
        dh_fcn_linear mse adamw 0 1 \
        --use-rm-weights-for-layer dh_dm1_dm2 dense_layer " -1e9" \
        --quit-after-loading-rm
