# Commands Lookup

## Data Conversion

Convert FITS datafiles from the `piccsim` library to HDF:

    # A mask of the pixels used on the science camera
    python3 main.py convert_piccsim_fits_data darkhole_mask \
        /home/picture/code/picture/piccsim/plots/rx_picture_d_efcnn/sim_system_dmcmd \
        --sci-cam-mask-file sim_system_dmcmd_rx_picture_d_efcnn_pol0_sci_dhmask.fits

    # 1nm * Gaussian between -1 and 1, only uses the first HODM
    python3 main.py convert_piccsim_fits_data dh_first_hodm_1nm_100k \
        /home/picture/code/picture/piccsim/all_sim_data/first_hodm_1nm_100k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000

    # 20nm * Gaussian between -1 and 1, uses both HODMs
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

    # RM SVD modes (HODM 1 and 2)
    python3 main.py convert_piccsim_fits_data hodm1_300_modes \
        /home/picture/code/picture/piccsim/output/svd_modes/rx_ch6_vvc_dm1 \
        --fits-file-globs 'rx_ch6_vvc_scif_dm1_mode_*' \
        --fits-table-names dm1_modes --first-n-rows 300
    python3 main.py convert_piccsim_fits_data hodm2_300_modes \
        /home/picture/code/picture/piccsim/output/svd_modes/rx_ch6_vvc_dm2 \
        --fits-file-globs 'rx_ch6_vvc_scif_dm1_mode_*' \
        --fits-table-names dm1_modes --first-n-rows 300

## Data Preprocessing

Preprocess the datasets:

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_20nm val_dh_20nm test_dh_20nm 70 15 15 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding

    python3 main.py preprocess_data_dark_hole dh_both_hodms_20nm_84k \
        train_dh_20nm_xl val_dh_20nm_xl test_dh_20nm_xl 70 15 15 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding \
        --additional-raw-data-tags dh_both_hodms_20nm_100k_ch1 dh_both_hodms_20nm_100k_ch2
