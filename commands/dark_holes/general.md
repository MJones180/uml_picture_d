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
    python3 main.py convert_piccsim_fits_data dh_both_hodms_20nm_100k \
        /home/picture/code/picture/piccsim/all_sim_data/both_hodm_20nm_100k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000

## Data Preprocessing

Preprocess the datasets:

    python3 main.py preprocess_data_dms_sci_cam dh_both_hodms_20nm_100k \
        train_dh_20nm val_dh_20nm test_dh_20nm 70 15 15 \
        --dm-tables dm1 dm2 --electric-field-tables sci_r sci_i \
        --dark-zone-mask-tag darkhole_mask --remove-dark-zone-padding
