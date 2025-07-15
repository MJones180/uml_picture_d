# Commands Lookup

## Data Preprocessing

Preprocess FITS datafiles from the `piccsim` library:

    # 1nm * Gaussian between -1 and 1, only uses the first HODM
    python3 main.py convert_piccsim_fits_data dm_sci_cam_first_hodm_1nm_100k \
        /home/picture/code/picture/piccsim/plots/all_sim_data/first_hodm_1nm_100k \
        --fits-file-globs 'dm1_*' 'dm2_*' 'sci_*i' 'sci*r' \
        --fits-table-names dm1 dm2 sci_i sci_r \
        --rows-per-chunk 25000 --first-n-rows 100000
