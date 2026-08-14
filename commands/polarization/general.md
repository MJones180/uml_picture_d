# Commands Lookup

## Data Conversion

Polarization data:

    # Single wavelength using rx_picture_d_lab_nn
    # Saves the camera image in both polarizations, along with the EF in the plane of HODM1
    python3 main.py convert_piccsim_fits_data_merger dh_pol_XXXXX \
        /home/michael_jones6_student_uml_edu/work/piccsim/plots/ polarization_dataset_ 314 314 1 \
        --file-names hodm1_600_r hodm1_600_i intensity_pol0 intensity_pol1 \
        --allow-missing-dirs --save-as-float32
    # dh_pol_34957: 34957 simulations
    # dh_pol_35371: 35371 simulations
    # dh_pol_36910: 36910 simulations
    # dh_pol_37922: 37922 simulations
    # dh_pol_41728: 41728 simulations
    # ---------------
    python3 main.py convert_ef_to_phase_and_amp \
        dh_pol_34957 hodm1_600_r hodm1_600_i \
        dh_pol_34957_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        dh_pol_35371 hodm1_600_r hodm1_600_i \
        dh_pol_35371_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        dh_pol_36910 hodm1_600_r hodm1_600_i \
        dh_pol_36910_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        dh_pol_37922 hodm1_600_r hodm1_600_i \
        dh_pol_37922_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        dh_pol_41728 hodm1_600_r hodm1_600_i \
        dh_pol_41728_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    # ---------------
    python3 main.py apply_data_transformation \
        dh_pol_34957_phase_and_amp dh_pol_34957_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        dh_pol_35371_phase_and_amp dh_pol_35371_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        dh_pol_36910_phase_and_amp dh_pol_36910_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        dh_pol_37922_phase_and_amp dh_pol_37922_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        dh_pol_41728_phase_and_amp dh_pol_41728_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    # ---------------
    python3 main.py apply_data_transformation \
        dh_pol_36910_phase_and_amp dh_pol_36910_phase_and_amp_log_int  \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-10

## Data Preprocessing

Preprocess the datasets:

## Random Commands

Create a new basis from PCA:

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_ef_modes_2000_36910 2000 \
        --raw-data-tags dh_pol_36910 \
        --table-names hodm1_600_r hodm1_600_i --auto-mask

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_phase_modes_1000_36910 1000 \
        --raw-data-tags dh_pol_36910_phase_and_amp \
        --table-names phase --auto-mask

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_amp_modes_1000_36910 1000 \
        --raw-data-tags dh_pol_36910_phase_and_amp \
        --table-names amp --auto-mask

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_masked_36910 2000 \
        --raw-data-tags dh_pol_36910_phase_and_amp \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_log_masked_36910 2000 \
        --raw-data-tags dh_pol_36910_phase_and_amp_log_int \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_sqrt_masked_36910 2000 \
        --raw-data-tags dh_pol_36910_phase_and_amp_sqrt_int \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask

Plot SVD basis reconstructions:

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_ef_modes_2000_36910 modes \
        --display-from-mask dh_pol_34957 hodm1_600_r 0 \
        --modes-are-complex 1 \
        --reconstruct-data dh_pol_34957 2000 hodm1_600_r hodm1_600_i \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --reconstruct-data-mean-subtraction pol_hodm_plane_ef_modes_2000_36910 mean

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_phase_modes_1000_36910 modes \
        --display-from-mask dh_pol_34957_phase_and_amp phase 0 \
        --reconstruct-data dh_pol_34957_phase_and_amp 1000 phase \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --reconstruct-data-mean-subtraction pol_hodm_plane_phase_modes_1000_36910 mean

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_amp_modes_1000_36910 modes \
        --display-from-mask dh_pol_34957_phase_and_amp amp 0 \
        --reconstruct-data dh_pol_34957_phase_and_amp 1000 amp \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --reconstruct-data-mean-subtraction pol_hodm_plane_amp_modes_1000_36910 mean

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_masked_36910 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --modes-are-complex 1 \
        --reconstruct-data dh_pol_34957_phase_and_amp 2000 intensity_pol0 intensity_pol1 \
        --reconstruct-data-circle-mask --reconstruct-data-trim 21 80 21 80 \
        --reconstruct-data-mean-subtraction pol_hodm_plane_pol0_pol1_modes_2000_masked_36910 mean \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_sqrt_masked_36910 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --modes-are-complex 1 --plot-modes-range 0 10

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_pol0_pol1_modes_2000_sqrt_masked_36910 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --modes-are-complex 1 \
        --reconstruct-data dh_pol_34957_phase_and_amp_sqrt_int 2000 intensity_pol0 intensity_pol1 \
        --reconstruct-data-circle-mask --reconstruct-data-trim 21 80 21 80 \
        --reconstruct-data-mean-subtraction pol_hodm_plane_pol0_pol1_modes_2000_sqrt_masked_36910 mean \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots
