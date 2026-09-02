# Commands Lookup

## Data Conversion

Convert the polarization data to HDF:

    # Single wavelength using rx_picture_d_lab_nn
    # Saves the camera image in both polarizations, along with the EF in the plane of HODM1
    python3 main.py convert_piccsim_fits_data_merger pol_XXXXX \
        /home/michael_jones6_student_uml_edu/work/piccsim/plots/ polarization_dataset_ 314 314 1 \
        --file-names hodm1_600_r hodm1_600_i intensity_pol0 intensity_pol1 \
        --allow-missing-dirs --save-as-float32
    # pol_34957: 34957 simulations
    # pol_35371: 35371 simulations
    # pol_36910: 36910 simulations
    # pol_37922: 37922 simulations
    # pol_41728: 41728 simulations

Convert to the EF to phase and amp:

    python3 main.py convert_ef_to_phase_and_amp \
        pol_34957 hodm1_600_r hodm1_600_i \
        pol_34957_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        pol_35371 hodm1_600_r hodm1_600_i \
        pol_35371_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        pol_36910 hodm1_600_r hodm1_600_i \
        pol_36910_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        pol_37922 hodm1_600_r hodm1_600_i \
        pol_37922_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1
    python3 main.py convert_ef_to_phase_and_amp \
        pol_41728 hodm1_600_r hodm1_600_i \
        pol_41728_phase_and_amp phase amp \
        --tables-to-copy intensity_pol0 intensity_pol1

Take the sqrt of the intensities (with the EF):

    python3 main.py apply_data_transformation \
        pol_34957 pol_34957_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy hodm1_600_r hodm1_600_i --sqrt-data
    python3 main.py apply_data_transformation \
        pol_35371 pol_35371_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy hodm1_600_r hodm1_600_i --sqrt-data
    python3 main.py apply_data_transformation \
        pol_36910 pol_36910_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy hodm1_600_r hodm1_600_i --sqrt-data
    python3 main.py apply_data_transformation \
        pol_37922 pol_37922_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy hodm1_600_r hodm1_600_i --sqrt-data
    python3 main.py apply_data_transformation \
        pol_41728 pol_41728_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy hodm1_600_r hodm1_600_i --sqrt-data

Take the sqrt of the intensities (with phase and amp):

    python3 main.py apply_data_transformation \
        pol_34957_phase_and_amp pol_34957_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        pol_35371_phase_and_amp pol_35371_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        pol_36910_phase_and_amp pol_36910_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        pol_37922_phase_and_amp pol_37922_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data
    python3 main.py apply_data_transformation \
        pol_41728_phase_and_amp pol_41728_phase_and_amp_sqrt_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --sqrt-data

Take the log10 difference of the intensities (with phase and amp):

    python3 main.py apply_data_transformation \
        pol_34957_phase_and_amp pol_34957_phase_and_amp_log10_int_diff \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-7 \
        --table-difference intensity_pol0 intensity_pol1 intensity
    python3 main.py apply_data_transformation \
        pol_35371_phase_and_amp pol_35371_phase_and_amp_log10_int_diff \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-7 \
        --table-difference intensity_pol0 intensity_pol1 intensity
    python3 main.py apply_data_transformation \
        pol_36910_phase_and_amp pol_36910_phase_and_amp_log10_int_diff \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-7 \
        --table-difference intensity_pol0 intensity_pol1 intensity
    python3 main.py apply_data_transformation \
        pol_37922_phase_and_amp pol_37922_phase_and_amp_log10_int_diff \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-7 \
        --table-difference intensity_pol0 intensity_pol1 intensity
    python3 main.py apply_data_transformation \
        pol_41728_phase_and_amp pol_41728_phase_and_amp_log10_int_diff \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-7 \
        --table-difference intensity_pol0 intensity_pol1 intensity

Take the log10 of the intensities (with phase and amp):

    python3 main.py apply_data_transformation \
        pol_34957_phase_and_amp pol_34957_phase_and_amp_log10_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-15
    python3 main.py apply_data_transformation \
        pol_35371_phase_and_amp pol_35371_phase_and_amp_log10_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-15
    python3 main.py apply_data_transformation \
        pol_36910_phase_and_amp pol_36910_phase_and_amp_log10_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-15
    python3 main.py apply_data_transformation \
        pol_37922_phase_and_amp pol_37922_phase_and_amp_log10_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-15
    python3 main.py apply_data_transformation \
        pol_41728_phase_and_amp pol_41728_phase_and_amp_log10_int \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-15

    python3 main.py apply_data_transformation \
        pol_34957_phase_and_amp pol_34957_phase_and_amp_log10_int_v2 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-11
    python3 main.py apply_data_transformation \
        pol_35371_phase_and_amp pol_35371_phase_and_amp_log10_int_v2 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-11
    python3 main.py apply_data_transformation \
        pol_36910_phase_and_amp pol_36910_phase_and_amp_log10_int_v2 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-11
    python3 main.py apply_data_transformation \
        pol_37922_phase_and_amp pol_37922_phase_and_amp_log10_int_v2 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-11
    python3 main.py apply_data_transformation \
        pol_41728_phase_and_amp pol_41728_phase_and_amp_log10_int_v2 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --log10-data 1e-11

Take the arcsinh of the intensities (with phase and amp):

    python3 main.py apply_data_transformation \
        pol_34957_phase_and_amp pol_34957_phase_and_amp_arcsinh_10 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --arcsinh-data 1e-10
    python3 main.py apply_data_transformation \
        pol_35371_phase_and_amp pol_35371_phase_and_amp_arcsinh_10 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --arcsinh-data 1e-10
    python3 main.py apply_data_transformation \
        pol_36910_phase_and_amp pol_36910_phase_and_amp_arcsinh_10 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --arcsinh-data 1e-10
    python3 main.py apply_data_transformation \
        pol_37922_phase_and_amp pol_37922_phase_and_amp_arcsinh_10 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --arcsinh-data 1e-10
    python3 main.py apply_data_transformation \
        pol_41728_phase_and_amp pol_41728_phase_and_amp_arcsinh_10 \
        --tables-to-transform intensity_pol0 intensity_pol1 \
        --tables-to-copy phase amp --arcsinh-data 1e-10

## Basis commands

Create a new EF basis from PCA:

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_ef_modes_2000_36910 2000 \
        --raw-data-tags pol_36910 \
        --table-names hodm1_600_r hodm1_600_i --auto-mask

    # Not enough memory to create the modes from more data
    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_ef_modes_2000_41728 2000 \
        --raw-data-tags pol_41728_sqrt_int \
        --table-names hodm1_600_r hodm1_600_i --auto-mask

Create a new phase basis from PCA:

    # Not enough memory to create the modes from more data
    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_phase_modes_1000_79650 1000 \
        --raw-data-tags pol_37922_phase_and_amp_log10_int_diff \
                        pol_41728_phase_and_amp_log10_int_diff \
        --table-names phase --auto-mask --save-explained-variance

Create a Zernike basis:

    python3 main.py create_zernike_basis_modes \
        zernike_modes_331_2000_modes 2000 331 \
        --apply-mask pol_34957_phase_and_amp phase

Create a new amp basis from PCA:

    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_amp_modes_1000_36910 1000 \
        --raw-data-tags pol_36910_phase_and_amp \
        --table-names amp --auto-mask

    # Not enough memory to create the modes from more data
    python3 main.py create_pca_basis_modes \
        pol_hodm_plane_amp_modes_1000_79650 1000 \
        --raw-data-tags pol_37922_phase_and_amp_sqrt_int \
                        pol_41728_phase_and_amp_sqrt_int \
        --table-names amp --auto-mask

Create a new intensity basis from PCA:

    python3 main.py create_pca_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_masked_36910 2000 \
        --raw-data-tags pol_36910_phase_and_amp \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask --save-explained-variance

    python3 main.py create_pca_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 2000 \
        --raw-data-tags pol_35371_sqrt_int \
                        pol_36910_sqrt_int \
                        pol_37922_sqrt_int \
                        pol_41728_sqrt_int \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask

    python3 main.py create_pca_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 2000 \
        --raw-data-tags pol_35371_phase_and_amp_log10_int_diff \
                        pol_36910_phase_and_amp_log10_int_diff \
                        pol_37922_phase_and_amp_log10_int_diff \
                        pol_41728_phase_and_amp_log10_int_diff \
        --table-names intensity --dh-mask darkhole_mask --unit-variance --save-explained-variance

    python3 main.py create_pca_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_log10_masked_151931 2000 \
        --raw-data-tags pol_35371_phase_and_amp_log10_int \
                        pol_36910_phase_and_amp_log10_int \
                        pol_37922_phase_and_amp_log10_int \
                        pol_41728_phase_and_amp_log10_int \
        --table-names intensity_pol0 intensity_pol1 --dh-mask darkhole_mask --unit-variance --save-explained-variance

## Data Preprocessing

V1:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v1 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase amp \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       phase     pol_hodm_plane_phase_modes_1000_79650            modes 1000 \
                       amp       pol_hodm_plane_amp_modes_1000_79650              modes 1000 \
        --input-tables intensity --output-tables phase amp --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v1 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase amp \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       phase     pol_hodm_plane_phase_modes_1000_79650            modes 1000 \
                       amp       pol_hodm_plane_amp_modes_1000_79650              modes 1000 \
        --input-tables intensity --output-tables phase amp --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v1 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase amp \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       phase     pol_hodm_plane_phase_modes_1000_79650            modes 1000 \
                       amp       pol_hodm_plane_amp_modes_1000_79650              modes 1000 \
        --input-tables intensity --output-tables phase amp --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v1 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase amp \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       phase     pol_hodm_plane_phase_modes_1000_79650            modes 1000 \
                       amp       pol_hodm_plane_amp_modes_1000_79650              modes 1000 \
        --input-tables intensity --output-tables phase amp --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v1_norm train_pol_v1 \
        --z-score-norm-inputs --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v1 test_pol_v1 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase amp \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       phase     pol_hodm_plane_phase_modes_1000_79650            modes 1000 \
                       amp       pol_hodm_plane_amp_modes_1000_79650              modes 1000 \
        --input-tables intensity --output-tables phase amp --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v1_norm val_pol_v1 \
        --z-score-norm-inputs --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v1_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v1_norm test_pol_v1 \
        --z-score-norm-inputs \
        --use-existing-norm-vals train_pol_v1_norm

V2:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v2 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       ef        pol_hodm_plane_ef_modes_2000_41728               modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v2 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       ef        pol_hodm_plane_ef_modes_2000_41728               modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v2 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       ef        pol_hodm_plane_ef_modes_2000_41728               modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data --switch-asis-skip-reconstruction-error
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v2 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       ef        pol_hodm_plane_ef_modes_2000_41728               modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data --switch-basis-skip-reconstruction-error
    python3 main.py normalize_processed_dataset \
        train_pol_v2_norm train_pol_v2 \
        --z-score-norm-inputs --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v2 test_pol_v2 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes 2000 \
                       ef        pol_hodm_plane_ef_modes_2000_41728               modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v2_norm val_pol_v2 \
        --z-score-norm-inputs --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v2_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v2_norm test_pol_v2 \
        --z-score-norm-inputs \
        --use-existing-norm-vals train_pol_v2_norm

V3:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v3 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-crop intensity_pol0 21 80 21 80 intensity_pol1 21 80 21 80 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis ef pol_hodm_plane_ef_modes_2000_41728 modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v3 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-crop intensity_pol0 21 80 21 80 intensity_pol1 21 80 21 80 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis ef pol_hodm_plane_ef_modes_2000_41728 modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data --switch-basis-skip-reconstruction-error
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v3 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-crop intensity_pol0 21 80 21 80 intensity_pol1 21 80 21 80 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis ef pol_hodm_plane_ef_modes_2000_41728 modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data --switch-basis-skip-reconstruction-error
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v3 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-crop intensity_pol0 21 80 21 80 intensity_pol1 21 80 21 80 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis ef pol_hodm_plane_ef_modes_2000_41728 modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314 \
        --extend-existing-preprocessed-data --switch-basis-skip-reconstruction-error
    python3 main.py normalize_processed_dataset \
        train_pol_v3_norm train_pol_v3 \
        --max-scale-inputs --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v3 test_pol_v3 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_sqrt_int \
        --tables-to-load intensity_pol0 intensity_pol1 hodm1_600_r hodm1_600_i \
        --apply-crop intensity_pol0 21 80 21 80 intensity_pol1 21 80 21 80 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
                       hodm1_600_r    hodm1_600_i    ef \
        --switch-basis ef pol_hodm_plane_ef_modes_2000_41728 modes 2000 \
        --input-tables intensity --output-tables ef --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v3_norm val_pol_v3 \
        --max-scale-inputs --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v3_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v3_norm test_pol_v3 \
        --max-scale-inputs \
        --use-existing-norm-vals train_pol_v3_norm

V4:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v4 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp_log10_int_diff \
        --tables-to-load intensity phase \
        --apply-mask darkhole_mask dark_zone_mask intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 modes 750 \
                       phase     pol_hodm_plane_phase_modes_1000_79650                  modes 400 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v4 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp_log10_int_diff \
        --tables-to-load intensity phase \
        --apply-mask darkhole_mask dark_zone_mask intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 modes 750 \
                       phase     pol_hodm_plane_phase_modes_1000_79650                  modes 400 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v4 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp_log10_int_diff \
        --tables-to-load intensity phase \
        --apply-mask darkhole_mask dark_zone_mask intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 modes 750 \
                       phase     pol_hodm_plane_phase_modes_1000_79650                  modes 400 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v4 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp_log10_int_diff \
        --tables-to-load intensity phase \
        --apply-mask darkhole_mask dark_zone_mask intensity \
        --switch-basis intensity pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 modes 750 \
                       phase     pol_hodm_plane_phase_modes_1000_79650                  modes 400 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v4_norm train_pol_v4 \
        --z-score-norm-inputs --z-score-norm-outputs
    python3 main.py normalize_processed_dataset \
        test_on_train_pol_v4_norm train_pol_v4 \
        --z-score-norm-inputs \
        --use-existing-norm-vals train_pol_v4_norm

V5:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v5 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp_log10_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 500 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v5 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp_log10_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 500 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v5 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp_log10_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 500 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v5 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp_log10_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 500 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v5_norm train_pol_v5 \
        --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v5 test_pol_v5 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp_log10_int \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 500 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v5_norm val_pol_v5 \
        --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v5_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v5_norm test_pol_v5 \
        --z-score-norm-inputs-global \
        --use-existing-norm-vals train_pol_v5_norm

V6:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v6 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp_log10_int_v2 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v6 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp_log10_int_v2 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v6 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp_log10_int_v2 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v6 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp_log10_int_v2 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v6_norm train_pol_v6 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v6 test_pol_v6 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp_log10_int_v2 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v6_norm val_pol_v6 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v6_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v6_norm test_pol_v6 \
        --input-zero-mean-pixels --z-score-norm-inputs-global \
        --use-existing-norm-vals train_pol_v6_norm

V7:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v7 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v7 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v7 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v7 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v7_norm train_pol_v7 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v7 test_pol_v7 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v7_norm val_pol_v7 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v7_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v7_norm test_pol_v7 \
        --input-zero-mean-pixels --z-score-norm-inputs-global \
        --use-existing-norm-vals train_pol_v7_norm

V8:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v8 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp_arcsinh_10 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v8 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp_arcsinh_10 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v8 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp_arcsinh_10 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v8 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp_arcsinh_10 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v8_norm train_pol_v8 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v8 test_pol_v8 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp_arcsinh_10 \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 20 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v8_norm val_pol_v8 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v8_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v8_norm test_pol_v8 \
        --input-zero-mean-pixels --z-score-norm-inputs-global \
        --use-existing-norm-vals train_pol_v8_norm

V9:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v9 val_pol_v9 \
        --output-tag-percentages 90 10 \
        --raw-data-tags pol_41728_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 1000 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        train_pol_v9_norm train_pol_v9 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py normalize_processed_dataset \
        val_pol_v9_norm val_pol_v9 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v9_norm

V10:

    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v10 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_35371_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 555 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v10 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_36910_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 555 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v10 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_37922_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 555 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py preprocess_data_pol \
        --output-tags train_pol_v10 \
        --output-tag-percentages 100 \
        --raw-data-tags pol_41728_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 555 \
        --input-tables intensity --output-tables phase --fix-seed 314 \
        --extend-existing-preprocessed-data
    python3 main.py normalize_processed_dataset \
        train_pol_v10_norm train_pol_v10 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs
    python3 main.py preprocess_data_pol \
        --output-tags val_pol_v10 test_pol_v10 \
        --output-tag-percentages 75 25 \
        --raw-data-tags pol_34957_phase_and_amp \
        --tables-to-load intensity_pol0 intensity_pol1 phase \
        --apply-mask darkhole_mask dark_zone_mask intensity_pol0 intensity_pol1 \
        --merge-tables intensity_pol0 intensity_pol1 intensity \
        --switch-basis phase zernike_modes_331_2000_modes modes 555 \
        --input-tables intensity --output-tables phase --fix-seed 314
    python3 main.py normalize_processed_dataset \
        val_pol_v10_norm val_pol_v10 \
        --input-zero-mean-pixels --z-score-norm-inputs-global --z-score-norm-outputs \
        --use-existing-norm-vals train_pol_v10_norm
    python3 main.py normalize_processed_dataset \
        test_pol_v10_norm test_pol_v10 \
        --input-zero-mean-pixels --z-score-norm-inputs-global \
        --use-existing-norm-vals train_pol_v10_norm

## Mode Reconstructions

EF Modes:

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_ef_modes_2000_36910 modes \
        --display-from-mask pol_34957 hodm1_600_r 0 \
        --modes-are-complex 1 \
        --reconstruct-data pol_34957 2000 hodm1_600_r hodm1_600_i \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots

Phase Modes:

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_phase_modes_1000_79650 modes \
        --display-from-mask pol_34957_phase_and_amp phase 0 \
        --reconstruct-data pol_34957_phase_and_amp 400 phase \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --plot-modes-range 0 10 --plot-orthogonality --print-mean-and-std \
        --plot-singular-values --plot-explained-variance

    python3 main.py analyze_basis_modes \
        zernike_modes_331_2000_modes modes \
        --display-from-mask pol_34957_phase_and_amp phase 0 \
        --reconstruct-data pol_34957_phase_and_amp 1000 phase \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --plot-modes-range 0 10 --plot-orthogonality --print-mean-and-std \
        --plot-explained-variance --compute-explained-variance

Amp Modes:

    python3 main.py analyze_basis_modes \
        pol_hodm_plane_amp_modes_1000_36910 modes \
        --display-from-mask pol_34957_phase_and_amp amp 0 \
        --reconstruct-data pol_34957_phase_and_amp 1000 amp \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots

Intensity Modes:

    python3 main.py analyze_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_masked_36910 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --modes-are-complex 1 \
        --reconstruct-data pol_34957_phase_and_amp 2000 intensity_pol0 intensity_pol1 \
        --reconstruct-data-circle-mask --reconstruct-data-trim 21 80 21 80 \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots \
        --plot-explained-variance

    python3 main.py analyze_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_sqrt_masked_151931 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --modes-are-complex 1 \
        --reconstruct-data pol_34957_phase_and_amp_sqrt_int 2000 intensity_pol0 intensity_pol1 \
        --reconstruct-data-circle-mask --reconstruct-data-trim 21 80 21 80 \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots

    python3 main.py analyze_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_log10_diff_masked_151931 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --plot-singular-values --plot-explained-variance --plot-modes-range 0 10 \
        --reconstruct-data pol_34957_phase_and_amp_log10_int_diff 750 intensity \
        --reconstruct-data-circle-mask --reconstruct-data-trim 21 80 21 80 \
        --reconstruct-data-first-n-rows 2000 \
        --reconstruct-data-select-row 0 --reconstruct-data-plots

    python3 main.py analyze_basis_modes \
        pol_psfs_pol0_pol1_modes_2000_log10_masked_151931 modes \
        --display-as-circle 59 1.03 --display-with-hole 0.24 \
        --plot-singular-values --plot-explained-variance --plot-modes-range 0 10 \
        --modes-are-complex 1

## Random

    # Create a response matrix from a processed dataset to determine how well the
    # output data can be determined from the input data (linear approximation)
    python3 main.py linear_observability_analysis \
        train_pol_v5_norm val_pol_v5_norm --alpha 1e6
    python3 main.py linear_observability_analysis \
        train_pol_v9_norm val_pol_v9_norm --alpha 1
