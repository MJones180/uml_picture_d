"""
This script was adapted from the `preprocess_data_complete` script. This script
was created due to significant differences in the inputs (electric field at the
science camera) and outputs (DMs).

This script preprocesses datasets that can be used for training and testing.
The datasets should have been simulated with the `piccsim` library and converted
with the `convert_piccsim_fits_data` script.

Three different datasets will be outputted: training, validation, and testing.
Old datasets will be overwritten if they already exist.

All data in this script will be treated as float 32.

In this script, all normalization is optional and performed with respect to the
training dataset.
"""

import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (DARK_ZONE_MASK, DATA_F, DM_ACTIVE_COL_IDXS,
                             DM_ACTIVE_IDXS, DM_ACTIVE_ROW_IDXS, DM_SIZE,
                             EF_ACTIVE_IDXS, EXTRA_VARS_F, INPUT_MAX_MIN_DIFF,
                             INPUT_MIN_X, INPUTS, INPUTS_ARCSINH,
                             NORM_RANGE_ONES_INPUT, NORM_RANGE_ONES_OUTPUT,
                             OUTPUTS, OUTPUT_MASK, OUTPUT_MAX_MIN_DIFF,
                             OUTPUT_MIN_X, PROC_DATA_P,
                             SCI_CAM_ACTIVE_COL_IDXS, SCI_CAM_ACTIVE_ROW_IDXS)
from utils.group_data_from_list import group_data_from_list
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.norm import find_min_max_norm, min_max_norm
from utils.path import make_dir, path_exists
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.response_matrix import ResponseMatrix
from utils.stats_and_error import mse
from utils.terminate_with_message import terminate_with_message


def preprocess_data_dark_hole_parser(subparsers):
    subparser = subparsers.add_parser(
        'preprocess_data_dark_hole',
        help=('preprocess data for training, validation, and testing for the '
              'dark holes'),
    )
    subparser.set_defaults(main=preprocess_data_dark_hole)
    subparser.add_argument(
        'raw_data_tag',
        help='tag of the raw simulated data',
    )
    subparser.add_argument(
        'training_tag',
        help='tag of the outputted training dataset',
    )
    subparser.add_argument(
        'validation_tag',
        help='tag of the outputted validation dataset',
    )
    subparser.add_argument(
        'testing_tag',
        help='tag of the outputted testing dataset',
    )
    subparser.add_argument(
        'training_percentage',
        type=int,
        help='int percentage of the data that will go to training',
    )
    subparser.add_argument(
        'validation_percentage',
        type=int,
        help='int percentage of the data that will go to validation',
    )
    subparser.add_argument(
        'testing_percentage',
        type=int,
        help='int percentage of the data that will go to testing',
    )
    subparser.add_argument(
        '--additional-raw-data-tags',
        nargs='*',
        help='additional raw simulated data to preprocess and merge together',
    )
    subparser.add_argument(
        '--additional-raw-data-tags-train-only',
        nargs='*',
        help=('additional raw simulated data to preprocess and merge '
              'together, however, this data will only go in the '
              'training dataset'),
    )
    subparser.add_argument(
        '--electric-field-tables',
        nargs='+',
        help=('name of the tables containing the electric field data; if one '
              'table is passed, then the table must be complex, if two tables '
              'are passed, then the first table must contain the real part '
              'and the second table must contain the imaginary part'),
    )
    subparser.add_argument(
        '--dm-tables',
        nargs='+',
        help=('name of the tables containing the DM(s) data; the order of '
              'these tables matter as they will decide the index order and '
              'the order that the DMs are added to the output'),
    )
    subparser.add_argument(
        '--dark-zone-mask-tag',
        help=('tag of the raw datafile that contains the dark zone mask '
              '(will be used) to mask out electric field pixels'),
    )
    subparser.add_argument(
        '--remove-dark-zone-padding',
        action='store_true',
        help=('the data might contain rows and columns to the sides of the '
              'dark hole that are all zero, this arg will chop them off'),
    )
    subparser.add_argument(
        '--electric-field-handling',
        help=('[channels] (default) the real and imaginary components both '
              'get separate channels in the input image; [stack] the real '
              'and imaginary components are stacked on top of each other for '
              'an image that is 2x the original height; [complex] use a '
              'complex datatype'),
    )
    subparser.add_argument(
        '--add-total-intensity',
        action='store_true',
        help=('add the total intensity as a third channel to the data; this '
              'option only works with `--electric-field-handling channels`'),
    )
    subparser.add_argument(
        '--use-dm-svd-basis',
        nargs='+',
        metavar=('[dm table] [datafile] [datafile table] '
                 '[max number of modes from the start]'),
        help=('use the SVD basis for the DM actuator height coeffs; the raw '
              'datafile must consist of the SVD modes for the DM, the modes '
              'will be inverted to find the new basis coeffs; the DM modes '
              'must have the same number of actuators as the data; the '
              'datafile must only be a single chunk of data; the four '
              'arguments must be repeated for each DM that is being used'),
    )
    subparser.add_argument(
        '--norm-inputs',
        action='store_true',
        help=('normalize training, validation, and test input values globally '
              'between 0 and 1'),
    )
    subparser.add_argument(
        '--norm-inputs-ones',
        action='store_true',
        help=('normalize training, validation, and test input values globally '
              'between -1 and 1'),
    )
    subparser.add_argument(
        '--norm-outputs',
        action='store_true',
        help=('normalize training and validation output values individually '
              'between -1 and 1'),
    )
    subparser.add_argument(
        '--norm-outputs-globally',
        action='store_true',
        help=('normalize training and validation output values globally '
              'between -1 and 1'),
    )
    subparser.add_argument(
        '--input-arcsinh',
        action='store_true',
        help='take the arcsinh of the input data, done before norm',
    )
    subparser.add_argument(
        '--flatten-input',
        action='store_true',
        help=('flatten the 2D electric field into a 1D array; multiple '
              'channels are flattened into one channel; the imaginary part '
              'is added after the real part; all inactive pixels are removed'),
    )
    subparser.add_argument(
        '--do-not-flatten-output',
        action='store_true',
        help='do not flatten the output data, keep it as a 2D array',
    )
    subparser.add_argument(
        '--use-rm-residuals',
        help=('have the output be the residuals of the DM actuator '
              'heights after a RM prediction; the passed RM tag is '
              'loaded in and then used to create the residuals'),
    )
    subparser.add_argument(
        '--disable-shuffle',
        action='store_true',
        help='do not shuffle the rows',
    )
    subparser.add_argument(
        '--extend-existing-preprocessed-data',
        action='store_true',
        help=('add more data chunks to existing preprocessed datasets; the '
              'data tags must match existing ones for this arg to be used; '
              'the extra variables from the training dataset will be used '
              'and will not be written out a second time; the input and '
              'output norm from the training dataset will be used'),
    )


def preprocess_data_dark_hole(cli_args):
    title('Preprocess data dark hole script')

    # ==========================================================================

    def _get_out_path(cli_arg):
        return f'{PROC_DATA_P}/{cli_args[cli_arg]}'

    training_tag_path = _get_out_path('training_tag')
    validation_tag_path = _get_out_path('validation_tag')
    testing_tag_path = _get_out_path('testing_tag')

    extend_existing_data = cli_args['extend_existing_preprocessed_data']
    if extend_existing_data:
        step_ri('Will extend existing data')
        # Verifying all the datasets exist
        if not path_exists(training_tag_path):
            terminate_with_message(f'{training_tag_path} must exist')
        elif not path_exists(validation_tag_path):
            terminate_with_message(f'{validation_tag_path} must exist')
        elif not path_exists(testing_tag_path):
            terminate_with_message(f'{testing_tag_path} must exist')
        # Loading in the extra variables so they can just be used
        extra_vars = read_hdf(f'{training_tag_path}/{EXTRA_VARS_F}')
    else:
        step_ri('Setting up dataset outputs')

        def _create_dataset(out_path):
            print(f'Making {out_path}')
            make_dir(out_path)
            # Write out the CLI args that this script was called with
            save_cli_args(out_path, cli_args, 'preprocess_data_dark_hole')

        _create_dataset(training_tag_path)
        _create_dataset(validation_tag_path)
        _create_dataset(testing_tag_path)

        # Extra variables that may be written out at the end
        extra_vars = {}

        def _save_var(arg, val):
            extra_vars[arg] = val
            print(f'Will save `{arg}` at the end')

    def _use_var(var, scalar=False):
        print(f'Using {var} from `extra_vars`')
        if scalar:
            return extra_vars[var][()]
        return extra_vars[var][:]

    # ==========================================================================

    step_ri('Preparing to load in the data chunks')

    dm_tables = cli_args['dm_tables']
    ef_tables = cli_args['electric_field_tables']
    if dm_tables is None or ef_tables is None:
        terminate_with_message('The `--electric-field-tables` and '
                               '`--dm-tables` args must be passed')
    # The index that each DM corresponds to
    dm_idx_lookup = {dm_table: idx for idx, dm_table in enumerate(dm_tables)}
    # The actuator heights for each of the DMs
    all_dm_data = {dm_table: [] for dm_table in dm_tables}
    # The real and imaginary parts of the electric field
    ef_data_real = []
    ef_data_imag = []
    dm_str = ', '.join([
        f'{dm_table} ({dm_idx})' for dm_table, dm_idx in dm_idx_lookup.items()
    ])
    print(f'DM tables and indexes: {dm_str}')
    if len(ef_tables) == 1:
        print(f'Electric field table: `{ef_tables[0]}` (complex)')
    elif len(ef_tables) == 2:
        print(f'Electric field tables: `{ef_tables[0]}` (real) '
              f'and `{ef_tables[1]}` (imag) ')
    else:
        terminate_with_message('One or two electric field tables must '
                               'be passed')

    # ==========================================================================

    # A lot of type conversions will be done to float32 from float64, so this
    # variable is just saved for convenience
    F32 = np.float32

    def _load_datafile_tables(tag):
        for data_path in raw_sim_data_chunk_paths(tag):
            print(f'Loading in data from {data_path}')
            data = read_hdf(data_path)
            for dm_table in dm_tables:
                all_dm_data[dm_table].extend(data[dm_table][:].astype(F32))
            # The real and imaginary parts are separate
            if len(ef_tables) == 2:
                ef_data_real.extend(data[ef_tables[0]][:].astype(F32))
                ef_data_imag.extend(data[ef_tables[1]][:].astype(F32))
            # The table should be complex
            else:
                ef_data_complex = data[ef_tables[0]][:].astype(F32)
                ef_data_real.extend(ef_data_complex.real)
                ef_data_imag.extend(ef_data_complex.imag)

    step_ri('Loading in data chunks')
    _load_datafile_tables(cli_args['raw_data_tag'])

    if cli_args.get('additional_raw_data_tags') is not None:
        step_ri('Loading in additional data chunks')
        for tag in cli_args.get('additional_raw_data_tags'):
            _load_datafile_tables(tag)

    train_only_mask = None
    if cli_args.get('additional_raw_data_tags_train_only') is not None:
        step_ri('Loading in additional data chunks for training only')
        starting_idx = len(ef_data_real)
        for tag in cli_args.get('additional_raw_data_tags_train_only'):
            _load_datafile_tables(tag)
        # Create a mask to grab the rows that should be used for training only
        train_only_mask = np.zeros(len(ef_data_real)).astype(bool)
        train_only_mask[starting_idx:] = 1

    # ==========================================================================

    step_ri('Converting loaded data to numpy arrays')
    for dm_table, dm_data in all_dm_data.items():
        all_dm_data[dm_table] = np.asarray(dm_data)
        print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
    ef_data_real = np.asarray(ef_data_real)
    print(f'EF real shape: {ef_data_real.shape}')
    ef_data_imag = np.asarray(ef_data_imag)
    print(f'EF imag shape: {ef_data_imag.shape}')

    # ==========================================================================

    step_ri('Creating the input array (electric field)')
    ef_handling = cli_args.get('electric_field_handling')
    if ef_handling is None:
        ef_handling = 'channels'
    print(f'Method chosen: {ef_handling}')
    if ef_handling == 'complex':
        print('The electric field will be a single, complex channel')
        input_data = ef_data_real + ef_data_imag * 1j
        # Create a new dimension for the channel
        input_data = input_data[:, None, :, :]
    else:
        print('The electric field will be two channels of real data')
        print('The first channel is the real part, '
              'the second channel is the imag part')
        if ef_handling == 'stack':
            print('The stacking will be done later')
        input_data = np.stack((ef_data_real, ef_data_imag), axis=1)
    print(f'Input shape: {input_data.shape}')

    # ==========================================================================

    if cli_args['input_arcsinh']:
        step_ri('Taking the arcsinh of the input data')
        input_data = np.arcsinh(input_data)
        _save_var(INPUTS_ARCSINH, True)

    # ==========================================================================

    dark_zone_mask_tag = cli_args['dark_zone_mask_tag']
    if dark_zone_mask_tag is not None:
        step_ri('Applying the dark zone mask to the inputs')
        if extend_existing_data:
            dark_zone_mask = _use_var(DARK_ZONE_MASK)
        else:
            mask_path = raw_sim_data_chunk_paths(dark_zone_mask_tag)[0]
            print(f'Loading in the dark zone mask from {mask_path}')
            dark_zone_mask = read_hdf(mask_path)[DARK_ZONE_MASK][:]
            _save_var(DARK_ZONE_MASK, dark_zone_mask)
        print(f'This mask has {dark_zone_mask.sum()} active pixels')
        # Zero out all pixels that are not within the mask
        input_data[:, :, ~dark_zone_mask] = 0
        print('Inactive pixels zeroed out')

    # ==========================================================================

    if cli_args['remove_dark_zone_padding']:
        step_ri('Chopping off zero padded pixels in the inputs')
        if extend_existing_data:
            active_col_idxs = _use_var(SCI_CAM_ACTIVE_COL_IDXS)
            active_row_idxs = _use_var(SCI_CAM_ACTIVE_ROW_IDXS)
        else:
            # Create an array where each pixel will say if it is nonzero
            # across any of the simulations
            nonzero_pixels = (input_data != 0).any(axis=(0, 1))
            # Indexes of the rows and columns in the input where there is at
            # least one pixel in that row or column that has a nonzero value
            active_col_idxs = np.where(nonzero_pixels.any(axis=(0)))[0]
            active_row_idxs = np.where(nonzero_pixels.any(axis=(1)))[0]
            _save_var(SCI_CAM_ACTIVE_COL_IDXS, active_col_idxs)
            _save_var(SCI_CAM_ACTIVE_ROW_IDXS, active_row_idxs)
        # Chop off the padding rows and columns
        input_data = input_data[:, :, :, active_col_idxs]
        input_data = input_data[:, :, active_row_idxs]
        print(f'Input shape: {input_data.shape}')

    # ==========================================================================

    if ef_handling == 'stack':
        step_ri('Stacking the electric field components')
        print('The electric field will be one channel of real data')
        print('The real part will be stacked on top of the imag part')
        input_data = np.concatenate((input_data[:, 0], input_data[:, 1]),
                                    axis=1)[:, None, :, :]
        print(f'Input shape: {input_data.shape}')

    # ==========================================================================

    if cli_args['add_total_intensity']:
        step_ri('Adding the total intensity as a third channel')
        if ef_handling != 'channels':
            terminate_with_message('This option can only be called with '
                                   '--electric-field-handling channels')
        intensity = input_data[:, 0]**2 + input_data[:, 1]**2
        intensity = intensity[:, None, :, :]
        input_data = np.concatenate((input_data, intensity), axis=1)
        print(f'Input shape: {input_data.shape}')

    # ==========================================================================

    def _grab_flattened_ef_data():
        orig_shape = input_data.shape
        data_flat = input_data.reshape(orig_shape[0], -1)
        updated_shape = data_flat.shape
        print(f'Shape: {orig_shape} -> {updated_shape}')
        print('Removing inactive pixels')
        if extend_existing_data:
            active_idxs = _use_var(EF_ACTIVE_IDXS)
        else:
            # Create an array where each pixel will say if it is nonzero
            # across any of the simulations
            nonzero_pixels = (data_flat != 0).any(axis=0)
            print(f'EF has {nonzero_pixels.sum()} active pixels')
            # Idxs where there is at least one pixel that has a nonzero value
            active_idxs = np.where(nonzero_pixels)[0]
            _save_var(EF_ACTIVE_IDXS, nonzero_pixels)
        # Filter out the inactive pixels
        data_flat = data_flat[:, active_idxs]
        print(f'Shape: {updated_shape} -> {data_flat.shape}')
        return data_flat

    flatten_input = cli_args['flatten_input']
    if flatten_input:
        step_ri('Flattening the input data into a 1D array')
        input_data = _grab_flattened_ef_data()

    # ==========================================================================

    use_rm_residuals = cli_args.get('use_rm_residuals')
    if use_rm_residuals is not None:
        step_ri('Will use RM residuals for the output')
        resp_mat = ResponseMatrix(use_rm_residuals)
        if flatten_input:
            input_data_flat = input_data
        else:
            step('Flattening the input data for the RM')
            input_data_flat = _grab_flattened_ef_data()
            dec_print_indent()
        print('Calling the RM')
        rm_output = resp_mat(ef=input_data_flat)
        print('Reshaping the data to 2D and creating the residuals')
        for dm_table, dm_data in all_dm_data.items():
            # Create a 2D template which the RM command will be filled in on
            rm_data_2d = np.zeros_like(dm_data)
            # Find the indexes for each of the DM actuators
            idxs_of_actuators = np.where(dm_data[0] != 0)
            # Grab the total number of actuators on the DM
            numb_of_actuators = len(idxs_of_actuators[0])
            # Grab the actuators associated with this DM
            rm_output_for_dm = rm_output[:, :numb_of_actuators]
            # Remove the actuators so they are not grabbed for the next DM
            rm_output = rm_output[:, numb_of_actuators:]
            # Put the actuators on the 2D grid
            rm_data_2d[:, *idxs_of_actuators] = rm_output_for_dm
            # Have the DM data store just the residuals
            dm_data -= rm_data_2d

    # ==========================================================================

    do_not_flatten_output = cli_args['do_not_flatten_output']
    if do_not_flatten_output:
        step_ri('Will not flatten the output data')
        print('Chopping off zero padded pixels in the outputs')
        for dm_table, dm_data in all_dm_data.items():
            # The index of the current DM
            dm_idx = dm_idx_lookup[dm_table]
            if extend_existing_data:
                active_col_idxs = _use_var(DM_ACTIVE_COL_IDXS(dm_idx))
                active_row_idxs = _use_var(DM_ACTIVE_ROW_IDXS(dm_idx))
            else:
                # Create an array where each pixel will say if it is nonzero
                # across any of the simulations
                nonzero_pixels = (dm_data != 0).any(axis=0)
                # Indexes of the rows and columns in the input where there is at
                # least one pixel in that row or column that has a nonzero value
                active_col_idxs = np.where(nonzero_pixels.any(axis=(0)))[0]
                active_row_idxs = np.where(nonzero_pixels.any(axis=(1)))[0]
                _save_var(DM_ACTIVE_COL_IDXS(dm_idx), active_col_idxs)
                _save_var(DM_ACTIVE_ROW_IDXS(dm_idx), active_row_idxs)
            # Chop off the padding rows and columns
            dm_data = dm_data[:, :, active_col_idxs]
            dm_data = dm_data[:, active_row_idxs]
            all_dm_data[dm_table] = dm_data
            # Create a mask of all the active actuator locations
            output_mask = (dm_data[0] != 0).astype(bool)
            _save_var(OUTPUT_MASK, output_mask)
            print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
    else:
        step_ri('Flattening the DM actuator height data')
        for dm_table, dm_data in all_dm_data.items():
            dm_shape = dm_data.shape
            all_dm_data[dm_table] = np.reshape(dm_data, (dm_shape[0], -1))
            print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
            if not extend_existing_data:
                _save_var(DM_SIZE(dm_idx_lookup[dm_table]), dm_shape[1:])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        step_ri('Finding and removing inactive actuators on the DM(s)')
        for dm_table, dm_data in all_dm_data.items():
            # The index of the current DM
            dm_idx = dm_idx_lookup[dm_table]
            if extend_existing_data:
                active_idxs = _use_var(DM_ACTIVE_IDXS(dm_idx))
            else:
                # Create an array where each actuator will say if it is nonzero
                # across any of the simulations
                nonzero_acts = (dm_data != 0).any(axis=0)
                print(
                    f'DM {dm_table} has {nonzero_acts.sum()} active actuators')
                # Idxs where there is >= 1 actuator with a nonzero value
                active_idxs = np.where(nonzero_acts)[0]
                _save_var(DM_ACTIVE_IDXS(dm_idx), active_idxs)
            # Filter out the inactive actuators
            all_dm_data[dm_table] = dm_data[:, active_idxs]
            print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')

    # ==========================================================================

    use_dm_svd_basis = cli_args.get('use_dm_svd_basis')
    if use_dm_svd_basis is not None:
        step_ri('Using the SVD basis functions for the DM actuator heights')
        err_msg = 'Four arguments must be passed for each DM'
        for group in group_data_from_list(use_dm_svd_basis, 4, err_msg):
            dm_table, modes_tag, modes_table_name, max_modes = group
            print(f'[DM {dm_table}] Using {max_modes} modes from {modes_tag}')
            modes_path = raw_sim_data_chunk_paths(modes_tag)[0]
            modes = read_hdf(modes_path)[modes_table_name][:].astype(F32)
            modes = modes.reshape(modes.shape[0], -1)
            # Filter out the inactive pixels from the modes
            active_idxs = _use_var(DM_ACTIVE_IDXS(dm_idx_lookup[dm_table]))
            modes = modes[:, active_idxs]
            # Pick out the correct number of modes from the start
            modes = modes[:int(max_modes)]
            # Invert the modes
            modes_inv = np.linalg.pinv(modes)
            # The original actuator heights for the DM
            actuator_heights = all_dm_data[dm_table]
            # The coefficients in the new basis
            new_basis_coeffs = actuator_heights @ modes_inv
            # What the actuator heights look like in the new basis
            reconstructed_actuator_heights = new_basis_coeffs @ modes
            # The error when switching to the new basis representation
            error = mse(actuator_heights, reconstructed_actuator_heights)
            print(f'Actuator height reconstruction MSE error of {error:0.3e}')
            all_dm_data[dm_table] = new_basis_coeffs
            print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')

    # ==========================================================================

    step_ri('Creating the output array (dm actuator height coeffs)')
    output_data = None
    for dm_table, dm_data in all_dm_data.items():
        if output_data is None:
            output_data = dm_data
        else:
            if do_not_flatten_output:
                output_data = np.stack((output_data, dm_data), axis=1)
            else:
                output_data = np.hstack((output_data, dm_data))
    print(f'Output shape: {output_data.shape}')

    # ==========================================================================

    if not cli_args['disable_shuffle']:
        step_ri('Shuffling')
        random_shuffle_idxs = np.random.permutation(len(input_data))
        input_data = input_data[random_shuffle_idxs]
        output_data = output_data[random_shuffle_idxs]
        if train_only_mask is not None:
            train_only_mask = train_only_mask[random_shuffle_idxs]

    # ==========================================================================

    step_ri('Splitting')
    training_percentage = cli_args['training_percentage']
    validation_percentage = cli_args['validation_percentage']
    testing_percentage = cli_args['testing_percentage']
    psum = training_percentage + validation_percentage + testing_percentage
    if psum != 100:
        terminate_with_message(f'Percentages must add up to 100%, at {psum}%')
    # Split apart the train only data so it does not go into the other datasets
    if train_only_mask is not None:
        input_data_train_only = input_data[train_only_mask]
        output_data_train_only = output_data[train_only_mask]
        input_data = input_data[~train_only_mask]
        output_data = output_data[~train_only_mask]
    # Add up the percentages for where the data will be split
    idxs = np.cumsum((training_percentage, validation_percentage)) / 100
    # Need to figure out how many rows this equates to
    idxs *= input_data.shape[0]
    # Convert to integers so the values can be split
    idxs = idxs.astype(int)
    # Split the data into the three datasets
    train_inputs, val_inputs, test_inputs = np.split(input_data, idxs)
    train_outputs, val_outputs, test_outputs = np.split(output_data, idxs)
    # Add back in the train only data if needed
    if train_only_mask is not None:
        train_inputs = np.vstack((train_inputs, input_data_train_only))
        train_outputs = np.vstack((train_outputs, output_data_train_only))

    def _print_split(word, percentage, inputs):
        print(f'{word} percentage: {(percentage)}%, '
              f'rows: {inputs.shape[0]}')

    _print_split('Training', training_percentage, train_inputs)
    _print_split('Validation', validation_percentage, val_inputs)
    _print_split('Testing', testing_percentage, test_inputs)

    # ==========================================================================

    norm_inputs_ones = cli_args['norm_inputs_ones']
    if cli_args['norm_inputs'] or norm_inputs_ones:
        step_ri('Normalizing training inputs globally')
        if norm_inputs_ones:
            print('Using range [-1, 1]')
            _save_var(NORM_RANGE_ONES_INPUT, True)
        else:
            print('Using range [0, 1]')
        if extend_existing_data:
            max_min_diff = _use_var(INPUT_MAX_MIN_DIFF, True)
            min_x = _use_var(INPUT_MIN_X, True)
            train_inputs = min_max_norm(train_inputs,
                                        max_min_diff,
                                        min_x,
                                        ones_range=norm_inputs_ones)
        else:
            train_inputs, max_min_diff, min_x = find_min_max_norm(
                train_inputs, globally=True, ones_range=norm_inputs_ones)
            _save_var(INPUT_MAX_MIN_DIFF, max_min_diff)
            _save_var(INPUT_MIN_X, min_x)
        print('Normalizing inputs of validation data and test data based on '
              'training normalization values')
        val_inputs = min_max_norm(val_inputs, max_min_diff, min_x)
        test_inputs = min_max_norm(test_inputs, max_min_diff, min_x)

    # ==========================================================================

    norm_outputs_globally = cli_args['norm_outputs_globally']
    if cli_args['norm_outputs'] or norm_outputs_globally:
        step_ri('Normalizing training outputs between -1 and 1')
        if norm_outputs_globally:
            print('Normalizing globally')
            scalar_values = True
        else:
            print('Normalizing individually')
            scalar_values = False
        if extend_existing_data:
            max_min_diff = _use_var(OUTPUT_MAX_MIN_DIFF, scalar_values)
            min_x = _use_var(OUTPUT_MIN_X, scalar_values)
            train_outputs = min_max_norm(train_outputs,
                                         max_min_diff,
                                         min_x,
                                         ones_range=True)
        else:
            train_outputs, max_min_diff, min_x = find_min_max_norm(
                train_outputs, globally=norm_outputs_globally, ones_range=True)
            _save_var(OUTPUT_MAX_MIN_DIFF, max_min_diff)
            _save_var(OUTPUT_MIN_X, min_x)
            _save_var(NORM_RANGE_ONES_OUTPUT, True)
        print('Normalizing outputs of validation data based on training '
              'normalization values')
        val_outputs = min_max_norm(val_outputs, max_min_diff, min_x, True)

    # ==========================================================================

    def _write_data(out_path, inputs, outputs):
        datafile_path = f'{out_path}/{DATA_F}'
        step_ri(f'Writing out to {datafile_path}')
        if extend_existing_data:
            print('Merging existing data in')
            # Add on to the existing datafiles
            with read_hdf(datafile_path) as existing_data:
                inputs = np.vstack((existing_data[INPUTS][:], inputs))
                outputs = np.vstack((existing_data[OUTPUTS][:], outputs))
        else:
            # Add a file with extra necessary variables
            file_path = f'{out_path}/{EXTRA_VARS_F}'
            HDFWriteModule(file_path).create_and_write_hdf_simple(extra_vars)
        # Write out the processed HDF file
        HDFWriteModule(datafile_path).create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })
        print(f'Input shape: {inputs.shape}')
        print(f'Output shape: {outputs.shape}')

    _write_data(training_tag_path, train_inputs, train_outputs)
    _write_data(validation_tag_path, val_inputs, val_outputs)
    _write_data(testing_tag_path, test_inputs, test_outputs)
