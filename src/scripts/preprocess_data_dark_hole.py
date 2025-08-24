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
from utils.constants import (DARK_ZONE_MASK, DATA_F, DM_ACTIVE_IDXS, DM_SIZE,
                             EXTRA_VARS_F, INPUT_MAX_MIN_DIFF, INPUT_MIN_X,
                             INPUTS, NORM_RANGE_ONES_OUTPUT, OUTPUTS,
                             OUTPUT_MAX_MIN_DIFF, OUTPUT_MIN_X, PROC_DATA_P,
                             SCI_CAM_ACTIVE_COL_IDXS, SCI_CAM_ACTIVE_ROW_IDXS)
from utils.group_data_from_list import group_data_from_list
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.norm import find_min_max_norm, min_max_norm
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
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
              'option only works with `--add-total-intensity channels`'),
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
        '--norm-outputs',
        action='store_true',
        help=('normalize training and validation output values individually '
              'between -1 and 1'),
    )


def preprocess_data_dark_hole(cli_args):
    title('Preprocess data dark hole script')

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
        all_dm_data[dm_table] = np.array(dm_data).astype(F32)
        print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
    ef_data_real = np.array(ef_data_real).astype(F32)
    print(f'EF real shape: {ef_data_real.shape}')
    ef_data_imag = np.array(ef_data_imag).astype(F32)
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

    # Extra variables that may be written out at the end
    extra_vars = {}

    def _save_var(arg, val):
        extra_vars[arg] = val
        print(f'Will save `{arg}` at the end')

    # ==========================================================================

    dark_zone_mask_tag = cli_args['dark_zone_mask_tag']
    if dark_zone_mask_tag is not None:
        step_ri('Applying the dark zone mask to the inputs')
        mask_path = raw_sim_data_chunk_paths(dark_zone_mask_tag)[0]
        print(f'Loading in the dark zone mask from {mask_path}')
        dark_zone_mask = read_hdf(mask_path)[DARK_ZONE_MASK][:]
        print(f'This mask has {dark_zone_mask.sum()} active pixels')
        # Zero out all pixels that are not within the mask
        input_data[:, :, ~dark_zone_mask] = 0
        print('Inactive pixels zeroed out')
        _save_var(DARK_ZONE_MASK, dark_zone_mask)

    # ==========================================================================

    if cli_args['remove_dark_zone_padding']:
        step_ri('Chopping off zero padded pixels in the inputs')
        # Create an array where each pixel will say if it is nonzero
        # across any of the simulations
        nonzero_pixels = (input_data != 0).any(axis=(0, 1))
        # Indexes of the rows and columns in the input where there is at least
        # one pixel in that row or column that has a nonzero value
        active_col_idxs = np.where(nonzero_pixels.any(axis=(0)))[0]
        active_row_idxs = np.where(nonzero_pixels.any(axis=(1)))[0]
        # Chop off the padding rows and columns
        input_data = input_data[:, :, :, active_col_idxs]
        input_data = input_data[:, :, active_row_idxs]
        print(f'Input shape: {input_data.shape}')
        _save_var(SCI_CAM_ACTIVE_COL_IDXS, active_col_idxs)
        _save_var(SCI_CAM_ACTIVE_ROW_IDXS, active_row_idxs)

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

    step_ri('Flattening the DM actuator height data')
    for dm_table, dm_data in all_dm_data.items():
        dm_shape = dm_data.shape
        all_dm_data[dm_table] = np.reshape(dm_data, (dm_shape[0], -1))
        print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
        _save_var(DM_SIZE(dm_idx_lookup[dm_table]), dm_shape[1:])

    # ==========================================================================

    step_ri('Finding and removing inactive actuators on the DM(s)')
    for dm_table, dm_data in all_dm_data.items():
        # Create an array where each actuator will say if it is nonzero
        # across any of the simulations
        nonzero_acts = (dm_data != 0).any(axis=0)
        print(f'DM {dm_table} has {nonzero_acts.sum()} active actuators')
        # Idxs where there is at least one actuator that has a nonzero value
        active_idxs = np.where(nonzero_acts)[0]
        # Filter out the inactive actuators
        all_dm_data[dm_table] = dm_data[:, active_idxs]
        print(f'DM {dm_table} shape: {all_dm_data[dm_table].shape}')
        _save_var(DM_ACTIVE_IDXS(dm_idx_lookup[dm_table]), active_idxs)

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
            active_idxs = extra_vars[DM_ACTIVE_IDXS(dm_idx_lookup[dm_table])]
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
        print(f'Appending DM {dm_table} data')
        if output_data is None:
            output_data = dm_data
        else:
            output_data = np.hstack((output_data, dm_data))
    print(f'Output shape: {output_data.shape}')

    # ==========================================================================

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

    if cli_args['norm_inputs']:
        step_ri('Normalizing training inputs globally between 0 and 1')
        train_inputs, max_min_diff, min_x = find_min_max_norm(train_inputs,
                                                              globally=True)
        _save_var(INPUT_MAX_MIN_DIFF, max_min_diff)
        _save_var(INPUT_MIN_X, min_x)
        print('Normalizing inputs of validation data and test data based on '
              'training normalization values')
        val_inputs = min_max_norm(val_inputs, max_min_diff, min_x)
        test_inputs = min_max_norm(test_inputs, max_min_diff, min_x)

    # ==========================================================================

    if cli_args['norm_outputs']:
        step_ri('Normalizing training outputs individually between -1 and 1')
        train_outputs, max_min_diff, min_x = find_min_max_norm(train_outputs,
                                                               ones_range=True)
        _save_var(OUTPUT_MAX_MIN_DIFF, max_min_diff)
        _save_var(OUTPUT_MIN_X, min_x)
        _save_var(NORM_RANGE_ONES_OUTPUT, True)
        print('Normalizing outputs of validation data based on training '
              'normalization values')
        val_outputs = min_max_norm(val_outputs, max_min_diff, min_x, True)

    # ==========================================================================

    step_ri('Creating new datasets')

    def _create_dataset(cli_arg, inputs, outputs):
        out_path = f'{PROC_DATA_P}/{cli_args[cli_arg]}'
        print(f'Making {out_path}')
        make_dir(out_path)
        # Write out the CLI args that this script was called with
        save_cli_args(out_path, cli_args, 'preprocess_data_dark_hole')
        # Add a file with other necessary variables (includes norm values)
        HDFWriteModule(f'{out_path}/{EXTRA_VARS_F}'
                       ).create_and_write_hdf_simple(extra_vars)
        # Write out the processed HDF file
        HDFWriteModule(f'{out_path}/{DATA_F}').create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })

    _create_dataset('training_tag', train_inputs, train_outputs)
    _create_dataset('validation_tag', val_inputs, val_outputs)
    _create_dataset('testing_tag', test_inputs, test_outputs)
