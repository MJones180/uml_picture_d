"""
This script expects a wf intensity as input and Zernike coefficients as output.

This script preprocesses datasets that can be used for training and testing.
The datasets should have been simulated by the `sim_data` script.

Three different datasets will be outputted: training, validation, and testing.
Old datasets will be overwritten if they already exist.

In this script, all normalization is optional. By default, inputs are normalized
and outputs are not normalized. All normalization is always performed with
respect to the training dataset. Input normalization can be done globally and
output normalization can be done either globally or individually.
"""

import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (BASE_INT_FIELD, CAMERA_SAMPLING, DATA_F,
                             EXTRA_VARS_F, INPUTS, INPUTS_SUM_TO_ONE,
                             INPUT_MIN_X, INPUT_MAX_MIN_DIFF,
                             NORM_RANGE_ONES_INPUT, NORM_RANGE_ONES_OUTPUT,
                             OUTPUTS, OUTPUT_MIN_X, OUTPUT_MAX_MIN_DIFF,
                             PROC_DATA_P, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.norm import find_min_max_norm, min_max_norm, sum_to_one
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message

OUTPUT_NORM_OPTIONS = ['globally', 'individually']


def preprocess_data_complete_parser(subparsers):
    subparser = subparsers.add_parser(
        'preprocess_data_complete',
        help='preprocess data for training, validation, and testing',
    )
    subparser.set_defaults(main=preprocess_data_complete)
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
        default=70,
        help='int percentage of the data that will go to training',
    )
    subparser.add_argument(
        'validation_percentage',
        type=int,
        default=15,
        help='int percentage of the data that will go to validation',
    )
    subparser.add_argument(
        'testing_percentage',
        type=int,
        default=15,
        help='int percentage of the data that will go to testing',
    )
    subparser.add_argument(
        '--disable-norm-inputs',
        action='store_true',
        help=('disable global input normalization (does not effect the '
              '`inputs_sum_to_one` arg)'),
    )
    subparser.add_argument(
        '--inputs-sum-to-one',
        action='store_true',
        help='make the pixel values in each input sum to 1',
    )
    subparser.add_argument(
        '--norm-outputs',
        choices=OUTPUT_NORM_OPTIONS,
        help=('normalize training and validation output values either '
              'globally or individually'),
    )
    subparser.add_argument(
        '--norm-range-ones',
        action='store_true',
        help=('normalize data between -1 and 1 instead of 0 to 1; applies to '
              'both input and output normalization'),
    )
    subparser.add_argument(
        '--use-field-diff',
        help=('for the intensity fields (the inputs), take the difference '
              'between the base field (no aberrations) and each aberrated '
              'field; should be the name of the raw dataset with one row '
              'containing no aberrations'),
    )
    subparser.add_argument(
        '--use-field-diff-mapping',
        nargs='*',
        type=int,
        help=('map specific base fields to different portions of the data; '
              'the arguments can be repeated as many times as necessary and '
              'should specify <base field index> <starting row> <ending row>'),
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
        '--outputs-in-surface-error',
        action='store_true',
        help=('the Zernike coefficients are in terms of surface error instead '
              'of wavefront error'),
    )
    subparser.add_argument(
        '--outputs-scaling-factor',
        type=float,
        help='multiply the Zernike coefficients by a scaling factor',
    )
    subparser.add_argument(
        '--keep-only-unique-rows',
        action='store_true',
        help='remove any rows that are not unique',
    )


def preprocess_data_complete(cli_args):
    title('Preprocess data complete script')

    # ==========================================================================

    step_ri('Loading in data chunks')
    (input_data, output_data, zernike_terms,
     camera_sampling) = load_raw_sim_data_chunks(cli_args['raw_data_tag'])
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    print(f'Zernike terms: {zernike_terms}')

    def _load_and_merge_chunks(tag):
        (additional_input_data, additional_output_data,
         additional_zernike_terms, _) = load_raw_sim_data_chunks(tag)
        # Keep stacking on the other datasets
        input_data_merged = np.vstack((input_data, additional_input_data))
        output_data_merged = np.vstack((output_data, additional_output_data))
        if not np.array_equal(additional_zernike_terms, zernike_terms):
            terminate_with_message('Zernike terms from additional dataset '
                                   'are different')
        print(f'Merged input shape: {input_data_merged.shape}')
        print(f'Merged output shape: {output_data_merged.shape}')
        return input_data_merged, output_data_merged

    if cli_args.get('additional_raw_data_tags') is not None:
        step_ri('Loading in additional data chunks')
        for tag in cli_args.get('additional_raw_data_tags'):
            input_data, output_data = _load_and_merge_chunks(tag)

    if cli_args['keep_only_unique_rows']:
        step_ri('Removing any non-unique rows')
        unique_idxs = np.unique(output_data, return_index=True, axis=0)[1]
        non_unique_count = output_data.shape[0] - unique_idxs.shape[0]
        print(f'Removing {non_unique_count} rows')
        input_data = input_data[unique_idxs]
        output_data = output_data[unique_idxs]
        print(f'Input shape: {input_data.shape}')
        print(f'Output shape: {output_data.shape}')

    train_only_mask = None
    if cli_args.get('additional_raw_data_tags_train_only') is not None:
        step_ri('Loading in additional data chunks for training only')
        starting_idx = input_data.shape[0]
        for tag in cli_args.get('additional_raw_data_tags_train_only'):
            input_data, output_data = _load_and_merge_chunks(tag)
        # Create a mask to grab the rows that should be used for training only
        train_only_mask = np.zeros(input_data.shape[0]).astype(bool)
        train_only_mask[starting_idx:] = 1

    # ==========================================================================

    inputs_sum_to_one = cli_args.get('inputs_sum_to_one')
    if inputs_sum_to_one:
        step_ri('Making pixel values in each input sum to 1')
        input_data = sum_to_one(input_data, (1, 2))

    # ==========================================================================

    step_ri('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    # ==========================================================================

    if cli_args['outputs_in_surface_error']:
        step_ri('Converting from surface error to wavefront error')
        print('Multiplying output data (Zernike coefficients) by 2')
        output_data *= 2

    # ==========================================================================

    outputs_scaling_factor = cli_args.get('outputs_scaling_factor')
    if outputs_scaling_factor:
        step_ri('Adding a scaling factor to the outputs')
        print(f'Multiplying output data by {outputs_scaling_factor}')
        output_data *= outputs_scaling_factor

    # ==========================================================================

    # The rows with no aberrations, these are equal to the base field
    no_aber_rows = np.all(output_data == 0, axis=1)
    no_aber_input_row = None
    if no_aber_rows.any():
        step_ri('Removing all rows with no aberrations')
        # This should be the same as the base_field, but the base_field may not
        # be passed in from the separate datafile
        no_aber_input_row = input_data[no_aber_rows][0]
        no_aber_output_row = output_data[no_aber_rows][0]
        # Chop of all rows with no aberrations
        input_data = input_data[~no_aber_rows]
        output_data = output_data[~no_aber_rows]
        if train_only_mask is not None:
            train_only_mask = train_only_mask[~no_aber_rows]

    # ==========================================================================

    use_field_diff = cli_args['use_field_diff']
    base_field = None
    if use_field_diff:
        step_ri('Loading in the base field')
        base_field, _, _, _ = load_raw_sim_data_chunks(use_field_diff)
        if inputs_sum_to_one:
            print('Making pixel values in the base field(s) sum to 1')
            base_field = sum_to_one(base_field, (1, 2))
        # All rows may not use the same base field
        use_field_diff_mapping = cli_args.get('use_field_diff_mapping')
        if use_field_diff_mapping:
            elements = len(use_field_diff_mapping)
            if elements % 3 != 0:
                terminate_with_message('Incorrect number of mapping arguments')
            for arg_idx in range(elements // 3):
                starting_arg = arg_idx * 3
                base_field_idx = use_field_diff_mapping[starting_arg]
                idx_low = use_field_diff_mapping[starting_arg + 1]
                idx_high = use_field_diff_mapping[starting_arg + 2]
                print(f'Using base field at index {base_field_idx} on '
                      f'rows {idx_low} - {idx_high}')
                input_data[idx_low:idx_high] -= base_field[base_field_idx]
            print('Creating an averaged base field that will be saved')
            base_field_idxs = np.array(use_field_diff_mapping[::3])
            base_field = np.sum(base_field[base_field_idxs], axis=0)
            base_field /= len(base_field_idxs)
            base_field = base_field[None]
        else:
            step_ri('Taking the difference between the inputs and base field')
            # Diff between the base field and each of the individual fields
            input_data -= base_field

    # ==========================================================================

    step_ri('Shuffling')
    random_shuffle_idxs = np.random.permutation(len(input_data))
    input_data = input_data[random_shuffle_idxs]
    output_data = output_data[random_shuffle_idxs]
    if train_only_mask is not None:
        train_only_mask = train_only_mask[random_shuffle_idxs]

        step_ri('Splitting apart the train only data')
        input_data_train_only = input_data[train_only_mask]
        output_data_train_only = output_data[train_only_mask]
        input_data = input_data[~train_only_mask]
        output_data = output_data[~train_only_mask]

    # ==========================================================================

    step_ri('Splitting')
    training_percentage = cli_args['training_percentage']
    validation_percentage = cli_args['validation_percentage']
    testing_percentage = cli_args['testing_percentage']
    psum = training_percentage + validation_percentage + testing_percentage
    if psum != 100:
        terminate_with_message(f'Percentages must add up to 100%, at {psum}%')
    # Add up the percentages for where the data will be split
    idxs = np.cumsum((training_percentage, validation_percentage)) / 100
    # Need to figure out how many rows this equates to
    idxs *= input_data.shape[0]
    # Convert to integers so the values can be split
    idxs = idxs.astype(int)
    train_inputs, val_inputs, test_inputs = np.split(input_data, idxs)
    train_outputs, val_outputs, test_outputs = np.split(output_data, idxs)
    if train_only_mask is not None:
        # Adding back in the train only datasets
        train_inputs = np.vstack((train_inputs, input_data_train_only))
        train_outputs = np.vstack((train_outputs, output_data_train_only))

    # Add back in the base field if it was removed and the field diff is not
    # being done
    if not use_field_diff and no_aber_input_row is not None:
        train_inputs = np.vstack((train_inputs, no_aber_input_row[None]))
        train_outputs = np.vstack((train_outputs, no_aber_output_row[None]))

    if use_field_diff:
        # Add a row with no aberrations to the training data
        train_inputs = np.vstack(
            (train_inputs, np.zeros_like(base_field)[None]))
        train_outputs = np.vstack(
            (train_outputs, np.zeros(len(zernike_terms))[None]))

    def _print_split(word, percentage, inputs):
        print(f'{word} percentage: {(percentage)}%, '
              f'rows: {inputs.shape[0]}')

    _print_split('Training', training_percentage, train_inputs)
    _print_split('Validation', validation_percentage, val_inputs)
    _print_split('Testing', testing_percentage, test_inputs)

    # ==========================================================================

    step_ri('Normalizing inputs')
    # Normalize between -1 and 1 if set to true
    nro = cli_args['norm_range_ones']
    norm_values = {NORM_RANGE_ONES_INPUT: nro, NORM_RANGE_ONES_OUTPUT: nro}
    if cli_args.get('disable_norm_inputs'):
        print('Not performing any input normalization')
    else:
        print('Globally normalizing inputs of training data')
        train_inputs, max_min_diff, min_x = find_min_max_norm(
            train_inputs, True, nro)
        # These will both be singular floats. In the future, if individual norm
        # is needed, then these two vars should be arrays instead of scalars.
        norm_values[INPUT_MIN_X] = min_x
        norm_values[INPUT_MAX_MIN_DIFF] = max_min_diff
        print('Normalizing inputs of validation/testing data based '
              'on training normalization values')
        val_inputs = min_max_norm(val_inputs, max_min_diff, min_x, nro)
        test_inputs = min_max_norm(test_inputs, max_min_diff, min_x, nro)

    # ==========================================================================

    step_ri('Normalizing outputs')
    output_normalization = True
    norm_outputs = cli_args.get('norm_outputs')
    if norm_outputs == 'individually':
        print('Individually normalizing outputs of training data')
        train_outputs, max_min_diff, min_x = find_min_max_norm(train_outputs,
                                                               ones_range=nro)
    elif norm_outputs == 'globally':
        print('Globally normalizing outputs of training data')
        train_outputs, max_min_diff, min_x = find_min_max_norm(train_outputs,
                                                               globally=True,
                                                               ones_range=nro)
        # For the output normalization, it is easier if there is a norm value
        # for every single element
        min_x = np.repeat(min_x, train_outputs.shape[1])
        max_min_diff = np.repeat(max_min_diff, train_outputs.shape[1])
    else:
        output_normalization = False
        print('Not performing any output normalization')
    if output_normalization is True:
        print('Normalizing outputs of validation data based on training '
              'normalization values')
        norm_values[OUTPUT_MIN_X] = min_x
        norm_values[OUTPUT_MAX_MIN_DIFF] = max_min_diff
        val_outputs = min_max_norm(val_outputs, max_min_diff, min_x, nro)

    # ==========================================================================

    step_ri('Creating new datasets')
    # Extra tables of information taken from the raw datafile
    extra_vars = {
        CAMERA_SAMPLING: camera_sampling,
        ZERNIKE_TERMS: zernike_terms,
        INPUTS_SUM_TO_ONE: inputs_sum_to_one,
        **norm_values,
    }
    # Need to save the base field so that it can be subtracted if necessary,
    # this is unnormalized and should be subtracted before normalization occurs
    if base_field is not None:
        extra_vars[BASE_INT_FIELD] = base_field

    def _create_dataset(cli_arg, inputs, outputs):
        out_path = f'{PROC_DATA_P}/{cli_args[cli_arg]}'
        print(f'Making {out_path}')
        make_dir(out_path)
        # Write out the CLI args that this script was called with
        save_cli_args(out_path, cli_args, 'preprocess_data_complete')
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
