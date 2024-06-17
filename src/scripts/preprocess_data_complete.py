"""
This script preprocesses datasets that can be used for training and testing.
The datasets should have been simulated by the `sim_data` script.

Three different datasets will be outputted: training, validation, and testing.
Old datasets will be overwritten if they already exist.

In this script, all inputs are normalized based on the training normalization
values. The output normalization for the training and validation data is
optional and can either be global or individual. If output normalization is
performed, it will be based on the training normalization values.
"""

import numpy as np
from utils.constants import (ARGS_F, BASE_INT_FIELD, CCD_SAMPLING, DATA_F,
                             EXTRA_VARS_F, INPUTS, INPUT_MIN_X,
                             INPUT_MAX_MIN_DIFF, NORM_F, OUTPUTS, OUTPUT_MIN_X,
                             OUTPUT_MAX_MIN_DIFF, PROC_DATA_P, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.norm import find_min_max_norm, min_max_norm
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message

OUTPUT_NORM_OPTIONS = ['globally', 'individually']


def preprocess_data_complete_parser(subparsers):
    """
    Example commands:
        python3 main.py preprocess_data_complete \
            fixed_10nm \
            train_fixed_10nm_gl val_fixed_10nm_gl test_fixed_10nm_gl \
            75 10 15 \
            --norm-outputs globally
        python3 main.py preprocess_data_complete \
            random_50nm_single \
            train_rand_50nm_s_gl val_rand_50nm_s_gl test_rand_50nm_s_gl \
            75 10 15 \
            --norm-outputs globally
    """
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
        default=7,
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
        '--norm-outputs',
        choices=OUTPUT_NORM_OPTIONS,
        help=('normalize training and validation output values either '
              'globally or individually'),
    )
    subparser.add_argument(
        '--use-field-diff',
        action='store_true',
        help=('for the intensity fields (the inputs), take the difference '
              'between the base field (no aberrations) and each aberrated '
              'field; this requires that the last row of the raw simulated '
              'data contains no aberrations'),
    )
    subparser.add_argument(
        '--additional-raw-data-tags',
        nargs='*',
        help='additional raw simulated data to preprocess and merge together',
    )


def preprocess_data_complete(cli_args):
    title('Preprocess data complete script')

    step_ri('Loading in data chunks')
    (input_data, output_data, zernike_terms,
     ccd_sampling) = load_raw_sim_data_chunks(cli_args['raw_data_tag'])
    for tag in cli_args.get('additional_raw_data_tags') or []:
        (additional_input_data, additional_output_data, _,
         _) = load_raw_sim_data_chunks(tag)
        # Keep stacking on the other datasets
        input_data = np.vstack((input_data, additional_input_data))
        output_data = np.vstack((output_data, additional_output_data))
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    # Check if the last row contains the base field
    base_field = None
    no_aber_output_row = None
    if np.all(output_data[-1] == 0):
        # Chop off the input and output base field
        base_field = input_data[-1]
        no_aber_output_row = output_data[-1]
        input_data = input_data[:-1]
        output_data = output_data[:-1]

    if cli_args['use_field_diff']:
        step_ri('Taking the difference between the inputs and the base field')
        if no_aber_output_row is None:
            terminate_with_message('Last row not aberration free')
        # Take the diff between the base field and each of the individual fields
        input_data = input_data - base_field
    elif no_aber_output_row is not None:
        step_ri('Last row has zeros for all Zernike coefficients')
        print('This row has no aberrations')
        print('Removing row so that it can be added to the training dataset')

    step_ri('Shuffling')
    random_shuffle_idxs = np.random.permutation(len(input_data))
    input_data = input_data[random_shuffle_idxs]
    output_data = output_data[random_shuffle_idxs]

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

    # Add back in the base field if it was removed and the field diff is not
    # being done
    if no_aber_output_row is not None and not cli_args['use_field_diff']:
        train_inputs = np.vstack((train_inputs, base_field[None]))
        train_outputs = np.vstack((train_outputs, no_aber_output_row[None]))

    def _print_split(word, percentage, inputs):
        print(f'{word} percentage: {(percentage)}%, '
              f'rows: {inputs.shape[0]}')

    _print_split('Training', training_percentage, train_inputs)
    _print_split('Validation', validation_percentage, val_inputs)
    _print_split('Testing', testing_percentage, test_inputs)

    step_ri('Normalizing inputs')
    norm_values = {}
    print('Globally normalizing inputs of training data')
    train_inputs, max_min_diff, min_x = find_min_max_norm(train_inputs, True)
    # These will both be singular floats. If individual input normalization
    # could be handy and needs to be added, then these should always be arrays
    norm_values[INPUT_MIN_X] = min_x
    norm_values[INPUT_MAX_MIN_DIFF] = max_min_diff
    print('Normalizing inputs of validation/testing data based on training '
          'normalization values')
    val_inputs = min_max_norm(val_inputs, max_min_diff, min_x)
    test_inputs = min_max_norm(test_inputs, max_min_diff, min_x)

    step_ri('Normalizing outputs')
    output_normalization = True
    norm_outputs = cli_args.get('norm_outputs')
    if norm_outputs == 'individually':
        print('Individually normalizing outputs of training data')
        train_outputs, max_min_diff, min_x = find_min_max_norm(train_outputs)
    elif norm_outputs == 'globally':
        print('Globally normalizing outputs of training data')
        train_outputs, max_min_diff, min_x = find_min_max_norm(
            train_outputs, True)
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
        val_outputs = min_max_norm(val_outputs, max_min_diff, min_x)

    step_ri('Creating new datasets')
    # Extra tables of information taken from the raw datafile
    extra_vars = {
        CCD_SAMPLING: ccd_sampling,
        ZERNIKE_TERMS: zernike_terms,
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
        json_write(f'{out_path}/{ARGS_F}', cli_args)
        # Add a file with other necessary variables
        HDFWriteModule(f'{out_path}/{EXTRA_VARS_F}'
                       ).create_and_write_hdf_simple(extra_vars)
        # Add the file with the normalization input and output values
        json_write(f'{out_path}/{NORM_F}', norm_values)
        # Write out the processed HDF file
        HDFWriteModule(f'{out_path}/{DATA_F}').create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })

    _create_dataset('training_tag', train_inputs, train_outputs)
    _create_dataset('validation_tag', val_inputs, val_outputs)
    _create_dataset('testing_tag', test_inputs, test_outputs)
