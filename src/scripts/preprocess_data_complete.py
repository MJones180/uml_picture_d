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

from glob import glob
import numpy as np
from utils.constants import (ARGS_F, CCD_INTENSITY, CCD_SAMPLING, DATA_F,
                             DS_RAW_INFO_F, INPUTS, INPUT_MIN_X,
                             INPUT_MAX_MIN_DIFF, NORM_F, OUTPUTS, OUTPUT_MIN_X,
                             OUTPUT_MAX_MIN_DIFF, PROC_DATA_P,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.json import json_write
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
            0.75 0.10 0.15 \
            --norm-outputs globally
        python3 main.py preprocess_data_complete \
            random_50nm_single \
            train_rand_50nm_s_gl val_rand_50nm_s_gl test_rand_50nm_s_gl \
            0.75 0.10 0.15 \
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
        type=float,
        default=0.7,
        help='percentage of the data that will go to training',
    )
    subparser.add_argument(
        'validation_percentage',
        type=float,
        default=0.15,
        help='percentage of the data that will go to validation',
    )
    subparser.add_argument(
        'testing_percentage',
        type=float,
        default=0.15,
        help='percentage of the data that will go to testing',
    )
    subparser.add_argument(
        '--norm-outputs',
        choices=OUTPUT_NORM_OPTIONS,
        help=('normalize training and validation output values either '
              'globally or individually'),
    )


def preprocess_data_complete(cli_args):
    title('Preprocess data complete script')

    step_ri('Loading in data chunks')
    raw_data_tag = cli_args['raw_data_tag']
    base_path = f'{RAW_SIMULATED_DATA_P}/{raw_data_tag}'
    # Instead of globbing the paths, it is safer to load in the datafiles using
    # their chunk number so that they are guaranteed to be in order
    chunk_vals = sorted([
        # Grab the number associated with each chunk
        int(path.split('/')[-1][:-len(DATA_F) - 1])
        # All datafiles should follow the format [chunk]_[DATA_F]
        for path in glob(f'{base_path}/*_{DATA_F}')
    ])
    input_data = []
    output_data = []
    for idx, chunk_val in enumerate(chunk_vals):
        path = f'{base_path}/{chunk_val}_{DATA_F}'
        print(f'Path: {path}')
        data = read_hdf(path)
        # For our models, we will want to feed in our intensity fields and
        # output the associated Zernike polynomials
        input_data.extend(data[CCD_INTENSITY][:])
        output_data.extend(data[ZERNIKE_COEFFS][:])
        # This data will be the same across all chunks, so only read it once
        if idx == 0:
            # Other data that will be written out for reference
            ccd_sampling = data[CCD_SAMPLING][()]
            zernike_terms = data[ZERNIKE_TERMS][:]
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    step_ri('Shuffling')
    random_shuffle_idxs = np.random.permutation(len(input_data))
    input_data = input_data[random_shuffle_idxs]
    output_data = output_data[random_shuffle_idxs]

    step_ri('Splitting')
    training_percentage = cli_args['training_percentage']
    validation_percentage = cli_args['validation_percentage']
    testing_percentage = cli_args['testing_percentage']
    if training_percentage + validation_percentage + testing_percentage != 1:
        terminate_with_message('Percentages must add up to 100%')
    # Add up the percentages for where the data will be split
    idxs = np.cumsum((training_percentage, validation_percentage))
    # Need to figure out how many rows this equates to
    idxs *= input_data.shape[0]
    # Convert to integers so the values can be split
    idxs = idxs.astype(int)
    train_inputs, val_inputs, test_inputs = np.split(input_data, idxs)
    train_outputs, val_outputs, test_outputs = np.split(output_data, idxs)

    def _print_split(word, percentage, inputs):
        print(f'{word} percentage: {(percentage * 100)}%, '
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
    ds_raw_info = {
        # This is likely a small float, so write it as a string
        CCD_SAMPLING: ccd_sampling,
        ZERNIKE_TERMS: zernike_terms,
    }

    def _create_dataset(cli_arg, inputs, outputs):
        out_path = f'{PROC_DATA_P}/{cli_args[cli_arg]}'
        print(f'Making {out_path}')
        make_dir(out_path)
        # Write out the CLI args that this script was called with
        json_write(f'{out_path}/{ARGS_F}', cli_args)
        # Add a file with unused data from the raw dataset
        json_write(f'{out_path}/{DS_RAW_INFO_F}', ds_raw_info)
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
