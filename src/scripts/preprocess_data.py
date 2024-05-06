"""
This script preprocesses datasets that can be used for training and testing.

Three different datasets will be outputted: training, validation, and testing.
Old datasets will be overwritten if they already exist.

In this script, all inputs are normalized based on the training normalization
values. The output normalization for the training and validation data is
optional and can either be global or individual. If output normalization is
performed, it will be based on the training normalization values.
"""

from astropy.io import fits
from glob import glob
import numpy as np
from utils.constants import (DATA_F, INPUTS, INPUT_MIN_X, INPUT_MAX_MIN_DIFF,
                             NORM_F, OUTPUTS, OUTPUT_MIN_X,
                             OUTPUT_MAX_MIN_DIFF, PROC_DATA_P, RAW_FITS_DATA_P)
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.norm import find_min_max_norm, min_max_norm
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def preprocess_data_parser(subparsers):
    """
    Example commands:
    python3 main.py preprocess_data \
        --raw-fits-data-dirs picture_c_llowfs_data_03 picture_c_llowfs_data_05 \
        --training-dir training_03_05_global \
        --validation-dir validation_03_05_global \
        --testing-dir testing_03_05_global \
        --training-percentage 0.75 \
        --validation-percentage 0.10 \
        --testing-percentage 0.15 \
        --norm-outputs-globally
    python3 main.py preprocess_data \
        --raw-fits-data-dirs picture_c_llowfs_data_03 picture_c_llowfs_data_05 \
        --training-dir training_03_05_ind \
        --validation-dir validation_03_05_ind \
        --testing-dir testing_03_05_ind \
        --training-percentage 0.75 \
        --validation-percentage 0.10 \
        --testing-percentage 0.15 \
        --norm-outputs-individually
    """
    subparser = subparsers.add_parser(
        'preprocess_data',
        help='preprocess data for training + testing',
    )
    subparser.set_defaults(main=preprocess_data)
    subparser.add_argument(
        '--raw-fits-data-dirs',
        nargs='+',
        help=('name of the directories containing the raw FITS data, all '
              'data will all be merged together'),
    )
    subparser.add_argument(
        '--training-dir',
        help='name of the training dataset',
    )
    subparser.add_argument(
        '--validation-dir',
        help='name of the validation dataset',
    )
    subparser.add_argument(
        '--testing-dir',
        help='name of the testing dataset',
    )
    subparser.add_argument(
        '--training-percentage',
        type=float,
        default=0.7,
        help='percentage of the data that will go to training',
    )
    subparser.add_argument(
        '--validation-percentage',
        type=float,
        default=0.15,
        help='percentage of the data that will go to validation',
    )
    subparser.add_argument(
        '--testing-percentage',
        type=float,
        default=0.15,
        help='percentage of the data that will go to testing',
    )
    subparser.add_argument(
        '--norm-outputs-globally',
        action='store_true',
        help='normalize output values globally',
    )
    subparser.add_argument(
        '--norm-outputs-individually',
        action='store_true',
        help='normalize output values individually',
    )


def preprocess_data(cli_args):
    title('Preprocess data script')

    step_ri('Loading in data')
    raw_fits_data_dirs = cli_args['raw_fits_data_dirs']
    input_data = []
    output_data = []
    # Loop through each of the raw datasets
    for raw_fits_data_dir in raw_fits_data_dirs:
        raw_fits_data_path = f'{RAW_FITS_DATA_P}/{raw_fits_data_dir}'
        print(f'Loading data from {raw_fits_data_path}')
        fits_paths = sorted(glob(f'{raw_fits_data_path}/*.fits'))
        for input_image_path in fits_paths:
            # Use of `memmap=False` prevents the following exception:
            #    OSError: [Errno 24] Too many open
            # By doing this, everything must fit in memory, so if there is a
            # lot of data, things will need to be reworked
            input_data.append(fits.getdata(input_image_path, memmap=False))
        csv_path = glob(f'{raw_fits_data_path}/*input*.csv')[0]
        csv_data = np.genfromtxt(csv_path, delimiter=',')
        # Chop off the ID column
        csv_data = csv_data[:, 1:]
        # Here, the CSV file contains the inputs values that were used to
        # generate the different FITS images. However, they will be our outputs
        # since we want the nn to determine what they are.
        output_data.extend(csv_data)
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')

    step_ri('Reshaping inputs')
    print('Removing empty values (hardcoded)')
    # Remove the first column and row since all they contain is zero
    input_data = input_data[:, 1:, 1:]
    print(f'Input shape: {input_data.shape}')
    print('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    step_ri('Shuffling')
    random_shuffle_idxs = np.random.permutation(len(input_data))
    input_data = input_data[random_shuffle_idxs]
    output_data = output_data[random_shuffle_idxs]

    step_ri('Splitting')
    training_frac = cli_args['training_percentage']
    print(f'Training Frac: {training_frac}')
    validation_frac = cli_args['validation_percentage']
    print(f'Validation Frac: {validation_frac}')
    testing_frac = cli_args['testing_percentage']
    print(f'Testing Frac: {testing_frac}')
    if training_frac + validation_frac + testing_frac != 1:
        raise ValueError('Percentages must equal 100%')
    total_rows = input_data.shape[0]
    idx_br_1 = int(training_frac * total_rows)
    idx_br_2 = idx_br_1 + int(validation_frac * total_rows)
    training_inputs = input_data[:idx_br_1]
    training_outputs = output_data[:idx_br_1]
    print(f'Training inputs shape: {training_inputs.shape}')
    print(f'Training outputs shape: {training_outputs.shape}')
    validation_inputs = input_data[idx_br_1:idx_br_2]
    validation_outputs = output_data[idx_br_1:idx_br_2]
    print(f'Validation inputs shape: {validation_inputs.shape}')
    print(f'Validation outputs shape: {validation_outputs.shape}')
    testing_inputs = input_data[idx_br_2:]
    testing_outputs = output_data[idx_br_2:]
    print(f'Testing inputs shape: {testing_inputs.shape}')
    print(f'Testing outputs shape: {testing_outputs.shape}')

    step_ri('Normalizing')
    norm_values = {}
    print('Globally normalizing inputs of training data')
    training_inputs, max_min_diff, min_x = find_min_max_norm(
        training_inputs, True)
    # These will both be singular floats. If individual input normalization
    # could be handy, then these should always be arrays
    norm_values[INPUT_MIN_X] = min_x
    norm_values[INPUT_MAX_MIN_DIFF] = max_min_diff
    print('Normalizing inputs of validation/testing data based on training '
          'normalization values')
    validation_inputs = min_max_norm(validation_inputs, max_min_diff, min_x)
    testing_inputs = min_max_norm(testing_inputs, max_min_diff, min_x)
    output_normalization = True
    if cli_args['norm_outputs_individually'] is True:
        print('Individually normalizing outputs of training data')
        training_outputs, max_min_diff, min_x = find_min_max_norm(
            training_outputs)
    elif cli_args['norm_outputs_globally'] is True:
        print('Globally normalizing outputs of training data')
        training_outputs, max_min_diff, min_x = find_min_max_norm(
            training_outputs, True)
        # For the output normalization, it is easier if there is a norm value
        # for every single element
        min_x = np.repeat(min_x, training_outputs.shape[1])
        max_min_diff = np.repeat(max_min_diff, training_outputs.shape[1])
    else:
        output_normalization = False
        print('Not performing any output normalization')
    if output_normalization is True:
        print('Normalizing outputs of validation data based on training '
              'normalization values...')
        norm_values[OUTPUT_MIN_X] = min_x
        norm_values[OUTPUT_MAX_MIN_DIFF] = max_min_diff
        validation_outputs = min_max_norm(validation_outputs, max_min_diff,
                                          min_x)

    step_ri('Creating new datasets')

    def _create_dataset(cli_arg, inputs, outputs):
        out_path = f'{PROC_DATA_P}/{cli_args[cli_arg]}'
        print(f'Making {out_path}')
        make_dir(out_path)
        # Add the file with the normalization input and output values
        json_write(f'{out_path}/{NORM_F}', norm_values)

        # Write out the processed HDF file
        HDFWriteModule(f'{out_path}/{DATA_F}').create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })

    _create_dataset('training_dir', training_inputs, training_outputs)
    _create_dataset('validation_dir', validation_inputs, validation_outputs)
    _create_dataset('testing_dir', testing_inputs, testing_outputs)
