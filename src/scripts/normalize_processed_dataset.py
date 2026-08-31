"""
This script is designed to be used after the preprocess_data_pol script.
The reason this script is separate is because all the raw data cannot be
preprocessed at once, but we want to normalize it all together.
"""

import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (DATA_F, EXTRA_VARS_F, INPUTS, INPUTS_Z_SCORE_MEAN,
                             INPUTS_Z_SCORE_STD, OUTPUTS, OUTPUTS_Z_SCORE_MEAN,
                             OUTPUTS_Z_SCORE_STD, PROC_DATA_P)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.norm import z_score_normalize
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def normalize_processed_dataset_parser(subparsers):
    subparser = subparsers.add_parser(
        'normalize_processed_dataset',
        help='normalize a processed dataset',
    )
    subparser.set_defaults(main=normalize_processed_dataset)
    subparser.add_argument(
        'new_output_tag',
        help='tag of the new datasets to create',
    )
    subparser.add_argument(
        'existing_data_tag',
        help='tag of the processed data to normalize',
    )
    subparser.add_argument(
        '--max-scale-inputs',
        action='store_true',
        help='perform max scaling on the inputs',
    )
    subparser.add_argument(
        '--z-score-norm-inputs-global',
        action='store_true',
        help='perform global z-score normalization on the inputs',
    )
    subparser.add_argument(
        '--z-score-norm-inputs',
        action='store_true',
        help='perform individual z-score normalization on the inputs',
    )
    subparser.add_argument(
        '--z-score-norm-outputs',
        action='store_true',
        help='perform individual z-score normalization on the outputs',
    )
    subparser.add_argument(
        '--use-existing-norm-vals',
        help=('use existing norm values from another processed dataset; '
              'the passed argument should specify the tag of the dataset '
              'to use the norm values from'),
    )


def normalize_processed_dataset(cli_args):
    title('Preprocess data pol script')

    def _get_tag_path(tag):
        return f'{PROC_DATA_P}/{tag}'

    step_ri('Creating the new processed dataset')
    new_output_tag = cli_args['new_output_tag']
    print(f'Tag: {new_output_tag}')
    new_output_path = _get_tag_path(new_output_tag)
    print(f'Making {new_output_path}')
    make_dir(new_output_path)
    # Write out the CLI args that this script was called with
    save_cli_args(new_output_path, cli_args, 'normalize_processed_dataset')

    step_ri('Loading in the processed dataset')
    existing_data_tag = cli_args['existing_data_tag']
    print(f'Tag: {existing_data_tag}')
    existing_data_path = _get_tag_path(existing_data_tag)
    # Load in the inputs and outputs that need to be normalized
    existing_data_obj = read_hdf(f'{existing_data_path}/{DATA_F}')
    input_data = existing_data_obj[INPUTS][:]
    output_data = existing_data_obj[OUTPUTS][:]
    print(f'Input data shape: {input_data.shape}')
    print(f'Output data shape: {output_data.shape}')
    # Load in all the extra variables so they can be written to the new dataset
    extra_vars_obj = read_hdf(f'{existing_data_path}/{EXTRA_VARS_F}')
    extra_vars = {key: value[()] for key, value in extra_vars_obj.items()}

    use_existing_norm_vals = cli_args.get('use_existing_norm_vals')
    if use_existing_norm_vals is not None:
        step_ri('Will use existing norm values')
        print(f'Using norm values from tag: {use_existing_norm_vals}')
        existing_norm_path = _get_tag_path(use_existing_norm_vals)
        existing_norm_ev = read_hdf(f'{existing_norm_path}/{EXTRA_VARS_F}')

    max_scale_inputs = cli_args['max_scale_inputs']
    if max_scale_inputs:
        step_ri('Max scaling input values')
        if use_existing_norm_vals:
            max_value = existing_norm_ev['max_value'][()]
        else:
            max_value = np.max(input_data)
        extra_vars['max_value'] = max_value
        input_data /= max_value
        print(f'Inputs min: {np.min(input_data)}')
        print(f'Inputs max: {np.max(input_data)}')

    z_score_norm_inputs_global = cli_args['z_score_norm_inputs_global']
    if z_score_norm_inputs_global:
        step_ri('Z-score normalizing input values globally')
        if use_existing_norm_vals:
            inputs_mean = existing_norm_ev[INPUTS_Z_SCORE_MEAN][:]
            inputs_std = existing_norm_ev[INPUTS_Z_SCORE_STD][:]
        else:
            inputs_mean = np.mean(input_data)
            inputs_std = np.std(input_data)
        extra_vars[INPUTS_Z_SCORE_MEAN] = inputs_mean
        extra_vars[INPUTS_Z_SCORE_STD] = inputs_std
        input_data = z_score_normalize(input_data, inputs_mean, inputs_std)
        print(f'Inputs min: {np.min(input_data)}')
        print(f'Inputs max: {np.max(input_data)}')

    z_score_norm_inputs = cli_args['z_score_norm_inputs']
    if z_score_norm_inputs:
        step_ri('Z-score normalizing input values')
        if use_existing_norm_vals:
            inputs_mean = existing_norm_ev[INPUTS_Z_SCORE_MEAN][:]
            inputs_std = existing_norm_ev[INPUTS_Z_SCORE_STD][:]
        else:
            inputs_mean = np.mean(input_data, axis=0)
            inputs_std = np.std(input_data, axis=0)
        extra_vars[INPUTS_Z_SCORE_MEAN] = inputs_mean
        extra_vars[INPUTS_Z_SCORE_STD] = inputs_std
        input_data = z_score_normalize(input_data, inputs_mean, inputs_std)
        print(f'Inputs min: {np.min(input_data)}')
        print(f'Inputs max: {np.max(input_data)}')

    z_score_norm_outputs = cli_args['z_score_norm_outputs']
    if z_score_norm_outputs:
        step_ri('Z-score normalizing output values')
        if use_existing_norm_vals:
            outputs_mean = existing_norm_ev[OUTPUTS_Z_SCORE_MEAN][:]
            outputs_std = existing_norm_ev[OUTPUTS_Z_SCORE_STD][:]
        else:
            outputs_mean = np.mean(output_data, axis=0)
            outputs_std = np.std(output_data, axis=0)
        extra_vars[OUTPUTS_Z_SCORE_MEAN] = outputs_mean
        extra_vars[OUTPUTS_Z_SCORE_STD] = outputs_std
        output_data = z_score_normalize(output_data, outputs_mean, outputs_std)
        print(f'Outputs min: {np.min(output_data)}')
        print(f'Outputs max: {np.max(output_data)}')

    step_ri('Writing out the normalized data')
    datafile_path = f'{new_output_path}/{DATA_F}'
    print(f'Writing out to {datafile_path}')
    HDFWriteModule(datafile_path).create_and_write_hdf_simple({
        INPUTS: input_data,
        OUTPUTS: output_data,
    })
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    ev_path = f'{new_output_path}/{EXTRA_VARS_F}'
    HDFWriteModule(ev_path).create_and_write_hdf_simple(extra_vars)
