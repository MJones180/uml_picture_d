"""
This script expects a wf intensity as input and Zernike coefficients as output.

This script does not do much preprocessing, it simply ensures that the data can
be read in with the expected format.

No normalization is done in this script.

This script is mainly handy when one of the following arguments are passed to
the `sim_data` script:
    --fixed-amount-per-zernike
    --fixed-amount-per-zernike-range

Note: some code is shared with the `preprocess_data_complete` script (hardcoded)
"""

from utils.cli_args import save_cli_args
from utils.constants import (CAMERA_SAMPLING, DATA_F, EXTRA_VARS_F, INPUTS,
                             OUTPUTS, PROC_DATA_P, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def preprocess_data_bare_parser(subparsers):
    subparser = subparsers.add_parser(
        'preprocess_data_bare',
        help='preprocess data for training, validation, and testing',
    )
    subparser.set_defaults(main=preprocess_data_bare)
    subparser.add_argument(
        'raw_data_tag',
        help='tag of the raw simulated data',
    )
    subparser.add_argument(
        'tag',
        help='tag of the outputted dataset',
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


def preprocess_data_bare(cli_args):
    title('Preprocess data bare script')

    step_ri('Loading in data chunks')
    raw_data_tag = cli_args['raw_data_tag']
    (input_data, output_data, zernike_terms,
     camera_sampling) = load_raw_sim_data_chunks(raw_data_tag)
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    if cli_args['outputs_in_surface_error']:
        step_ri('Converting from surface error to wavefront error')
        print('Multiplying output data (Zernike coefficients) by 2')
        output_data *= 2

    outputs_scaling_factor = cli_args.get('outputs_scaling_factor')
    if outputs_scaling_factor:
        step_ri('Adding a scaling factor to the outputs')
        print(f'Multiplying output data by {outputs_scaling_factor}')
        output_data *= outputs_scaling_factor

    step_ri('Creating new dataset')
    # Extra tables of information taken from the raw datafile
    extra_vars = {
        CAMERA_SAMPLING: camera_sampling,
        ZERNIKE_TERMS: zernike_terms,
    }

    def _create_dataset(cli_arg, inputs, outputs):
        out_path = f'{PROC_DATA_P}/{cli_args[cli_arg]}'
        print(f'Making {out_path}')
        make_dir(out_path)
        # Write out the CLI args that this script was called with
        save_cli_args(out_path, cli_args, 'preprocess_data_bare')
        # Add a file with other necessary variables
        HDFWriteModule(f'{out_path}/{EXTRA_VARS_F}'
                       ).create_and_write_hdf_simple(extra_vars)
        # Write out the processed HDF file
        HDFWriteModule(f'{out_path}/{DATA_F}').create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })

    _create_dataset('tag', input_data, output_data)
