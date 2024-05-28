"""
This script does not do much preprocessing, it simply ensures that the data can
be read in with the expected format.

No normalization is done in this script.

This script is mainly handy when one of the following arguments are passed to
the `sim_data` script:
    --fixed-amount-per-zernike
    --fixed-amount-per-zernike-range
"""

from utils.constants import (ARGS_F, CCD_INTENSITY, CCD_SAMPLING, DATA_F,
                             DS_RAW_INFO_F, INPUTS, OUTPUTS, PROC_DATA_P,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.json import json_write
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def preprocess_data_bare_parser(subparsers):
    """
    Example commands:
        python3 main.py preprocess_data_bare \
            fixed_50nm_range fixed_50nm_range_processed
    """
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


def preprocess_data_bare(cli_args):
    title('Preprocess data bare script')

    step_ri('Loading in data')
    raw_data_tag = cli_args['raw_data_tag']
    datafile_path = f'{RAW_SIMULATED_DATA_P}/{raw_data_tag}/{DATA_F}'
    print(f'Using path: {datafile_path}')
    print(datafile_path)
    data = read_hdf(datafile_path)
    # For our models, we will want to feed in our intensity fields and output
    # the associated Zernike polynomials
    input_data = data[CCD_INTENSITY][:]
    output_data = data[ZERNIKE_COEFFS][:]
    print(f'Input shape: {input_data.shape}')
    print(f'Output shape: {output_data.shape}')
    # Other data that will be written out for reference
    ccd_sampling = data[CCD_SAMPLING][()]
    zernike_terms = data[ZERNIKE_TERMS][:]

    step_ri('Adding in dimension for the channels')
    # Since this is a grayscale image, there is only one channel
    input_data = input_data[:, None, :, :]
    print(f'Input shape: {input_data.shape}')

    step_ri('Creating new dataset')
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
        # Write out the processed HDF file
        HDFWriteModule(f'{out_path}/{DATA_F}').create_and_write_hdf_simple({
            INPUTS: inputs,
            OUTPUTS: outputs,
        })

    _create_dataset('tag', input_data, output_data)
