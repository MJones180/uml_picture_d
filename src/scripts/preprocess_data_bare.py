"""
This script does not do much preprocessing, it simply ensures that the data can
be read in with the expected format.

No normalization is done in this script.

This script is mainly handy when one of the following arguments are passed to
the `sim_data` script:
    --fixed-amount-per-zernike
    --fixed-amount-per-zernike-range

Note: some code is shared with the `preprocess_data_complete` script (hardcoded)
"""

from glob import glob
import numpy as np
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
