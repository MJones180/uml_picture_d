"""
Randomly select rows from a raw dataset and create a new raw dataset.
Used to shrink down the number of rows in a large dataset.
"""

from glob import glob
import numpy as np
from utils.constants import (ARGS_F, CAMERA_INTENSITY, CAMERA_SAMPLING, DATA_F,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.json import json_load, json_write
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def random_trim_raw_dataset_parser(subparsers):
    subparser = subparsers.add_parser(
        'random_trim_raw_dataset',
        help='trim down the number of rows in a dataset',
    )
    subparser.set_defaults(main=random_trim_raw_dataset)
    subparser.add_argument(
        'original_tag',
        help='tag of the original raw simulated dataset',
    )
    subparser.add_argument(
        'new_tag',
        help='tag of the new raw simulated dataset with less rows',
    )
    subparser.add_argument(
        'row_count',
        type=int,
        help='number of rows in the new dataset',
    )


def random_trim_raw_dataset(cli_args):
    title('Randomly trim raw dataset script')

    original_tag = cli_args['original_tag']
    new_tag = cli_args['new_tag']
    row_count = cli_args['row_count']

    # The tables that need to be merged together from all the datafiles
    camera_intensity = []
    camera_sampling = None
    zernike_coeffs = []
    zernike_terms = None

    # Loop through all the raw datafiles and combine them
    step_ri('Loading in the data')
    in_dir = f'{RAW_SIMULATED_DATA_P}/{original_tag}'
    for path in glob(f'{in_dir}/*_{DATA_F}'):
        print(f'Path: {path}')
        data = read_hdf(path)
        # Tables that are the same between all datafiles
        if camera_sampling is None:
            camera_sampling = data[CAMERA_SAMPLING][()]
            zernike_terms = data[ZERNIKE_TERMS][:]
        camera_intensity.extend(data[CAMERA_INTENSITY][:])
        zernike_coeffs.extend(data[ZERNIKE_COEFFS][:])

    # Turn the data back into numpy arrays
    camera_intensity = np.array(camera_intensity)
    zernike_coeffs = np.array(zernike_coeffs)

    # Pick the random rows that will be chosen
    step_ri('Randomly picking the rows')
    rng = np.random.default_rng()
    idxs = rng.choice(camera_intensity.shape[0], row_count, replace=False)
    # Trim down the rows in the datasets
    camera_intensity = camera_intensity[idxs]
    zernike_coeffs = zernike_coeffs[idxs]

    # Create the new output directory
    step_ri('Creating output directory')
    out_dir = f'{RAW_SIMULATED_DATA_P}/{new_tag}'
    make_dir(out_dir)

    # Write out the new data
    out_path = f'{out_dir}/0_{DATA_F}'
    step_ri('Writing new datafile')
    print(f'Path: {out_path}')
    HDFWriteModule(out_path).create_and_write_hdf_simple({
        CAMERA_INTENSITY: camera_intensity,
        CAMERA_SAMPLING: camera_sampling,
        ZERNIKE_COEFFS: zernike_coeffs,
        ZERNIKE_TERMS: zernike_terms,
    })

    # Load in the args file
    step_ri('Writing to the tag lookup')
    args_data = json_load(f'{in_dir}/{ARGS_F}')
    # Add in which datafile the rows are being taken from
    args_data['trimmed_from_tag'] = original_tag
    json_write(f'{out_dir}/{ARGS_F}', args_data)
