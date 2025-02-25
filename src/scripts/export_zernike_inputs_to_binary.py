"""
Will export aberration rows in CSV format to binary so that they can be run on
the PICTURE-D instrument to obtain real data. The CSV data must be created with
the `--save-aberrations-csv` arg in the `sim_data` script, it will have the file
name of the `ABERRATIONS_F` constant.
"""

import numpy as np
from utils.constants import (ABERRATIONS_F, ARGS_F, BINARY_DATA_F,
                             RAW_SIMULATED_DATA_P)
from utils.json import json_write
from utils.load_raw_sim_data import load_raw_sim_data_aberrations_file
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def export_zernike_inputs_to_binary_parser(subparsers):
    subparser = subparsers.add_parser(
        'export_zernike_inputs_to_binary',
        help='export aberration CSV files to binary',
    )
    subparser.set_defaults(main=export_zernike_inputs_to_binary)
    subparser.add_argument(
        'binary_data_tag',
        help='tag of the binary data, will be prefixed with `binary_`',
    )
    subparser.add_argument(
        '--append-no-aberrations-row',
        action='store_true',
        help='add a row with no aberrations',
    )
    subparser.add_argument(
        '--simulated-data-tags',
        nargs='+',
        help='tags of the simulated datasets',
    )


def export_zernike_inputs_to_binary(cli_args):
    title('Export zernike inputs to binary script')

    step_ri('Loading in aberration files')
    all_aberrations = []
    for simulated_data_tag in cli_args['simulated_data_tags']:
        print(f'Loading in {simulated_data_tag}')
        try:
            aberrations, zernike_terms = load_raw_sim_data_aberrations_file(
                simulated_data_tag)
        except Exception:
            terminate_with_message(
                f'{ABERRATIONS_F} missing in {simulated_data_tag}')
        print(f'Adding {len(aberrations)} rows')
        all_aberrations.extend(aberrations)

    if cli_args['append_no_aberrations_row']:
        step_ri('Appending no aberrations row')
        all_aberrations.append(np.zeros_like(zernike_terms))

    step_ri('Data that will be written out')
    all_aberrations = np.array(all_aberrations)
    print(f'Aberrations shape: {all_aberrations.shape}')
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Creating the output directory')
    output_binary_tag = cli_args['binary_data_tag']
    out_path = f'{RAW_SIMULATED_DATA_P}/binary_{output_binary_tag}'
    print(f'Making {out_path}')
    make_dir(out_path)

    step_ri('Writing out CLI args')
    # Write out the CLI args that this script was called with
    json_write(f'{out_path}/{ARGS_F}', cli_args)

    step_ri('Writing out binary data')
    with open(f'{out_path}/{BINARY_DATA_F}', 'wb') as file:
        for row in all_aberrations:
            file.write(row.tobytes())
