"""
Will export aberration rows in CSV format to binary so that they can be run on
the PICTURE-D instrument to obtain real data. The CSV data must be created with
the `--save-aberrations-csv` arg in the `sim_data` script and have the file name
of the `ABERRATIONS_F` constant. The outputted files will be in binary with the
Zernike coefficients in (n or nm) RMS error.
"""

import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import ABERRATIONS_F, BINARY_DATA_F, RAW_DATA_P
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
        '--put-in-nm',
        action='store_true',
        help='put in nm RMS error instead of m RMS error',
    )
    subparser.add_argument(
        '--put-in-single-precision',
        action='store_true',
        help='put in single precision instead of double precision',
    )
    subparser.add_argument(
        '--simulated-data-tags',
        nargs='+',
        help='tags of the simulated datasets',
    )
    subparser.add_argument(
        '--output-chunk-size',
        type=int,
        help='number of rows per chunk to break the output binary files into',
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

    if cli_args['put_in_nm']:
        step_ri('Putting data in nm RMS error')
        print('Going from m RMS error to nm RMS error (*1e9)')
        all_aberrations *= 1e9

    if cli_args['put_in_single_precision']:
        step_ri('Putting data in single precision')
        print('Going from double (64 bit) to single (32 bit) precision')
        all_aberrations = all_aberrations.astype(np.float32)

    step_ri('Creating the output directory')
    output_binary_tag = cli_args['binary_data_tag']
    out_path = f'{RAW_DATA_P}/binary_{output_binary_tag}'
    print(f'Making {out_path}')
    make_dir(out_path)

    step_ri('Writing out CLI args')
    save_cli_args(out_path, cli_args, 'export_zernike_inputs_to_binary')

    step_ri('Writing out binary data')
    current_idx = 0

    def _write_binary_data(data):
        nonlocal current_idx
        path = f'{out_path}/{current_idx}_{BINARY_DATA_F}'
        current_idx += 1
        print(f'Writing out {len(data)} rows to {path}')
        with open(path, 'wb') as file:
            file.write(data.tobytes())

    chunk_size = cli_args['output_chunk_size']
    if chunk_size:
        print(f'Will create chunks of {chunk_size} rows')
        while all_aberrations.shape[0] > chunk_size:
            data_chunk = all_aberrations[:chunk_size]
            all_aberrations = all_aberrations[chunk_size:]
            _write_binary_data(data_chunk)
    _write_binary_data(all_aberrations)
