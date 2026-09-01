import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, MODES, RAW_DATA_P
from utils.create_zernike_mode import create_zernike_mode
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def create_zernike_basis_modes_parser(subparsers):
    subparser = subparsers.add_parser(
        'create_zernike_basis_modes',
        help='create a set of zernike basis modes',
    )
    subparser.set_defaults(main=create_zernike_basis_modes)
    subparser.add_argument(
        'output_tag',
        help='tag to give to the modes; will be put in the raw data dir',
    )
    subparser.add_argument(
        'number_modes',
        type=int,
        help='number of basis modes to create',
    )
    subparser.add_argument(
        'number_pixels',
        type=int,
        help='number of pixels in the grid (one axis only)',
    )
    subparser.add_argument(
        '--apply-mask',
        nargs=2,
        help=('apply a mask from a raw dataset; the first row will be used '
              'to mask the data; two args expected: raw tag name, table name'),
    )


def create_zernike_basis_modes(cli_args):
    title('Create zernike basis modes script')

    step_ri('Computing Zernike modes')
    modes = []
    numb_modes = cli_args['number_modes']
    print(f'Will compute Noll modes 2 - {numb_modes+1} (skipping piston)')
    pixels = cli_args['number_pixels']
    print(f'Zernikes will go on grid {pixels}x{pixels}')
    for mode_idx in range(numb_modes):
        modes.append(create_zernike_mode(mode_idx + 2, pixels))
    modes = np.array(modes)
    print(f'Modes shape: {modes.shape}')

    apply_mask = cli_args.get('apply_mask')
    if apply_mask:
        step_ri('Apply mask to the data')
        mask_tag, table_name = apply_mask
        mask_path = raw_sim_data_chunk_paths(mask_tag)[0]
        print(f'Loading in the mask from {mask_path} ({table_name})')
        mask = read_hdf(mask_path)[table_name][:][0] != 0
        print(f'Mask shape: {mask.shape}')
        modes = modes[:, mask]
        print(f'Modes shape: {modes.shape}')

    step_ri('Performing QR decomposition on the modes')
    # Gives each mode a zero mean and unit norm; makes orthonormal modes
    Q, R = np.linalg.qr(modes.T, mode='reduced')
    modes = Q.T

    step_ri('Writing out modes')
    output_tag = cli_args['output_tag']
    print(f'Output tag: {output_tag}')
    out_dir = f'{RAW_DATA_P}/{output_tag}'
    print(f'Creating {out_dir}')
    make_dir(out_dir)
    # Write out the CLI args that this script was called with
    save_cli_args(out_dir, cli_args, 'create_zernike_basis_modes')
    datafile_path = f'{out_dir}/0_{DATA_F}'
    print(f'Path: {datafile_path}')
    HDFWriteModule(datafile_path).create_and_write_hdf_simple({MODES: modes})
