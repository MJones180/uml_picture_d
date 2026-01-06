"""
This script will convert a FITS file which contains a darkhole response matrix.
The response matrix will be saved as HDF so that it can be accessed easier.
The `batch.export_dh_rm.pro` script must first be run in `piccsim`.
"""

from astropy.io import fits
from utils.constants import RESPONSE_MATRICES_P, RESPONSE_MATRIX_INV
from utils.hdf_read_and_write import HDFWriteModule
from utils.printing_and_logging import step_ri, title


def convert_dh_rm_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_dh_rm',
        help='convert and merge FITS datafiles from piccsim to HDF',
    )
    subparser.set_defaults(main=convert_dh_rm)
    subparser.add_argument(
        'tag',
        help='tag to give the response matrix',
    )
    subparser.add_argument(
        'abs_path',
        help='absolute path to the FITS datafile',
    )


def convert_dh_rm(cli_args):
    title('Convert dh rm script')

    step_ri('Loading in the response matrix')
    fits_path = cli_args['abs_path']
    print(f'Path: {fits_path}')
    # Transpose the array so that the DM actuators are the second dimension
    fits_data = fits.getdata(fits_path).T
    print(f'Data shape: {fits_data.shape}')

    step_ri('Saving the response matrix')
    tag = cli_args['tag']
    print(f'Tag: {tag}')
    output_path = f'{RESPONSE_MATRICES_P}/{tag}.h5'
    print(f'Outputting to {output_path}')
    HDFWriteModule(output_path).create_and_write_hdf_simple({
        RESPONSE_MATRIX_INV: fits_data,
    })
