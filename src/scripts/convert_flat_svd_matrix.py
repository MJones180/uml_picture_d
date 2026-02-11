"""
This script will convert a flat SVD matrix in FITS format to HDF.
Previously, the `convert_piccsim_fits_data` script was used to export the
DM modes which were 3D in shape (mode, pixels, pixels).
However, an important note is that `convert_piccsim_fits_data` can do everything
that this script can do, but this script is just more simplified and easy.
"""

from astropy.io import fits
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def convert_flat_svd_matrix_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_flat_svd_matrix',
        help='convert a flat SVD matrix FITS datafile to HDF',
    )
    subparser.set_defaults(main=convert_flat_svd_matrix)
    subparser.add_argument(
        'tag',
        help='tag of the converted raw dataset',
    )
    subparser.add_argument(
        'file_path',
        help='path to the fits datafile',
    )
    subparser.add_argument(
        'table_name',
        help='name of the table in the HDF file',
    )


def convert_flat_svd_matrix(cli_args):
    title('Convert flat svd matrix')

    step_ri('Creating output directory and writing out CLI args')
    tag = cli_args['tag']
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_flat_svd_matrix')

    step_ri('Loading in the matrix')
    file_path = cli_args['file_path']
    print(f'File path: {file_path}')
    matrix_data = fits.getdata(file_path)

    step_ri('Writing the matrix to HDF')
    table_name = cli_args['table_name']
    print(f'Table name: {table_name}')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Writing to output HDF datafile: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple({
        table_name: matrix_data,
    })
