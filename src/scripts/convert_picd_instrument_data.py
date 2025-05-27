"""
Data obtained on the PICTURE-D instrument is in the FITS file format. To be able
to use the normal preprocessing scripts, the data should be converted to HDF and
have the same format as raw simulated data.
"""

from astropy.io import fits
from glob import glob
import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import (CAMERA_INTENSITY, CAMERA_SAMPLING, DATA_F,
                             RAW_DATA_P, ZERNIKE_COEFFS, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.path import make_dir, path_exists
from utils.printing_and_logging import (dec_print_indent, inc_print_indent,
                                        step_ri, title)
from utils.terminate_with_message import terminate_with_message


def convert_picd_instrument_data_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_picd_instrument_data',
        help='convert FITS data from the PICTURE-D instrument',
    )
    subparser.set_defaults(main=convert_picd_instrument_data)
    subparser.add_argument(
        'tag',
        help='tag of the converted raw dataset',
    )
    subparser.add_argument(
        'zernike_low',
        type=int,
        help='lower bound on the Zernike terms',
    )
    subparser.add_argument(
        'zernike_high',
        type=int,
        help='lower bound on the Zernike terms',
    )
    subparser.add_argument(
        '--fits-data-tags',
        nargs='+',
        help=('tags of the directories containing the FITS datafiles; '
              'multiple datafiles can be in the same directory with names of '
              '`*_data.fits`.'),
    )
    subparser.add_argument(
        '--first-n-rows',
        type=int,
        metavar='n',
        help='use only the first n rows',
    )
    subparser.add_argument(
        '--base-field-data',
        action='store_true',
        help='average together all the rows to form the base field',
    )
    subparser.add_argument(
        '--slice-row-ranges',
        type=int,
        nargs='*',
        help=('slice out rows from the given ranges in the form of '
              '[idx low, idx high]..., this will be done to each datafile'),
    )


def convert_picd_instrument_data(cli_args):
    title('Convert picd instrument data script')

    step_ri('Determining Zernike terms')
    zernike_terms = np.arange(cli_args['zernike_low'],
                              cli_args['zernike_high'] + 1)
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Creating output directory and writing out CLI args')
    tag = cli_args['tag']
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_picd_instrument_data')

    step_ri('Verifying tags exist')
    fits_data_tags = cli_args['fits_data_tags']
    print(f'FITS data tags: {fits_data_tags}')
    fits_data_paths = [f'{RAW_DATA_P}/{tag}' for tag in fits_data_tags]
    not_found_tags = [p for p in fits_data_paths if not path_exists(p)]
    if len(not_found_tags) != 0:
        terminate_with_message(f'Not all tags found: {not_found_tags}')

    step_ri('Will begin looping through all tags')

    outfile_idx = 0
    for fits_data_path in fits_data_paths:
        step_ri(f'Tag path: {fits_data_path}')
        found_datafiles_paths = glob(f'{fits_data_path}/*_data.fits')
        if len(found_datafiles_paths) == 0:
            terminate_with_message('No datafiles found')
        for datafile_path in found_datafiles_paths:
            print(f'Input FITS datafile: {datafile_path}')
            inc_print_indent()
            with fits.open(datafile_path) as hdul:
                if cli_args.get('base_field_data'):
                    print('Taking the average of all the rows.')
                    # This table name is different than the other datafiles
                    image_data = hdul['PRIMARY'].data
                    # The data from the PRIMARY table may not be good, so may
                    # need to use the aberration free rows from the IMAGE table
                    # image_data = hdul['IMAGE'].data[30000:]
                    image_data = np.average(image_data, axis=0)[None, :, :]
                    zernike_data = np.array([np.zeros_like(zernike_terms)])
                else:
                    image_data = hdul['IMAGE'].data
                    # The data is in μm of surface error
                    # The data should be in nm of wavefront error
                    zernike_data = hdul['ZCMD'].data * 2 * 1e-6
                    first_n_rows = cli_args.get('first_n_rows')
                    # Use only the first n rows
                    if first_n_rows:
                        image_data = image_data[:first_n_rows]
                        zernike_data = zernike_data[:first_n_rows]
                    # Slice out specific rows from each datafile
                    slice_row_ranges = cli_args.get('slice_row_ranges')
                    if slice_row_ranges:
                        print('Slicing out specific rows')
                        inc_print_indent()
                        if len(slice_row_ranges) % 2 == 1:
                            terminate_with_message('Invalid row slice params')
                        # A mask of the rows to keep
                        row_mask = np.full(image_data.shape[0], False)
                        for range_idx in range(len(slice_row_ranges) // 2):
                            idx_low = slice_row_ranges[range_idx * 2]
                            idx_high = slice_row_ranges[range_idx * 2 + 1]
                            print(f'Index low, high: {idx_low}, {idx_high}')
                            row_mask[idx_low:idx_high] = True
                        dec_print_indent()
                        image_data = image_data[row_mask]
                        zernike_data = zernike_data[row_mask]
            if zernike_data.shape[1] != len(zernike_terms):
                terminate_with_message('Incorrect number of Zernike terms')
            print(f'Image data shape: {image_data.shape}')
            print(f'Zernike data shape: {zernike_data.shape}')
            outfile = f'{output_path}/{outfile_idx}_{DATA_F}'
            print(f'Output HDF datafile: {outfile}')
            HDFWriteModule(outfile).create_and_write_hdf_simple({
                ZERNIKE_TERMS: zernike_terms,
                ZERNIKE_COEFFS: zernike_data.astype('float32'),
                CAMERA_INTENSITY: image_data.astype('float32'),
                CAMERA_SAMPLING: 0,
            })
            outfile_idx += 1
            dec_print_indent()
