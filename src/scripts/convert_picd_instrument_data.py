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
                                        step, step_ri, title)
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
        type=int,
        help=('the number of rows to take from the end of the `IMAGE` table '
              'that will be averaged over to form the base field, these rows '
              'should contain no aberrations; if a value of 0 is passed, then '
              'the rows from the PRIMARY table are used'),
    )
    subparser.add_argument(
        '--n-base-field-rows',
        type=int,
        help='only average over the last n rows from the base field data',
    )
    subparser.add_argument(
        '--slice-row-ranges',
        type=int,
        nargs='*',
        help=('slice out rows from the given ranges in the form of '
              '[idx low, idx high]..., this will be done to each datafile'),
    )
    subparser.add_argument(
        '--take-every-n-rows',
        type=int,
        nargs=2,
        help=('take every n rows out of the data starting at row x; '
              'expected parameters: n x'),
    )
    subparser.add_argument(
        '--flip-images-horizontally',
        action='store_true',
        help='flip each image horizontally',
    )
    subparser.add_argument(
        '--use-coeffs-from-csv',
        nargs='+',
        help=('replace the datafile coeffs with coeffs from CSV files; '
              'this is applied after all slicing is done; the following '
              'parameters are expected: base path from `data/raw`, '
              '*csv file names; the `.csv` extension will be added to '
              'every file name by default'),
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

    use_coeffs_from_csv = cli_args.get('use_coeffs_from_csv')
    if use_coeffs_from_csv is not None:
        step_ri('Will use coeffs from CSV files')
        base_path, *filenames = use_coeffs_from_csv
        all_csv_coeffs = []
        for filename in filenames:
            filepath = f'{RAW_DATA_P}/{base_path}/{filename}.csv'
            print(f'Filepath: {filepath}')
            csv_coeffs = np.loadtxt(filepath, delimiter=',')
            print(f'Shape: {csv_coeffs.shape}')
            all_csv_coeffs.extend(csv_coeffs)
        zernike_data_from_csv = np.array(all_csv_coeffs)
        dec_print_indent()

    outfile_idx = 0
    for fits_data_path in sorted(fits_data_paths):
        step_ri(f'Tag path: {fits_data_path}')
        found_datafiles_paths = glob(f'{fits_data_path}/*_data.fits')
        if len(found_datafiles_paths) == 0:
            terminate_with_message('No datafiles found')
        for datafile_path in sorted(found_datafiles_paths):
            print(f'Input FITS datafile: {datafile_path}')
            inc_print_indent()
            with fits.open(datafile_path) as hdul:
                if cli_args.get('base_field_data') is not None:
                    print('Taking the average of all the rows.')
                    row_count = cli_args['base_field_data']
                    if row_count == 0:
                        rows_to_average = hdul['PRIMARY'].data
                    else:
                        # This is the preferred method if there are extra rows
                        rows_to_average = hdul['IMAGE'].data[-row_count:]
                    n_base_field_rows = cli_args.get('n_base_field_rows')
                    if n_base_field_rows:
                        print(f'Only using the last {n_base_field_rows} rows')
                        rows_to_average = rows_to_average[-n_base_field_rows:]
                    image_data = np.average(
                        rows_to_average,
                        axis=0,
                    )[None, :, :]
                    zernike_data = np.array([np.zeros_like(zernike_terms)])
                else:
                    image_data = hdul['IMAGE'].data
                    # The data is in μm of surface error
                    # The data should be in m of wavefront error
                    zernike_data = hdul['ZCMD'].data * 2 * 1e-6
                    first_n_rows = cli_args.get('first_n_rows')
                    # Use only the first n rows
                    if first_n_rows:
                        image_data = image_data[:first_n_rows]
                        zernike_data = zernike_data[:first_n_rows]
                    # Slice out specific rows from each datafile
                    slice_row_ranges = cli_args.get('slice_row_ranges')
                    if slice_row_ranges:
                        step('Slicing out specific rows')
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
                    take_every_n_rows = cli_args.get('take_every_n_rows')
                    if take_every_n_rows:
                        skip_count, starting_row = take_every_n_rows
                        print(f'Image data shape: {image_data.shape}')
                        print(f'Zernike data shape: {zernike_data.shape}')
                        print(f'Will take every {skip_count} rows '
                              f'starting at row {starting_row}')
                        image_data = image_data[starting_row::skip_count]
                        zernike_data = zernike_data[starting_row::skip_count]
            if zernike_data.shape[1] != len(zernike_terms):
                terminate_with_message('Incorrect number of Zernike terms')
            if cli_args['flip_images_horizontally']:
                step_ri('Flipping images horizontally')
                image_data = image_data[:, :, ::-1]
            if use_coeffs_from_csv is not None:
                step_ri('Taking rows from CSV file')
                number_of_rows = zernike_data.shape[0]
                print(f'Number of rows: {number_of_rows}')
                zernike_data = zernike_data_from_csv[:number_of_rows]
                zernike_data_from_csv = zernike_data_from_csv[number_of_rows:]
                dec_print_indent()
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
