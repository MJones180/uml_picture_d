"""
This script will merge together FITS datafiles produced by the piccsim library.
Filenames must use the same indexing scheme because after each glob is sorted,
the files between each one will be matched up with each other; there are no
further checks to ensure that IDs match. This script will use a lot of memory as
all the FITS datafiles are first loaded in before they are written out; using
smaller chunks will reduce memory consumption.
This script can also be used to export the science camera mask to HDF.
"""

from astropy.io import fits
from glob import glob
import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DARK_ZONE_MASK, DATA_F, RAW_DATA_P, ZERNIKE_TERMS
from utils.hdf_read_and_write import HDFWriteModule
from utils.path import make_dir
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.terminate_with_message import terminate_with_message


def convert_piccsim_fits_data_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_piccsim_fits_data',
        help='convert and merge FITS datafiles from piccsim to HDF',
    )
    subparser.set_defaults(main=convert_piccsim_fits_data)
    subparser.add_argument(
        'tag',
        help='tag of the converted raw dataset',
    )
    subparser.add_argument(
        'dir_path',
        help='path to the directory containing the fits datafiles',
    )
    subparser.add_argument(
        '--fits-file-globs',
        nargs='+',
        help=('glob (omitting the `.fits` extension) for each of the files '
              'that should be grabbed; each file glob will get its own table '
              'in the output HDF file; each glob should be wrapped in single '
              'quotes to avoid evaluation before being passed; each set of '
              'filenames must use the same indexing scheme because they will '
              'all be sorted and aligned'),
    )
    subparser.add_argument(
        '--fits-table-names',
        nargs='+',
        help=('the name of the table each glob of files should receive in the '
              'output HDF file; the entries in this argument should '
              'correspond to the values in the `--fits-file-globs` argument'),
    )
    subparser.add_argument(
        '--slice-row-ranges',
        type=int,
        nargs='*',
        help=('slice out rows from the given ranges in the form of '
              '[idx low, idx high]..., this will be done to each table; '
              'cannot be used with the `--first-n-rows` arg'),
    )
    subparser.add_argument(
        '--first-n-rows',
        type=int,
        metavar='n',
        help=('use only the first n files found; cannot be used with the '
              '`--slice-row-ranges` arg'),
    )
    subparser.add_argument(
        '--rows-per-chunk',
        type=int,
        help='number of rows per chunk when writing out the data',
    )
    subparser.add_argument(
        '--add-dummy-tables',
        nargs='+',
        help='tables to add with a dummy value of 0 to the output datafiles')
    subparser.add_argument(
        '--add-zernikes',
        type=int,
        nargs=2,
        help='add Zernike terms to the output datafiles',
    )
    subparser.add_argument(
        '--sci-cam-mask-file',
        help=('name of the file containing the science camera mask; this '
              'script will terminate after the mask is converted, so there '
              'is no point in using any of the other args'),
    )
    subparser.add_argument(
        '--save-difference-only',
        nargs='+',
        help=('save only the difference of a row with respect to another row '
              'in groups, this is handy for saving the delta DM commands; the '
              'arguments should be: <number of rows per group> <reference row '
              'of group> <first n rows of group to keep> *<table names>; the '
              'tables in this argument should also be in the '
              '`--fits-table-names` argument; any arguments that augment the '
              'number of rows should be careful to not split up any groups; '
              'the reference row in each group will not be saved'),
    )


def convert_piccsim_fits_data(cli_args):
    title('Convert piccsim fits data script')

    step_ri('Creating output directory and writing out CLI args')
    tag = cli_args['tag']
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_piccsim_fits_data')

    dir_path = cli_args['dir_path']
    sci_cam_mask_file = cli_args.get('sci_cam_mask_file')
    if sci_cam_mask_file:
        step_ri('Saving the science camera mask')
        mask_file_path = f'{dir_path}/{sci_cam_mask_file}'
        print(f'Mask file path: {mask_file_path}')
        mask = fits.getdata(mask_file_path, memmap=False).astype(np.bool)
        out_data = {DARK_ZONE_MASK: mask}
        outfile = f'{output_path}/0_{DATA_F}'
        print(f'Writing to output HDF datafile: {outfile}')
        HDFWriteModule(outfile).create_and_write_hdf_simple(out_data)
        quit()

    step_ri('Verifying file glob and name arrays')
    file_globs = cli_args['fits_file_globs']
    table_names = cli_args['fits_table_names']
    if file_globs is None or table_names is None:
        terminate_with_message('The `--fits-file-globs` and '
                               '`--fits-table-names` args must be passed, '
                               'or `--sci-cam-mask-file` must be passed')
    if len(file_globs) != len(table_names):
        terminate_with_message('The `--fits-file-globs` and '
                               '`--fits-table-names` must be the same length')

    step_ri('Verifying each glob produces the same number of files')
    dir_path = cli_args['dir_path']

    def _make_glob(glob_str):
        return sorted(glob(f'{dir_path}/{glob_str}.fits'))

    total_file_count = None
    for file_glob in file_globs:
        step(f'Finding files with the glob {file_glob}')
        found_files = _make_glob(file_glob)
        datafile_count = len(found_files)
        print(f'A total of {datafile_count} found')
        print(f'First file path: {found_files[0]}')
        print(f'Last file path: {found_files[-1]}')
        if total_file_count is None:
            total_file_count = datafile_count
        # The glob is producing a different number of found datafiles
        elif total_file_count != datafile_count:
            terminate_with_message(f'{file_glob} produces a different number '
                                   'of found datafiles')
        dec_print_indent()

    row_slice_mask = None
    slice_row_ranges = cli_args.get('slice_row_ranges')
    first_n_rows = cli_args.get('first_n_rows')
    if slice_row_ranges is not None:
        step_ri('Slicing out specific simulations')
        if len(slice_row_ranges) % 2 == 1:
            terminate_with_message('Invalid row slice params')
        # A mask of the rows to keep
        row_slice_mask = np.full(total_file_count, False)
        for range_idx in range(len(slice_row_ranges) // 2):
            idx_low = slice_row_ranges[range_idx * 2]
            idx_high = slice_row_ranges[range_idx * 2 + 1]
            print(f'Index low, high: {idx_low}, {idx_high}')
            row_slice_mask[idx_low:idx_high] = True
        total_file_count = row_slice_mask.sum()
    elif first_n_rows:
        print(f'Using only the first {first_n_rows} rows')
        # Cap the number of datafiles that are loaded in and written out
        total_file_count = min(total_file_count, first_n_rows)

    step_ri('Calculating number of chunks')
    rows_per_chunk = cli_args.get('rows_per_chunk')
    if rows_per_chunk is None:
        chunk_count = 1
        rows_per_chunk = total_file_count
    else:
        chunk_count = np.ceil(total_file_count / rows_per_chunk)
        rows_per_chunk = min(rows_per_chunk, total_file_count)
    chunk_count = int(chunk_count)
    print(f'A total of {chunk_count} chunk(s) ({rows_per_chunk} row(s) each)')

    step_ri('Setting up the base tables that will be written out')
    base_tables = {}
    add_dummy_tables = cli_args.get('add_dummy_tables')
    add_zernikes = cli_args.get('add_zernikes')
    if add_dummy_tables is not None:
        print(f'Adding dummy tables: {add_dummy_tables}')
        for table in add_dummy_tables:
            base_tables[table] = 0
    if add_zernikes is not None:
        zernike_range = np.arange(add_zernikes[0], add_zernikes[1] + 1)
        print(f'Adding Zernikes: {zernike_range}')
        base_tables[ZERNIKE_TERMS] = zernike_range

    save_differences = False
    save_difference_only = cli_args.get('save_difference_only')
    if save_difference_only:
        step_ri('Saving the difference of some tables')
        rows_per_group = int(save_difference_only[0])
        print(f'The number of rows per group: {rows_per_group}')
        reference_row = int(save_difference_only[1])
        print(f'The reference row in each group (0 indexed): {reference_row}')
        first_n_rows = int(save_difference_only[2])
        print(f'Only the first {first_n_rows} rows will be kept in each group')
        save_difference_tables = save_difference_only[3:]
        print(f'Will work on the tables: {save_difference_tables}')
        save_differences = True

    step_ri('Loading in glob files and writing out data')
    for chunk_idx in range(chunk_count):
        idx_low = chunk_idx * rows_per_chunk
        idx_high = np.min((idx_low + rows_per_chunk, total_file_count))
        step(f'On chunk {chunk_idx} [idx {idx_low} - {idx_high}]')
        tables = base_tables.copy()
        for file_glob, table_name in zip(file_globs, table_names):
            step(f'Glob {file_glob}')
            found_datafiles = _make_glob(file_glob)
            # print(found_datafiles)
            if row_slice_mask is not None:
                # Take out the sliced rows if specified
                found_datafiles = np.array(found_datafiles)[row_slice_mask]
            # Slice out the simulations for this chunk
            found_datafiles = found_datafiles[idx_low:idx_high]
            first_filename = found_datafiles[0].split('/')[-1]
            last_filename = found_datafiles[-1].split('/')[-1]
            print(f'Files: {first_filename} - {last_filename}')
            print('Loading in files...')
            tables[table_name] = np.array([
                fits.getdata(datafile_path, memmap=False)
                for datafile_path in found_datafiles
            ])
            print(f'Data stored in the `{table_name}` table')
            dec_print_indent()
        if save_differences:
            step('Taking table differences')
            rows = idx_high - idx_low
            # Index of each reference row that will be used
            ref_row_idxs = np.arange(reference_row, rows, rows_per_group)
            # Index of each row that the difference will be taken of
            all_other_idxs = np.setdiff1d(np.arange(rows), ref_row_idxs, True)
            # Create a copy of this index for each row that will be a difference
            ref_row_idxs_rep = np.repeat(ref_row_idxs, rows_per_group - 1)
            # Update each table
            for table in table_names:
                # Take the difference for this table
                if table in save_difference_tables:
                    tables[table] = (tables[table][ref_row_idxs_rep] -
                                     tables[table][all_other_idxs])
                # Just delete the reference rows from this table
                else:
                    tables[table] = tables[table][all_other_idxs]
            # At this point, each group is now one row smaller due to the
            # reference row being removed. Therefore, decrease the number of
            # rows by one for each group
            orig_rows = rows
            rows -= rows // rows_per_group
            rows_per_group_m1 = rows_per_group - 1
            # Now, figure out how many rows at the start of each group to keep
            idxs_to_keep = []
            # Build up the indexes to keep one row per group at a time
            for start in range(first_n_rows):
                idxs_to_keep.extend(np.arange(start, rows, rows_per_group_m1))
            idxs_to_keep = [int(idx) for idx in idxs_to_keep]
            # Keep the rows grouped together
            idxs_to_keep.sort()
            # Remove any rows that are not wanted
            for table in table_names:
                tables[table] = tables[table][idxs_to_keep]
                print(table, np.min(tables[table]), np.max(tables[table]))
                print(f'Table {table}: {orig_rows} -> '
                      f'{len(tables[table])} rows')
            dec_print_indent()
        outfile = f'{output_path}/{chunk_idx}_{DATA_F}'
        print()
        print(f'Writing to output HDF datafile: {outfile}')
        HDFWriteModule(outfile).create_and_write_hdf_simple(tables)
        dec_print_indent()
