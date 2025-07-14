"""
This script will merge together FITS datafiles produced by the piccsim library.
Filenames must use the same indexing scheme because after each glob is sorted,
the files between each one will be matched up with each other; there are no
further checks to ensure that IDs match. This script will use a lot of memory as
all the FITS datafiles are first loaded in before they are written out; using
smaller chunks will reduce memory consumption.
"""

from astropy.io import fits
from glob import glob
import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
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
    ),
    subparser.add_argument(
        '--fits-table-names',
        nargs='+',
        help=('the name of the table each glob of files should receive in the '
              'output HDF file; the entries in this argument should '
              'correspond to the values in the `--fits-file-globs` argument'),
    )
    subparser.add_argument(
        '--first-n-rows',
        type=int,
        metavar='n',
        help='use only the first n files found',
    )
    subparser.add_argument(
        '--rows-per-chunk',
        type=int,
        help='number of rows per chunk when writing out the data',
    )


def convert_piccsim_fits_data(cli_args):
    title('Convert piccsim fits data script')

    step_ri('Creating output directory and writing out CLI args')
    tag = cli_args['tag']
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_piccsim_fits_data')

    step_ri('Verifying file glob and name arrays')
    file_globs = cli_args['fits_file_globs']
    table_names = cli_args['fits_table_names']
    if file_globs is None or table_names is None:
        terminate_with_message('The `--fits-file-globs` and '
                               '`--fits-table-names` args must be passed')
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

    step_ri('Calculating number of chunks')
    first_n_rows = cli_args['first_n_rows']
    rows_per_chunk = cli_args['rows_per_chunk']
    if first_n_rows:
        print(f'Using only the first {first_n_rows} rows')
        # Cap the number of datafiles that are loaded in and written out
        total_file_count = first_n_rows
    if rows_per_chunk is None:
        chunk_count = 1
        rows_per_chunk = total_file_count
    else:
        chunk_count = np.ceil(total_file_count / rows_per_chunk)
    chunk_count = int(chunk_count)
    print(f'A total of {chunk_count} chunk(s) ({rows_per_chunk} row(s) each)')

    step_ri('Loading in glob files and writing out data')
    for chunk_idx in range(chunk_count):
        idx_low = chunk_idx * rows_per_chunk
        idx_high = idx_low + rows_per_chunk
        step(f'On chunk {chunk_idx} [idx {idx_low} - {idx_high}]')
        tables = {}
        for file_glob, table_name in zip(file_globs, table_names):
            step(f'Glob {file_glob}')
            found_datafiles = _make_glob(file_glob)[idx_low:idx_high]
            first_filename = found_datafiles[0].split('/')[-1]
            last_filename = found_datafiles[-1].split('/')[-1]
            print(f'Files: {first_filename} - {last_filename}')
            print('Loading in files...')
            table_data = []
            for datafile_path in found_datafiles:
                with fits.open(datafile_path) as hdul:
                    table_data.append(hdul['PRIMARY'].data)
            print(f'Data stored in the `{table_name}` table')
            tables[table_name] = np.array(table_data)
            dec_print_indent()
        outfile = f'{output_path}/{chunk_idx}_{DATA_F}'
        print()
        print(f'Writing to output HDF datafile: {outfile}')
        HDFWriteModule(outfile).create_and_write_hdf_simple(tables)
        dec_print_indent()
