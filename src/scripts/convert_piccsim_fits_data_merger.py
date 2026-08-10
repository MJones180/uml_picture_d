"""
This script will merge together piccsim simulations across many folders into a
single HDF file. After this, the `convert_piccsim_fits_data` script can be
called to process all the data as needed. The reason this is a separate script
is because the FITS files are spread across many directories (jobs on Unity)
and the FITS datafiles store the data differently.
"""

from astropy.io import fits
import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
from utils.path import make_dir, path_exists
from utils.printing_and_logging import dec_print_indent, step, step_ri, title


def convert_piccsim_fits_data_merger_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_piccsim_fits_data_merger',
        help='merge FITS datafiles from piccsim to HDF',
    )
    subparser.set_defaults(main=convert_piccsim_fits_data_merger)
    subparser.add_argument(
        'tag',
        help='tag of the converted raw dataset',
    )
    subparser.add_argument(
        'base_path',
        help='path to the root directory containing the simulation directories',
    )
    subparser.add_argument(
        'sim_dir_shared',
        help='shared part of the name for each simulation directory',
    )
    subparser.add_argument(
        'sim_dir_idx_lower',
        type=int,
        help='lower idx of the simulation dirs to iterate through (inclusive)',
    )
    subparser.add_argument(
        'sim_dir_idx_upper',
        type=int,
        help='upper idx of the simulation dirs to iterate through (inclusive)',
    )
    subparser.add_argument(
        'rows_per_sim',
        type=int,
        help='the number of rows for each simulation',
    )
    subparser.add_argument(
        '--file-names',
        nargs='+',
        help=('name of each FITS datafile (omitting the extension); the same '
              'name will be used for each table in the HDF file'),
    )
    subparser.add_argument(
        '--allow-missing-dirs',
        action='store_true',
        help='allow sim directories to be missing',
    )
    subparser.add_argument(
        '--save-as-float32',
        action='store_true',
        help='save the data as float32',
    )


def convert_piccsim_fits_data_merger(cli_args):
    title('Convert piccsim fits data merger script')

    step_ri('Creating output directory and writing out CLI args')
    tag = cli_args['tag']
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_piccsim_fits_data_merger')

    step_ri('Simulation directory information')
    base_path = cli_args['base_path']
    sim_dir_shared = cli_args['sim_dir_shared']
    sim_dir_idx_lower = cli_args['sim_dir_idx_lower']
    sim_dir_idx_upper = cli_args['sim_dir_idx_upper']
    rows_per_sim = cli_args['rows_per_sim']
    print(f'Base path: {base_path}')
    print(f'Sim dir shared: {sim_dir_shared}')
    print(f'Sim dir idx lower: {sim_dir_idx_lower}')
    print(f'Sim dir idx upper: {sim_dir_idx_upper}')
    print(f'Rows per sim: {rows_per_sim}')

    allow_missing_dirs = cli_args['allow_missing_dirs']
    if allow_missing_dirs:
        step_ri('Missing sim directories allowed')

    step_ri('Setting up data that will be read')
    file_names = cli_args['file_names']
    print(f'File names: {file_names}')
    merged_data = {key: [] for key in file_names}

    save_as_float32 = cli_args['save_as_float32']
    if save_as_float32:
        step_ri('Will save the data as float32')

    step_ri('Iterating through each simulation directory')
    total_rows = 0
    total_sim_dirs = 0
    for dir_idx in range(sim_dir_idx_lower, sim_dir_idx_upper + 1):
        full_sim_dir_name = f'{sim_dir_shared}{dir_idx}'
        base_file_path = f'{base_path}/{full_sim_dir_name}'
        if allow_missing_dirs and not path_exists(base_file_path):
            continue
        # Do an initial pass to count the number of rows in each datafile
        rows_per_file = []
        for file_name in file_names:
            with fits.open(f'{base_file_path}/{file_name}.fits') as hdul:
                rows_per_file.append(len(hdul) - 1)
        step(full_sim_dir_name)
        print(f'Starting rows/file: {rows_per_file}')
        # If a job gets cancelled while data is being written out, then the
        # datafiles may not have the same number of rows; keep iterating until
        # - each datafile produces the same number of rows
        # - the rows encompass a complete set of simulations
        while len(set(rows_per_file)) > 1 or rows_per_file[0] % rows_per_sim:
            # Grab the file with the most rows
            max_idx = np.argmax(rows_per_file)
            max_val = rows_per_file[max_idx]
            # The number of rows over a complete set of simulations;
            # example: 16 rows when 5 rows/simulation -> leftover = 1
            leftover = max_val % rows_per_sim
            # This can occur if one datafile was completely written out, while
            # another was not; example: 20 rows in A, 25 rows in B, 5 rows/sim
            if leftover == 0:
                leftover = rows_per_sim
            rows_per_file[max_idx] = max_val - leftover
        # Now, the same number of rows can be grabbed from each file
        rows_per_file = rows_per_file[0]
        total_rows += rows_per_file
        total_sim_dirs += 1
        for file_name in file_names:
            fits_path = f'{base_file_path}/{file_name}.fits'
            with fits.open(fits_path) as hdul:
                for row_idx in range(rows_per_file):
                    # Need to offset by 1 to ignore the empty primary
                    row_data = hdul[row_idx + 1].data
                    if save_as_float32:
                        row_data.astype(np.float32)
                    merged_data[file_name].append(row_data)
        sims_per_file = rows_per_file // rows_per_sim
        print(f'{sims_per_file} simulations ({rows_per_file} rows)')
        dec_print_indent()

    step_ri('Overall statistics')
    print(f'Total rows: {total_rows}')
    total_sims = total_rows // rows_per_sim
    print(f'Total simulations: {total_sims}')
    print(f'Total simulation dirs: {total_sim_dirs}')
    print(f'Avg simulations per directory: {total_sims / total_sim_dirs}')

    step_ri('Writing out merged HDF data')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Path: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple(merged_data)
