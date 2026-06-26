"""
This script will merge together piccsim simulations across many folders into a
single HDF file. After this, the `convert_piccsim_fits_data` script can be
called to process all the data as needed. The reason this is a separate script
is because the FITS files are spread across many directories (jobs on Unity)
and the FITS datafiles store the data differently.
"""

from astropy.io import fits
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


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

    step_ri('Setting up data that will be read')
    file_names = cli_args['file_names']
    print(f'File names: {file_names}')
    merged_data = {key: [] for key in file_names}

    step_ri('Iterating through each simulation directory')
    total_rows = 0
    for dir_idx in range(sim_dir_idx_lower, sim_dir_idx_upper + 1):
        full_sim_dir_name = f'{sim_dir_shared}{dir_idx}'
        rows_per_file = None
        for file_name in file_names:
            fits_path = f'{base_path}/{full_sim_dir_name}/{file_name}.fits'
            with fits.open(fits_path) as hdul:
                file_rows = len(hdul) - 1
                # Verify that each file in this simulation directory has the
                # same number of rows
                if rows_per_file is None:
                    rows_per_file = file_rows
                    total_rows += rows_per_file
                elif rows_per_file != file_rows:
                    terminate_with_message(f'{fits_path} has a different '
                                           f'number of rows ({file_rows})')
                for row_idx in range(file_rows):
                    # Need to offset by 1 to ignore the empty primary
                    merged_data[file_name].append(hdul[row_idx + 1].data)
        # Ensure all the data is present
        if rows_per_file % rows_per_sim != 0:
            terminate_with_message('Incomplete data (rows are missing '
                                   'for one of the simulations)')
        sims_per_file = rows_per_file // rows_per_sim
        print(f'{full_sim_dir_name}: {sims_per_file} simulations '
              f'({rows_per_file} rows)')

    step_ri('Overall statistics')
    print(f'Total rows: {total_rows}')
    total_sims = total_rows // rows_per_sim
    print(f'Total simulations: {total_sims}')
    total_sim_dirs = sim_dir_idx_upper - sim_dir_idx_lower + 1
    print(f'Total simulation dirs: {total_sim_dirs}')
    print(f'Avg simulations per directory: {total_sims / total_sim_dirs}')

    step_ri('Writing out merged HDF data')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Path: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple(merged_data)
