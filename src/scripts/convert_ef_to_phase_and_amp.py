import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def convert_ef_to_phase_and_amp_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_ef_to_phase_and_amp',
        help='convert EFs to phase and amplitude',
    )
    subparser.set_defaults(main=convert_ef_to_phase_and_amp)
    subparser.add_argument(
        'current_tag',
        help='tag of the raw dataset containing the EF',
    )
    subparser.add_argument(
        'table_name_ef_r',
        help='name of the table in the HDF file for the real EF',
    )
    subparser.add_argument(
        'table_name_ef_i',
        help='name of the table in the HDF file for the imag EF',
    )
    subparser.add_argument(
        'new_tag',
        help='tag of the new raw dataset which will contain the phase and amp',
    )
    subparser.add_argument(
        'table_name_phase',
        help='name of the table in the HDF file for the phase',
    )
    subparser.add_argument(
        'table_name_amp',
        help='name of the table in the HDF file for the amp',
    )
    subparser.add_argument(
        '--tables-to-copy',
        nargs='*',
        help='names of tables to copy over to the new datafile',
    )


def convert_ef_to_phase_and_amp(cli_args):
    title('Convert ef to phase and amp')

    step_ri('Loading existing data')
    current_tag = cli_args['current_tag']
    table_name_ef_r = cli_args['table_name_ef_r']
    table_name_ef_i = cli_args['table_name_ef_i']
    print(f'Tag: {current_tag}')
    print(f'Real EF Table: {table_name_ef_r}')
    print(f'Imag EF Table: {table_name_ef_i}')
    tables_to_copy = cli_args.get('tables_to_copy') or []
    print(f'Copying tables: {tables_to_copy}')
    copy_table_data = {table: [] for table in tables_to_copy}
    ef_r_data = []
    ef_i_data = []
    for data_path in raw_sim_data_chunk_paths(current_tag):
        print(f'Loading in data from {data_path}')
        data = read_hdf(data_path)
        ef_r_data.extend(data[table_name_ef_r][:])
        ef_i_data.extend(data[table_name_ef_i][:])
        for table_name in tables_to_copy:
            copy_table_data[table_name].extend(data[table_name][:])
    ef_r_data = np.array(ef_r_data)
    ef_i_data = np.array(ef_i_data)
    print(f'Real EF Shape: {ef_r_data.shape}')
    print(f'Imag EF Shape: {ef_i_data.shape}')
    for table_name, table_data in copy_table_data.items():
        copy_table_data[table_name] = np.array(table_data)
        print(f'{table_name} Shape: {copy_table_data[table_name].shape}')

    step_ri('Creating mask of active pixels')
    active_pixels_mask = ef_r_data[0] != 0

    step_ri('Converting to phase and amp')
    phase = np.arctan2(ef_i_data, ef_r_data)
    amp = (ef_r_data**2 + ef_i_data**2)**0.5
    # Set all inactive pixels back to zero
    phase[:, ~active_pixels_mask] = 0
    amp[:, ~active_pixels_mask] = 0

    step_ri('Creating output directory and writing out CLI args')
    new_tag = cli_args['new_tag']
    output_path = f'{RAW_DATA_P}/{new_tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_ef_to_phase_and_amp')

    step_ri('Writing the phase and amp to HDF')
    table_name_phase = cli_args['table_name_phase']
    table_name_amp = cli_args['table_name_amp']
    print(f'Phase table: {table_name_phase}')
    print(f'Amp table: {table_name_amp}')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Writing to output HDF datafile: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple({
        table_name_phase: phase,
        table_name_amp: amp,
        **copy_table_data,
    })
