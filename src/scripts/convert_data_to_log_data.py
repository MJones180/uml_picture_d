import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def convert_data_to_log_data_parser(subparsers):
    subparser = subparsers.add_parser(
        'convert_data_to_log_data',
        help='take the log10 of data',
    )
    subparser.set_defaults(main=convert_data_to_log_data)
    subparser.add_argument(
        'current_tag',
        help='tag of the raw dataset containing the data',
    )
    subparser.add_argument(
        'new_tag',
        help='tag of the new raw dataset which will contain the log data',
    )
    subparser.add_argument(
        '--tables-to-log',
        nargs='*',
        help='names of tables to take the log of',
    )
    subparser.add_argument(
        '--tables-to-copy',
        nargs='*',
        help='names of tables to copy over to the new datafile',
    )
    subparser.add_argument(
        '--epsilon',
        type=float,
        help='epsilon to add to the log10',
    )


def convert_data_to_log_data(cli_args):
    title('Convert data to log data')

    step_ri('Loading existing data')
    current_tag = cli_args['current_tag']
    print(f'Tag: {current_tag}')
    tables_to_log = cli_args.get('tables_to_log') or []
    tables_to_copy = cli_args.get('tables_to_copy') or []
    print(f'Log tables: {tables_to_log}')
    print(f'Copying tables: {tables_to_copy}')
    log_table_data = {table: [] for table in tables_to_log}
    copy_table_data = {table: [] for table in tables_to_copy}
    for data_path in raw_sim_data_chunk_paths(current_tag):
        print(f'Loading in data from {data_path}')
        data = read_hdf(data_path)
        for table_name in tables_to_log:
            log_table_data[table_name].extend(data[table_name][:])
        for table_name in tables_to_copy:
            copy_table_data[table_name].extend(data[table_name][:])
    for table_name, table_data in log_table_data.items():
        log_table_data[table_name] = np.array(table_data)
        print(f'{table_name} Shape: {log_table_data[table_name].shape}')
    for table_name, table_data in copy_table_data.items():
        copy_table_data[table_name] = np.array(table_data)
        print(f'{table_name} Shape: {copy_table_data[table_name].shape}')

    step_ri('Taking log of data')
    epsilon = cli_args.get('epsilon') or 1e-10
    for table_name, table_data in log_table_data.items():
        np.log10(table_data + epsilon, out=log_table_data[table_name])

    step_ri('Creating output directory and writing out CLI args')
    new_tag = cli_args['new_tag']
    output_path = f'{RAW_DATA_P}/{new_tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'convert_data_to_log_data')

    step_ri('Writing the data to HDF')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Writing to output HDF datafile: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple({
        **log_table_data,
        **copy_table_data,
    })
