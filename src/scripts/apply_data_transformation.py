import numpy as np
from utils.cli_args import save_cli_args
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def apply_data_transformation_parser(subparsers):
    subparser = subparsers.add_parser(
        'apply_data_transformation',
        help='apply a transformation to the data',
    )
    subparser.set_defaults(main=apply_data_transformation)
    subparser.add_argument(
        'current_tag',
        help='tag of the raw dataset',
    )
    subparser.add_argument(
        'new_tag',
        help=('tag of the new raw dataset which will contain '
              'the transformed data'),
    )
    subparser.add_argument(
        '--tables-to-transform',
        nargs='*',
        help='names of tables to transform',
    )
    subparser.add_argument(
        '--tables-to-copy',
        nargs='*',
        help='names of tables to copy over to the new datafile',
    )
    subparser.add_argument(
        '--log10-data',
        type=float,
        help='apply a log10 transformation; passed arg should be the epsilon',
    )
    subparser.add_argument(
        '--sqrt-data',
        action='store_true',
        help='apply a sqrt transformation',
    )
    subparser.add_argument(
        '--table-difference',
        nargs=3,
        help=('take the difference between two tables; three args expected: '
              'table 1, table 2, new table; new table = table 1 - table 2'),
    )


def apply_data_transformation(cli_args):
    title('Apply data transformation')

    step_ri('Loading existing data')
    current_tag = cli_args['current_tag']
    print(f'Tag: {current_tag}')
    tables_to_transform = cli_args.get('tables_to_transform') or []
    tables_to_copy = cli_args.get('tables_to_copy') or []
    print(f'Transform tables: {tables_to_transform}')
    print(f'Copying tables: {tables_to_copy}')
    table_data = {
        table: []
        for table in [*tables_to_transform, *tables_to_copy]
    }
    for data_path in raw_sim_data_chunk_paths(current_tag):
        print(f'Loading in data from {data_path}')
        data = read_hdf(data_path)
        for table_name in table_data.keys():
            table_data[table_name].extend(data[table_name][:])
    for table_name in table_data.keys():
        table_data[table_name] = np.array(table_data[table_name])
        print(f'{table_name} Shape: {table_data[table_name].shape}')

    step_ri('Transforming the data')
    log10_data = cli_args.get('log10_data')
    sqrt_data = cli_args.get('sqrt_data')
    if log10_data is not None:
        print(f'Will apply a log10 transform, epsilon = {log10_data}')
    elif sqrt_data is not None:
        print('Will apply a sqrt transform')
    for table_name in tables_to_transform:
        values = table_data[table_name]
        if log10_data is not None:
            np.log10(values + log10_data, out=table_data[table_name])
        elif sqrt_data is not None:
            np.sqrt(values, out=table_data[table_name])

    table_difference = cli_args.get('table_difference')
    if table_difference is not None:
        step_ri('Taking difference of two tables')
        table1, table2, new_table = table_difference
        print(f'{new_table} = {table1} - {table2}')
        table_data[new_table] = table_data.pop(table1) - table_data.pop(table2)

    step_ri('Creating output directory and writing out CLI args')
    new_tag = cli_args['new_tag']
    output_path = f'{RAW_DATA_P}/{new_tag}'
    make_dir(output_path)
    save_cli_args(output_path, cli_args, 'apply_data_transformation')

    step_ri('Writing the data to HDF')
    outfile = f'{output_path}/0_{DATA_F}'
    print(f'Writing to output HDF datafile: {outfile}')
    HDFWriteModule(outfile).create_and_write_hdf_simple(table_data)
