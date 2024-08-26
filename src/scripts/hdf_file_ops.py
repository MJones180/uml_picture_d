"""
HDF file operations script.
"""

from utils.hdf_read_and_write import read_hdf
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def hdf_file_ops_parser(subparsers):
    subparser = subparsers.add_parser(
        'hdf_file_ops',
        help='tools to edit HDF files',
    )
    subparser.set_defaults(main=hdf_file_ops)
    subparser.add_argument(
        '--trim-rows-in-datafile-based-on-table',
        nargs='+',
        metavar='hdf_relative_datafile_path source_table *tables_to_trim',
        help=('given the source table, will trim all other listed tables to '
              'the same number of rows (operates on dimension zero); this '
              'option will overwrite the existing HDF datafile'),
    )


def hdf_file_ops(cli_args):
    title('HDF file ops script')

    if cli_args.get('trim_rows_in_datafile_based_on_table'):
        step_ri('Trim Rows in Datafile Based on Table')
        args = cli_args.get('trim_rows_in_datafile_based_on_table')
        if len(args) < 3:
            terminate_with_message('Expected 3 or more arguments.')
        datafile_path, source_table, *trim_tables = args
        with read_hdf(datafile_path, mode='r+') as datafile:
            target_rows = datafile[source_table].shape[0]
            print(f'{source_table} has {target_rows} rows')
            for table in trim_tables:
                # Grab the trimmed rows that will remain in the datafile
                trimmed_table_data = datafile[table][:target_rows]
                orig_shape = datafile[table].shape
                trim_shape = trimmed_table_data.shape
                print(f'{table}: {orig_shape} -> {trim_shape}')
                # The table needs to be deleted and then rewritten, this is
                # because table sizes cannot technically change
                del datafile[table]
                datafile[table] = trimmed_table_data
