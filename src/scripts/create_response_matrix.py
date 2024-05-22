"""
Create a response matrix.
"""

from utils.constants import DATA_F, RAW_SIMULATED_DATA_P
from utils.printing_and_logging import step_ri, title


def create_response_matrix_parser(subparsers):
    """
    Example command:
        python3 main.py create_response_matrix --simulated-data-tag ds_fixed_10nm
    """
    subparser = subparsers.add_parser(
        'create_response_matrix',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=create_response_matrix)
    subparser.add_argument(
        '--simulated-data-tag',
        help=('generate the response matrix from simulated data, the data '
              'should be simulated via the `sim_data` script with the '
              '`--fixed-amount-per-zernike` argument passed'),
    )


def create_response_matrix(cli_args):
    title('Create response matrix script')

    simulated_data_tag = cli_args['simulated_data_tag']
    if simulated_data_tag:
        datafile_path = f'{RAW_SIMULATED_DATA_P}/{simulated_data_tag}/{DATA_F}'
        print(datafile_path)
