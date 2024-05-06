"""
This script finds the best hyperparameters for a given dataset by training all
the different model combinations.
"""

from utils.printing_and_logging import title


def grid_search_parser(subparsers):
    """
    Example commands:
    ...
    """
    subparser = subparsers.add_parser(
        'grid_search_train_and_test',
        help='grid search train models ',
    )
    subparser.set_defaults(main=grid_search_train_and_test)
    subparser.add_argument(
        'argname',
        help='help',
    )
    subparser.add_argument(
        '--spawn-new-jobs',
        help='spawn new jobs for each training and testing call',
    )


def grid_search_train_and_test(cli_args):
    title('Grid search train and test script')
