"""
This script batch tests multiple models synchronously with one command in the
same process. No parallel or multiprocessed testing is performed.

This is a more opinionated version of the `model_test` script wrapped into one
convenient script for testing lots of models at once. Therefore, some
pre-existing features may not be present in this script.

The testing dataset must be the same for every trained model.

Analyses will be saved like normal.
"""

from scripts.model_test import model_test
from utils.printing_and_logging import dec_print_indent, divider, step_ri, title


def batch_model_test_parser(subparsers):
    """
    Example commands:
    python3 main.py batch_model_test \
        testing_03_05_global 5 5 \
        --epoch-and-tag-range last global_v1_ 1 9
    python3 main.py batch_model_test \
        testing_03_05_global 5 5 \
        --tag-and-epoch-range global_v1_1 3 10
    python3 main.py batch_model_test \
        testing_03_05_global 5 5 \
        --tags-and-epochs global_v1_1.last global_v1_5.last global_v1_8.last
    """
    subparser = subparsers.add_parser(
        'batch_model_test',
        help='batch model train different configurations',
    )
    subparser.set_defaults(main=batch_model_test)
    subparser.add_argument(
        'testing_ds',
        help='name of the testing dataset',
    )
    subparser.add_argument(
        'n_rows',
        type=int,
        help='number of rows in the plot for the output comparison',
    )
    subparser.add_argument(
        'n_cols',
        type=int,
        help='number of cols in the plot for the output comparison',
    )
    selection_group = subparser.add_mutually_exclusive_group()
    selection_group.add_argument(
        '--epoch-and-tag-range',
        help=('one epoch for multiple tags in a row appended with different '
              'index values, the `tag_base` must be the same across them and '
              'contain only the part that is shared'),
        metavar=('[epoch]', '[tag_base]', '[starting_idx]', '[ending_idx]'),
        nargs=4,
    )
    selection_group.add_argument(
        '--tag-and-epoch-range',
        help='one tag for multiple epochs in a row',
        metavar=('[tag]', '[epoch_low]', '[epoch_high]'),
        nargs=3,
    )
    selection_group.add_argument(
        '--tags-and-epochs',
        help=('specific tags and epochs, '
              '`last` can be passed for the epoch\'s value'),
        metavar='[tag].[epoch]',
        nargs='+',
    )


def batch_model_test(cli_args):
    title('Batch model test script')

    step_ri('Gathering all model tests that must be run')
    pairs = []

    def _append_pair(epoch, tag):
        pairs.append({
            'epoch': epoch,
            'tag': tag,
            'testing_ds': cli_args['testing_ds'],
            'n_rows': cli_args['n_rows'],
            'n_cols': cli_args['n_cols'],
        })

    epoch_and_tag_range = cli_args.get('epoch_and_tag_range')
    if epoch_and_tag_range:
        print('A range of tags at a given epoch')
        epoch, tag_base, low_idx, high_idx = epoch_and_tag_range
        for tag_idx in range(int(low_idx), int(high_idx) + 1):
            _append_pair(epoch, f'{tag_base}{tag_idx}')
    tag_and_epoch_range = cli_args.get('tag_and_epoch_range')
    if tag_and_epoch_range:
        print('A range of epochs for a specific tag given')
        tag, epoch_low, epoch_high = tag_and_epoch_range
        for epoch in range(int(epoch_low), int(epoch_high) + 1):
            _append_pair(epoch, tag)
    tags_and_epochs = cli_args.get('tags_and_epochs')
    if tags_and_epochs:
        print('Specific tags and epochs given')
        for tag_and_epoch in tags_and_epochs:
            tag, epoch = tag_and_epoch.split('.')
            _append_pair(epoch, tag)
    total_pairs = len(pairs)
    print(f'There are a total of {total_pairs} pairs')

    print('Will begin iteratively calling `model_test` for each pair')
    for idx, pair in enumerate(pairs):
        step_ri(f'Pair {idx}/{total_pairs}')
        for key, val in pair.items():
            print(f'{key}: {val}')
        dec_print_indent()
        divider()
        model_test(pair)
