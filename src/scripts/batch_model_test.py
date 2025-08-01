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
from utils.shared_argparser_args import shared_argparser_args


def batch_model_test_parser(subparsers):
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
        '--inputs-need-diff',
        action='store_true',
        help='the inputs need to subtract the base field to get the diff',
    )
    subparser.add_argument(
        '--inputs-need-norm',
        action='store_true',
        help='the inputs need to be normalized',
    )
    subparser.add_argument(
        '--outputs-no-denorm',
        action='store_true',
        help='the outputs do not need to be denormalized',
    )
    subparser.add_argument(
        '--scatter-plot',
        nargs=5,
        metavar=('[n_rows]', '[n_cols]', '[starting_zernike]',
                 '[filter_value]', '[plot_density]'),
        help=('generate a scatter plot; takes the args: number of rows, '
              'number of cols, first Zernike the model outputs, filter value '
              'range, points per pixel to use for the density plot'),
    )
    subparser.add_argument(
        '--zernike-plots',
        action='store_true',
        help='generate the Zernike plots',
    )
    subparser.add_argument(
        '--print-outputs',
        action='store_true',
        help='print out the truth and model outputs',
    )
    subparser.add_argument(
        '--take-rss-model-outputs',
        action='store_true',
        help='print out the RSS of the model outputs',
    )
    subparser.add_argument(
        '--max-rows-per-model-call',
        type=int,
        help='limit the number of rows per model call',
    )
    shared_argparser_args(subparser, ['force_cpu'])

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

    # CLI args to copy over
    base_dict = {
        arg: cli_args[arg]
        for arg in (
            'testing_ds',
            'inputs_need_diff',
            'inputs_need_norm',
            'outputs_no_denorm',
            'scatter_plot',
            'zernike_plots',
            'print_outputs',
            'take_rss_model_outputs',
            'force_cpu',
            'max_rows_per_model_call',
        )
    }

    def _append_pair(epoch, tag):
        pairs.append({
            **base_dict,
            'epoch': epoch,
            'tag': tag,
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
        step_ri(f'Pair {idx + 1}/{total_pairs}')
        for key, val in pair.items():
            print(f'{key}: {val}')
        dec_print_indent()
        divider()
        model_test(pair)
