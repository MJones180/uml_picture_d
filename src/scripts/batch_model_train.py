"""
This script batch trains multiple models synchronously with one command in the
same process. No parallel or multiprocessed training is performed. If I gain
access to a computing cluster with more than a few cores, then I will go ahead
and implement that functionality.

This is a more opinionated version of the `model_train` script wrapped into one
convenient script for training lots of model configurations. Therefore, some
pre-existing features may not be present in this script.

The training and validation datasets must be the same for every combination.

Models will be saved like normal.
"""

from scripts.model_train import model_train
from utils.constants import LOSS_FUNCTIONS, OPTIMIZERS
from utils.printing_and_logging import dec_print_indent, divider, step_ri, title


def batch_model_train_parser(subparsers):
    """
    Example commands:
        python3 main.py batch_model_train \
            train_fixed_10nm_gl val_fixed_10nm_gl \
            global_v1_ 20 \
            --networks test \
            --losses mae --optimizers adam \
            --lrs 1e-3 1e-4 1e-5 \
            --batch-sizes 32 64 128 \
            --overwrite-existing --only-best-epoch --early-stopping 10
    """
    subparser = subparsers.add_parser(
        'batch_model_train',
        help='batch model train different configurations',
    )
    subparser.set_defaults(main=batch_model_train)
    subparser.add_argument(
        'training_ds',
        help='name of the training dataset',
    )
    subparser.add_argument(
        'validation_ds',
        help='name of the validation dataset',
    )
    subparser.add_argument(
        'base_tag',
        help='base of the tag, auto inc int appened to end',
    )
    subparser.add_argument(
        'epochs',
        type=int,
        help='total number of epochs to train for',
    )
    subparser.add_argument(
        '--networks',
        help='networks to use',
        nargs='+',
    )
    subparser.add_argument(
        '--losses',
        type=str.lower,
        choices=LOSS_FUNCTIONS.keys(),
        help='loss functions to use',
        nargs='+',
    )
    subparser.add_argument(
        '--optimizers',
        type=str.lower,
        choices=OPTIMIZERS.keys(),
        help='optimizer functions to use',
        nargs='+',
    )
    subparser.add_argument(
        '--lrs',
        type=float,
        help='learning rates of the optimizers',
        nargs='+',
    )
    subparser.add_argument(
        '--batch-sizes',
        type=int,
        help='number of samples per batch',
        nargs='+',
    )
    subparser.add_argument(
        '--overwrite-existing',
        action='store_true',
        help='if existing model with tag, delete before training',
    )
    subparser.add_argument(
        '--only-best-epoch',
        action='store_true',
        help=('only the best epoch as based on the validation '
              'dataset will be saved'),
    )
    subparser.add_argument(
        '--early-stopping',
        type=int,
        metavar='n',
        help=('stop training if performance does not improve after n epochs, '
              'this is based on the validation loss'),
    )
    subparser.add_argument(
        '--max-threads',
        type=int,
        help='limit the number of threads PyTorch can use',
    )


def batch_model_train(cli_args):
    title('Batch model train script')

    # Args that will be passed directly to `model_train`
    training_ds = cli_args['training_ds']
    validation_ds = cli_args['validation_ds']
    epochs = cli_args['epochs']
    overwrite_existing = cli_args['overwrite_existing']
    only_best_epoch = cli_args['only_best_epoch']
    early_stopping = cli_args['early_stopping']
    max_threads = cli_args['max_threads']

    # This arg will be appeneded with each index
    base_tag = cli_args['base_tag']

    # The args that can be lists of multiple values
    networks = cli_args['networks']
    losses = cli_args['losses']
    optimizers = cli_args['optimizers']
    lrs = cli_args['lrs']
    batch_sizes = cli_args['batch_sizes']

    step_ri('Computing all possible combinations of the provided args.')
    combinations = []
    current_idx = 1
    # I know this is disgusting, a giant structure of nested for loops, but it
    # is quick and easy and will work as expected without extra complication
    for network in networks:
        for loss in losses:
            for optimizer in optimizers:
                for lr in lrs:
                    for batch_size in batch_sizes:
                        # Since we are calling `model_train`, the arg names must
                        # match up to the names in that script. They are
                        # slightly different than in here.
                        combinations.append({
                            'tag': f'{base_tag}{current_idx}',
                            'training_ds': training_ds,
                            'validation_ds': validation_ds,
                            'network_name': network,
                            'loss': loss,
                            'optimizer': optimizer,
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': epochs,
                            'overwrite_existing': overwrite_existing,
                            'only_best_epoch': only_best_epoch,
                            'early_stopping': early_stopping,
                            'max_threads': max_threads,
                        })
                        current_idx += 1
    total_combos = len(combinations)
    print(f'There are a total of {total_combos} combinations')

    print('Will begin iteratively calling `model_train` for each combination')
    for idx, combination in enumerate(combinations):
        step_ri(f'Combination {idx + 1}/{total_combos}')
        for key, val in combination.items():
            print(f'{key}: {val}')
        dec_print_indent()
        divider()
        model_train(combination)
