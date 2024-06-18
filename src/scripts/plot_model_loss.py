"""
Plot out training and validation loss at each epoch for a trained model.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ARGS_F, EPOCH_LOSS_F, RANDOM_P, TRAINED_MODELS_P
from utils.json import json_load
from utils.printing_and_logging import title
from utils.shared_argparser_args import shared_argparser_args


def plot_model_loss_parser(subparsers):
    subparser = subparsers.add_parser(
        'plot_model_loss',
        help='plot training loss of a model',
    )
    subparser.set_defaults(main=plot_model_loss)
    shared_argparser_args(subparser, ['tag'])


def plot_model_loss(cli_args):
    title('Plot model loss script')

    tag = cli_args['tag']
    base_path = f'{TRAINED_MODELS_P}/{tag}'
    # Columns of the datafile are epochs, train_loss, val_loss. If there is a
    # fourth column, it is the post training loss.
    data = np.loadtxt(f'{base_path}/{EPOCH_LOSS_F}', skiprows=1, delimiter=',')
    # Load in the loss function used
    loss_func = json_load(f'{base_path}/{ARGS_F}')['loss']

    # Create and save the plot
    fig, ax = plt.subplots()
    ax.set_title(f'Epoch Loss During Training\n{tag}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({loss_func})')
    epochs = data[:, 0]
    ax.plot(epochs, data[:, 1], 'b', label='Train Loss')
    ax.plot(epochs, data[:, 2], 'r', label='Validation Loss')
    if data.shape[1] == 4:
        ax.plot(epochs, data[:, 3], 'g', label='Post Train Loss')
    plt.legend()
    path = f'{RANDOM_P}/model_loss_{tag}.png'
    print(f'Saving plot to {path}')
    plt.savefig(path, bbox_inches='tight')
