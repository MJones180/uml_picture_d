"""
Plot out training and validation loss at each epoch for a trained model.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ANALYSIS_P, ARGS_F, EPOCH_LOSS_F, TRAINED_MODELS_P
from utils.json import json_load
from utils.printing_and_logging import title
from utils.shared_argparser_args import shared_argparser_args


def plot_model_loss_parser(subparsers):
    """
    Example command:
        python3 main.py plot_model_loss test
    """
    subparser = subparsers.add_parser(
        'plot_model_loss',
        help='plot loss of a model',
    )
    subparser.set_defaults(main=plot_model_loss)
    shared_argparser_args(subparser, ['tag'])


def plot_model_loss(cli_args):
    title('Plot model loss script')

    tag = cli_args['tag']
    base_path = f'{TRAINED_MODELS_P}/{tag}'
    data = np.loadtxt(f'{base_path}/{EPOCH_LOSS_F}', skiprows=1, delimiter=',')
    # Columns of the datafile are epochs, train_loss, val_loss
    epochs = data[:, 0]
    train_loss = data[:, 1]
    val_loss = data[:, 2]
    # Load in the loss function used
    loss_func = json_load(f'{base_path}/{ARGS_F}')['loss']

    # Create and save the plot
    fig, ax = plt.subplots()
    ax.set_title('Epoch Loss During Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({loss_func})')
    ax.plot(epochs, train_loss, 'r', label='Train Loss')
    ax.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.legend()
    plt.savefig(f'{ANALYSIS_P}/model_loss_{tag}.png', bbox_inches='tight')
