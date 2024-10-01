"""
This script is used to visualize the convolutional layers in a CNN.

Turns out, visualizing the convolutions inside of a multi-regression CNN is
difficult. Most papers discuss either visualizing classification based CNNs or
single-regression based CNNs. Not sure how to go about providing better
visualizations for these models.

Commands to run this script:
    python3 main_scnp.py visualize_cnn cnn_comp_multi_v2_4 last \
        fixed_50nm_range_processed --inputs-need-norm --inputs-need-diff
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from utils.constants import (BASE_INT_FIELD, INPUT_MAX_MIN_DIFF, INPUT_MIN_X,
                             NORM_RANGE_ONES)
from utils.model import Model
from utils.norm import min_max_norm
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.terminate_with_message import terminate_with_message
from utils.torch_hdf_ds_loader import DSLoaderHDF


def visualize_cnn_parser(subparsers):
    subparser = subparsers.add_parser(
        'visualize_cnn',
        help='visualize the convolutional layers of a CNN',
    )
    subparser.set_defaults(main=visualize_cnn)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'dataset',
        help='name of the processed dataset',
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='index of the row to take the wavefront of',
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
    shared_argparser_args(subparser, ['force_cpu'])


def visualize_cnn(cli_args):
    title('Visualize cnn script')

    step_ri('Loading in model')
    tag = cli_args['tag']
    epoch = cli_args['epoch']
    model_obj = Model(tag, epoch, force_cpu=cli_args['force_cpu'])
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model_obj.epoch
    model = model_obj.model
    model_vars = model_obj.extra_vars
    device = model_obj.device

    layer_idx = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Sum over the input dimension so we can visualize what the output
            # kernels look like. As an example, if we are going from 5 features
            # to 10 features with a 3x3 kernel, we will have the shape of
            # (10, 5, 3, 3). We will sum over all the learned input kernels, so
            # we can see what each output kernel looks like.
            weights_summed = torch.sum(layer.weight.data, dim=1)
            # Need to add back in the a fourth dimension
            weights_summed = weights_summed[:, None, :, :]
            grid = torchvision.utils.make_grid(
                weights_summed,
                normalize=True,
                padding=1,
            )
            layer_idx += 1
            plt.figure()
            plt.title(f'Conv2D Layer {layer_idx}')
            plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
            plt.show()

    step_ri('Loading in the dataset')
    dataset = cli_args['dataset']
    data = DSLoaderHDF(dataset)
    inputs = data.get_inputs()
    step_ri(dataset)
    print('Number of rows: ', len(data))
    print('Input shape: ', inputs.shape[1:])
    if cli_args.get('inputs_need_diff'):
        step_ri('Taking the diff of the inputs')
        if BASE_INT_FIELD not in list(model_vars):
            terminate_with_message('Base field not present in extra variables')
        inputs = inputs - model_vars[BASE_INT_FIELD]
    # Check if the data was normalized between -1 and 1
    norm_range_ones = (model_vars[NORM_RANGE_ONES][()]
                       if NORM_RANGE_ONES in model_vars else False)
    if cli_args.get('inputs_need_norm'):
        step_ri('Normalizing the inputs')
        inputs = min_max_norm(
            inputs,
            model_vars[INPUT_MAX_MIN_DIFF],
            model_vars[INPUT_MIN_X],
            norm_range_ones,
        )

    input_row = inputs[cli_args['row_idx']]
    plt.title('Original image')
    plt.imshow(input_row[0])
    plt.show()
    layer_idx = 0
    prev_layer_outputs = torch.from_numpy(input_row).to(device)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            conv_output = layer(prev_layer_outputs)
            prev_layer_outputs = conv_output
            conv_output = conv_output[:, None, :, :]
            grid = torchvision.utils.make_grid(
                conv_output,
                normalize=True,
                padding=1,
            )
            layer_idx += 1
            plt.figure()
            plt.title(f'Conv2D Layer {layer_idx}')
            plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
            plt.show()
            summed = torch.sum(conv_output, dim=0).detach().cpu().numpy()
            plt.figure()
            plt.title(f'Conv2D Layer {layer_idx} - Summed')
            plt.imshow(summed[0])
            plt.show()
