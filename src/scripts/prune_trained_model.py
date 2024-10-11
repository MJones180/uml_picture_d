"""
This script will prune a trained model.
https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

Only weights from convolutional and linear layers are supported in this script.

The pruned models that are saved will not be added to the `tag_lookup.json`.
"""
import torch
import torch.nn.utils.prune as prune
from utils.constants import ARGS_F, TRAINED_MODELS_P
from utils.json import json_write
from utils.model import Model
from utils.path import copy_dir
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args


def prune_trained_model_parser(subparsers):
    subparser = subparsers.add_parser(
        'prune_trained_model',
        help='prune a trained model',
    )
    subparser.set_defaults(main=prune_trained_model)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'new_tag',
        help='tag of the model after pruning',
    )
    shared_argparser_args(subparser, ['force_cpu'])

    prune_group = subparser.add_mutually_exclusive_group()
    prune_group.add_argument(
        '--local-prune-weights',
        nargs=2,
        metavar=('[convolutional percentage]', '[dense percentage]'),
        help=('percentage of weights to zero out for each convolutional and '
              'dense layer - this is performed locally; 0 means no pruning '
              'and 100 means all weights set to zero'),
    )
    prune_group.add_argument(
        '--global-prune-weights',
        nargs=2,
        metavar=('[convolutional percentage]', '[dense percentage]'),
        help=('percentage of weights to remove across all convolutional and '
              'dense layers - this is performed globally; 0 means no pruning '
              'and 100 means all weights set to zero'),
    )


def prune_trained_model(cli_args):
    title('Prune trained model script')

    step_ri('Loading in the model')
    tag = cli_args['tag']
    epoch = cli_args['epoch']
    model_obj = Model(tag, epoch, force_cpu=cli_args.get('force_cpu'))
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model_obj.epoch
    # Grab the actual PyTorch model
    model = model_obj.model

    new_tag = cli_args['new_tag']
    step_ri(f'Copying over the model to {new_tag}')
    original_model_dir = f'{TRAINED_MODELS_P}/{tag}'
    pruned_model_dir = f'{TRAINED_MODELS_P}/{new_tag}'
    copy_dir(original_model_dir, pruned_model_dir, True)
    print(f'Model directory copied to {pruned_model_dir}')

    step_ri('Saving all CLI args')
    json_write(f'{pruned_model_dir}/prune_{ARGS_F}', cli_args)

    # This is done as a function since sometimes an existing reference to a
    # layer after it has been changed is weird
    def _obtain_layers(linear=False):
        # Can only be linear or convolutional
        layer_type = torch.nn.Linear if linear else torch.nn.Conv2d
        return [
            module for name, module in model.named_modules()
            if isinstance(module, layer_type)
        ]

    def _parser_weight_amounts(arg):
        arg_weights = cli_args.get(arg)
        if arg_weights is not None:
            return [int(x) / 100 for x in arg_weights]

    local_prune_weights = _parser_weight_amounts('local_prune_weights')
    if local_prune_weights is not None:
        step_ri('Using local pruning of the weights')
        # Convert the weights to float ratios
        convolution_amount, linear_amount = local_prune_weights
        print(f'Convolutional amount: {convolution_amount}')
        print(f'Linear amount: {linear_amount}')

        def _prune_each_layer(linear=False):
            amount = linear_amount if linear else convolution_amount
            for module in _obtain_layers(linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')

        # Prune the convolutional layers
        _prune_each_layer()
        # Prune the linear layers
        _prune_each_layer(True)

    global_prune_weights = _parser_weight_amounts('global_prune_weights')
    if global_prune_weights is not None:
        step_ri('Using global pruning of the weights')
        # Convert the weights to float ratios
        convolution_amount, linear_amount = global_prune_weights
        print(f'Convolutional amount: {convolution_amount}')
        print(f'Linear amount: {linear_amount}')

        def _prune_each_layer(linear=False):
            amount = linear_amount if linear else convolution_amount
            layers = [(layer, 'weight') for layer in _obtain_layers(linear)]
            prune.global_unstructured(
                layers,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
            # Need to save the changes
            for module in _obtain_layers(linear):
                prune.remove(module, 'weight')

        # Prune the convolutional layers
        _prune_each_layer()
        # Prune the linear layers
        _prune_each_layer(True)

    step_ri('Saving the pruned model')
    torch.save(model.state_dict(), f'{pruned_model_dir}/epoch_{epoch}')
