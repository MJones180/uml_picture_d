"""
This script will prune a trained model to improve inference speeds.
https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

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
              'dense layer'),
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

    local_prune_weights = cli_args.get('local_prune_weights')
    if local_prune_weights is not None:
        step_ri('Using local pruning of the weights')
        # Convert the weights to float ratios
        convolution_amount, linear_amount = [
            int(x) / 100 for x in local_prune_weights
        ]
        print(f'Convolutional amount: {convolution_amount}')
        print(f'Linear amount: {linear_amount}')

        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=convolution_amount,
                )
                prune.remove(module, 'weight')
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=linear_amount,
                )
                prune.remove(module, 'weight')

    step_ri('Saving the pruned model')
    torch.save(model.state_dict(), f'{pruned_model_dir}/epoch_{epoch}')
