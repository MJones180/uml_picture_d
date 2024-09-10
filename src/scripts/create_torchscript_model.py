"""
This script will convert a PyTorch model to a TorchScript model.
TorchScript allows for the model to be run in C++ instead of Python.
"""

import torch
from utils.constants import TRAINED_MODELS_P
from utils.model import Model
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_hdf_ds_loader import DSLoaderHDF


def create_torchscript_model_parser(subparsers):
    subparser = subparsers.add_parser(
        'create_torchscript_model',
        help='covert a PyTorch model to a TorchScript model',
    )
    subparser.set_defaults(main=create_torchscript_model)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'validation_ds',
        help=('name of the validation dataset, data should already be '
              'normalized'),
    )


def create_torchscript_model(cli_args):
    title('Create TorchScript model script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']
    # Ensure everything is loaded in on the CPU, things get funky otherwise
    model_obj = Model(tag, epoch, force_cpu=True)
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model_obj.epoch

    step_ri('Loading in the inputs from the validation dataset')
    inputs = DSLoaderHDF(cli_args['validation_ds']).get_inputs_torch()

    step_ri('Tracing model')
    traced_model = torch.jit.trace(
        model_obj.model,
        # We can pass through many example inputs, but the model seems to do
        # fine when only one sample is passed through.
        example_inputs=inputs[1][None, :, :, :],
    )

    step_ri('Comparing traced model to native PyTorch model')
    # Run a comparison to see how the native PyTorch version compares to the
    # traced TorchScript version. Use at most 1k rows.
    comp_inputs = inputs[:1000]
    comp_rows = comp_inputs.shape[0]
    pytorch_out = torch.from_numpy(model_obj(comp_inputs))
    traced_out = traced_model(comp_inputs)
    avg_diff = (pytorch_out - traced_out).mean()
    print(f'Average difference of {avg_diff} over {comp_rows} rows')

    step_ri('Saving traced model')
    traced_model_path = f'{TRAINED_MODELS_P}/{tag}_epoch{epoch}.pt'
    print(f'Location: {traced_model_path}')
    traced_model.save(traced_model_path)
