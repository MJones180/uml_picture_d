"""
This script will convert a PyTorch model to a TorchScript model.
TorchScript allows for the model to be run in C++ instead of Python.

In addition to the model, the norm and base field data are also saved.
The following assumptions are made: the model is trained on the difference (so
a base field does exist), there is only one norm value used for all the inputs,
and each output has its own norm value.
"""

import numpy as np
import torch
from utils.constants import (BASE_INT_FIELD, INPUT_MAX_MIN_DIFF, INPUT_MIN_X,
                             OUTPUT_MAX_MIN_DIFF, OUTPUT_MIN_X,
                             TRAINED_MODELS_P)
from utils.model import Model
from utils.path import make_dir
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
    output_dir = f'{TRAINED_MODELS_P}/ts_{tag}_epoch{epoch}'
    make_dir(output_dir)
    traced_model_path = f'{output_dir}/model.pt'
    print(f'Location: {traced_model_path}')
    traced_model.save(traced_model_path)

    model_vars = model_obj.extra_vars

    step_ri('Saving normalization data')
    norm_data_path = f'{output_dir}/norm_data.txt'
    print(f'Location: {norm_data_path}')
    with open(norm_data_path, 'w') as out_file:

        def _write_data(data):
            np.savetxt(out_file, [data], fmt='%.8f')

        for key in (INPUT_MAX_MIN_DIFF, INPUT_MIN_X):
            _write_data(model_vars[key][()])
        for key in (OUTPUT_MAX_MIN_DIFF, OUTPUT_MIN_X):
            _write_data(model_vars[key][:])

    step_ri('Saving base intensity field data')
    base_field_path = f'{output_dir}/base_field.txt'
    print(f'Location: {base_field_path}')
    np.savetxt(base_field_path, model_vars[BASE_INT_FIELD][0], fmt='%.8f')

    step_ri('Saving the info file')
    readme_path = f'{output_dir}/README.txt'
    print(f'Location: {readme_path}')
    with open(readme_path, 'w') as out_file:
        out_file.write(
            f'{traced_model_path}:\n\tThe traced TorchScript model.\n'
            f'{norm_data_path}:\n\tContains the normalization info for the '
            'model. The lines in order are:\n'
            '\t\tinput max min diff\n\t\tinput min x\n'
            '\t\toutput max min diff\n\t\toutput min x\n'
            '\tThere is one norm value for all the inputs and a norm value '
            'for each of the output values.\n'
            f'{base_field_path}:\n\tContains the base field that should be '
            'subtracted off. This field of course has only one channel.')
