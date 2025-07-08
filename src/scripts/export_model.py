"""
This script will export a PyTorch model to both a TorchScript model and an
ONNX model. TorchScript allows for the model to be run in C++ instead of Python.
ONNX allows for the model to be run in many different runtimes (including C/C++)
with an added boost to the inference speed.

In addition to the model, the norm and base field data are also saved.
The following assumptions are made:
    - The model is trained on the difference (so a base field does exist)
    - There is only one norm value used for all the inputs (if any)
    - Each output has its own norm value
    - The model expects one input array and outputs a single array

Code for converting to ONNX and using ONNX Runtime taken from:
    https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
"""

import numpy as np
import onnxruntime
import torch
from utils.benchmark_nn import benchmark_nn
from utils.constants import (BASE_INT_FIELD, INPUT_MAX_MIN_DIFF, INPUT_MIN_X,
                             NORM_RANGE_ONES, OUTPUT_MAX_MIN_DIFF,
                             OUTPUT_MIN_X, TRAINED_MODELS_P)
from utils.model import Model
from utils.norm import min_max_denorm
from utils.path import make_dir
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_hdf_ds_loader import DSLoaderHDF


def export_model_parser(subparsers):
    subparser = subparsers.add_parser(
        'export_model',
        help='export a PyTorch model',
    )
    subparser.set_defaults(main=export_model)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'validation_ds',
        help=('name of the validation dataset, data should already be norm '
              'and have the base field subtracted off'),
    )
    subparser.add_argument(
        '--benchmark',
        type=int,
        help='benchmark exported models by running n rows (will use CPU only)',
    )


def export_model(cli_args):
    title('Export model script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']
    # Ensure everything is loaded in on the CPU, things get funky otherwise
    model_obj = Model(tag, epoch, force_cpu=True)
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model_obj.epoch
    pytorch_model = model_obj.model
    model_vars = model_obj.extra_vars

    step_ri('Loading in the inputs from the validation dataset')
    validation_ds_data = DSLoaderHDF(cli_args['validation_ds'])
    inputs = validation_ds_data.get_inputs_torch()
    outputs = validation_ds_data.get_outputs_torch()
    first_input_row = inputs[1][None, :, :, :]
    first_output_row = outputs[1].numpy()
    # Just use the first 1k rows
    inputs = inputs[:1000]

    step_ri('Creating output directory')
    output_dir = f'{TRAINED_MODELS_P}/exported_{tag}_epoch{epoch}'
    make_dir(output_dir)
    output_dir_ex = f'{output_dir}/example_data'
    make_dir(output_dir_ex)

    step_ri('Saving the input line (after norm and base field subtraction)')
    input_line_path = f'{output_dir_ex}/first_input_row_norm.txt'
    np.savetxt(input_line_path, first_input_row[0][0], fmt='%e')

    def denorm_data(data):
        return min_max_denorm(
            data,
            model_vars[OUTPUT_MAX_MIN_DIFF],
            model_vars[OUTPUT_MIN_X],
            model_vars[NORM_RANGE_ONES],
        )

    step_ri('Saving the true output line (no norm)')
    out_line_norm_path = f'{output_dir_ex}/first_output_row_norm_truth.txt'
    np.savetxt(out_line_norm_path, first_output_row, fmt='%e')

    step_ri('Saving the true output line (no norm)')
    out_line_path = f'{output_dir_ex}/first_output_row_truth.txt'
    np.savetxt(out_line_path, denorm_data(first_output_row), fmt='%e')

    # =================
    # TorchScript model
    # =================

    step_ri('Exporing TorchScript model')

    step('Tracing model')
    ts_model = torch.jit.trace(
        pytorch_model,
        # We can pass through many example inputs, but the model seems to do
        # fine when only one sample is passed through.
        example_inputs=first_input_row,
    )
    dec_print_indent()

    step('Saving model')
    ts_model_path = f'{output_dir}/model.pt'
    print(f'Location: {ts_model_path}')
    ts_model.save(ts_model_path)
    dec_print_indent()

    step('Comparing to native PyTorch model')
    # Run a comparison to see how the native PyTorch version compares to the
    # traced TorchScript version.
    row_count = inputs.shape[0]
    pytorch_model_out = model_obj(inputs)
    ts_model_out = ts_model(inputs).detach().numpy()
    avg_diff = np.sum(np.abs(pytorch_model_out - ts_model_out)) / row_count
    print(f'Average difference of {avg_diff:0.8f} per row')

    step_ri('Saving the TorchScript output line (before denormalization)')
    out_line_ts_norm_path = f'{output_dir_ex}/first_output_row_norm_ts.txt'
    np.savetxt(out_line_ts_norm_path, ts_model_out[1], fmt='%e')

    step_ri('Saving the TorchScript output line (after denormalization)')
    out_line_ts_path = f'{output_dir_ex}/first_output_row_ts.txt'
    np.savetxt(out_line_ts_path, denorm_data(ts_model_out[1]), fmt='%e')

    # ==========
    # ONNX model
    # ==========

    step_ri('Exporting ONNX model')

    step('Saving model')
    onnx_model_path = f'{output_dir}/model.onnx'
    print(f'Location: {onnx_model_path}')
    torch.onnx.export(
        pytorch_model,
        first_input_row,
        onnx_model_path,
        # Name of the input and output array
        input_names=['input'],
        output_names=['output'],
    )
    dec_print_indent()

    step('Comparing to native PyTorch model')
    # Load in the ONNX model and use only the CPU
    onnx_model = onnxruntime.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider'],
    )

    def _run_onnx(row):
        # To run the ONNX model, a dictionary of the inputs must be passed in.
        # Additionally, it seems that only batch sizes of one are allowed.
        return onnx_model.run(None, {'input': row.cpu().numpy()})

    # Run the ONNX model on the comparison rows
    onnx_model_out = np.array(
        [_run_onnx(row[None, :, :, :]) for row in inputs])
    # Remove the dimensions of size 1
    onnx_model_out = np.squeeze(onnx_model_out)
    avg_diff = np.sum(np.abs(pytorch_model_out - onnx_model_out)) / row_count
    print(f'Average difference of {avg_diff:0.8f} per row')

    step_ri('Saving the ONNX output line (before denormalization)')
    out_line_onnx_norm_path = f'{output_dir_ex}/first_output_row_norm_onnx.txt'
    np.savetxt(out_line_onnx_norm_path, onnx_model_out[1], fmt='%e')

    step_ri('Saving the ONNX output line (after denormalization)')
    out_line_onnx_path = f'{output_dir_ex}/first_output_row_onnx.txt'
    np.savetxt(out_line_onnx_path, denorm_data(onnx_model_out[1]), fmt='%e')

    # ====================
    # Save auxiliary files
    # ====================

    step_ri('Saving auxiliary data')

    step('Saving normalization data')
    norm_data_path = f'{output_dir}/norm_data.txt'
    print(f'Location: {norm_data_path}')
    with open(norm_data_path, 'w') as out_file:

        def _write_data(data):
            np.savetxt(out_file, [data], fmt='%e')

        for key in (INPUT_MAX_MIN_DIFF, INPUT_MIN_X):
            if key in model_vars:
                _write_data(model_vars[key][()])
            else:
                _write_data(0)
        for key in (OUTPUT_MAX_MIN_DIFF, OUTPUT_MIN_X):
            if key in model_vars:
                _write_data(model_vars[key][:])
            else:
                _write_data(0)
    dec_print_indent()

    step('Saving base intensity field data')
    base_field_path = f'{output_dir}/base_field.txt'
    print(f'Location: {base_field_path}')
    np.savetxt(base_field_path, model_vars[BASE_INT_FIELD][0], fmt='%e')
    dec_print_indent()

    step('Saving the info file')
    readme_path = f'{output_dir}/README.txt'
    print(f'Location: {readme_path}')
    with open(readme_path, 'w') as out_file:
        out_file.write(
            f'{ts_model_path}:\n\tThe TorchScript model.\n'
            f'{onnx_model_path}:\n\tThe ONNX model.\n'
            f'{norm_data_path}:\n\tContains the normalization info for the '
            'model. The lines in order are:\n'
            '\t\tinput max min diff\n\t\tinput min x\n'
            '\t\toutput max min diff\n\t\toutput min x\n'
            '\tThere is one norm value for all the inputs and a norm value '
            'for each of the output values.\n'
            f'{base_field_path}:\n\tContains the base field that should be '
            'subtracted off. This field of course has only one channel.\n'
            f'{input_line_path}:\n\tExample input row after norm is done and '
            'base field is subtracted.\n'
            f'{out_line_norm_path}:\n\tExample truth output row '
            'before denorm is done.\n'
            f'{out_line_path}:\n\tExample truth output row '
            'after denorm is done.\n'
            f'{out_line_ts_norm_path}:\n\tExample TorchScript output row '
            'before denorm is done.\n'
            f'{out_line_ts_path}:\n\tExample TorchScript output row '
            'after denorm is done.\n'
            f'{out_line_onnx_norm_path}:\n\tExample ONNX output row '
            'before denorm is done.\n'
            f'{out_line_onnx_path}:\n\tExample ONNX output row '
            'after denorm is done.')
    dec_print_indent()

    # =========
    # Benchmark
    # =========

    step_ri('Benchmarking models')
    input_data = model_obj.network.example_input()
    iterations = cli_args['benchmark']
    if iterations is not None:
        benchmark_nn(iterations, lambda: model_obj(input_data), 'PyTorch')
        benchmark_nn(iterations, lambda: ts_model(input_data), 'TorchScript')
        benchmark_nn(iterations, lambda: _run_onnx(first_input_row), 'ONNX')
