"""
This script tests a model's performance against a testing dataset.

Any prior results for a given epoch will be deleted.
"""

import numpy as np
import torch
from utils.constants import (ANALYSIS_P, MAE, MSE, OUTPUT_MIN_X,
                             OUTPUT_MAX_MIN_DIFF, RESULTS_F)
from utils.hdf_read_and_write import HDFWriteModule
from utils.model import Model
from utils.norm import min_max_denorm
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_test_parser(subparsers):
    """
    Example commands:
        python3 main.py model_test v1a last test_fixed_10nm_gl 5 5
    """
    subparser = subparsers.add_parser(
        'model_test',
        help='test a trained model',
    )
    subparser.set_defaults(main=model_test)
    shared_argparser_args(subparser, ['tag', 'epoch'])
    subparser.add_argument(
        'testing_ds',
        help=('name of the testing dataset, will use the norm values from the '
              'trained model - NOT from the testing dataset directly, outputs '
              'should already be denormalized'),
    )
    subparser.add_argument(
        'n_rows',
        type=int,
        help='number of rows in the plot for the output comparison',
    )
    subparser.add_argument(
        'n_cols',
        type=int,
        help='number of cols in the plot for the output comparison',
    )


def model_test(cli_args):
    title('Model test script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']

    model = Model(tag, epoch)
    norm_values = model.get_norm_values()
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model.get_epoch()

    step_ri('Creating the analysis directory')
    analysis_path = f'{ANALYSIS_P}/{tag}_epoch_{epoch}'
    delete_dir(analysis_path, quiet=True)
    make_dir(analysis_path)

    step_ri('Loading in the testing dataset')
    testing_dataset = DSLoaderHDF(cli_args['testing_ds'])

    step_ri('Calling the model and obtaining its outputs')
    outputs_model = model(testing_dataset.get_inputs_torch())

    step_ri('Denormalizing the outputs')
    # Denormalize the outputs
    outputs_model_denormed = min_max_denorm(
        outputs_model,
        norm_values[OUTPUT_MAX_MIN_DIFF],
        norm_values[OUTPUT_MIN_X],
    )
    # Testing output data should already be denormalized
    outputs_truth = testing_dataset.get_outputs()

    step_ri('Computing the MAE and MSE')

    def _compute_loss(loss_func):
        return loss_func(reduction='none')(
            torch.from_numpy(outputs_truth),
            torch.from_numpy(outputs_model_denormed)).numpy()

    mae = _compute_loss(torch.nn.L1Loss)
    mse = _compute_loss(torch.nn.MSELoss)
    print(f'MAE: {np.mean(mae)}')
    print(f'MSE: {np.mean(mse)}')

    step_ri('Writing results to HDF')
    out_file_path = f'{analysis_path}/{RESULTS_F}'
    print(f'File location: {out_file_path}')
    HDFWriteModule(out_file_path).create_and_write_hdf_simple({
        'outputs_truth': outputs_truth,
        'outputs_model_denormed': outputs_model_denormed,
        MAE: mae,
        MSE: mse,
    })

    step_ri('Generating plot')
    plot_comparison_scatter_grid(
        outputs_model_denormed,
        outputs_truth,
        cli_args['n_rows'],
        cli_args['n_cols'],
        get_abs_path(f'{analysis_path}/comparisons.png'),
    )
