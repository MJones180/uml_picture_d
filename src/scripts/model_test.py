"""
This script tests a model's performance against a testing dataset.

Any prior results for a given epoch will be deleted.

When generating any of the Zernike-related plots, it is expected that
the `testing_ds` was simulated with the `sim_data` script using the
`--fixed-amount-per-zernike-range` arg and preprocessed with the
`preprocess_data_bare` script.
"""

import numpy as np
import torch
from utils.constants import (ANALYSIS_P, BASE_INT_FIELD, EXTRA_VARS_F,
                             INPUT_MAX_MIN_DIFF, INPUT_MIN_X, MAE, MSE,
                             OUTPUT_MAX_MIN_DIFF, OUTPUT_MIN_X, PROC_DATA_P,
                             RESULTS_F, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.model import Model
from utils.norm import min_max_denorm, min_max_norm
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa
from utils.plots.plot_zernike_cross_coupling_animation import plot_zernike_cross_coupling_animation  # noqa
from utils.plots.plot_zernike_cross_coupling_mat_animation import plot_zernike_cross_coupling_mat_animation  # noqa
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.plots.plot_zernike_total_cross_coupling import plot_zernike_total_cross_coupling  # noqa
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.stats_and_error import mae, mse
from utils.terminate_with_message import terminate_with_message
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_test_parser(subparsers):
    """
    Example commands:
        python3 main.py model_test v1a last test_fixed_10nm_gl
        python3 main.py model_test fixed_10nm_gl last \
            fixed_50nm_range_processed \
            --inputs-need-norm \
            --scatter-plot 5 5 \
            --zernike-plots
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
        '--inputs-need-diff',
        action='store_true',
        help='the inputs need to subtract the base field to get the diff',
    )
    subparser.add_argument(
        '--inputs-need-norm',
        action='store_true',
        help='the inputs need to be normalized',
    )
    subparser.add_argument(
        '--scatter-plot',
        nargs=2,
        metavar=('[n_rows]', '[n_cols]'),
        help='generate a scatter plot',
    )
    subparser.add_argument(
        '--zernike-plots',
        action='store_true',
        help='generate the Zernike plots',
    )


def model_test(cli_args):
    title('Model test script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']

    model = Model(tag, epoch)
    norm_values = model.norm_values
    # Grab the epoch number so that the output directory has what epoch it is
    epoch = model.epoch

    step_ri('Creating the analysis directory')
    testing_ds_tag = cli_args['testing_ds']
    analysis_path = f'{ANALYSIS_P}/{testing_ds_tag}/{tag}_epoch_{epoch}'
    analysis_path = get_abs_path(analysis_path)
    delete_dir(analysis_path, quiet=True)
    make_dir(analysis_path)

    step_ri('Loading in the testing dataset')
    testing_dataset = DSLoaderHDF(testing_ds_tag)
    inputs = testing_dataset.get_inputs()
    extra_vars_path = f'{PROC_DATA_P}/{testing_ds_tag}/{EXTRA_VARS_F}'
    zernike_terms = read_hdf(extra_vars_path)[ZERNIKE_TERMS]
    print(f'Using zernike terms: {zernike_terms}')

    # If the model was trained on the difference between the aberrated and the
    # base field, then the model must be tested on the same thing. If the data
    # is not already of the difference, then the base field must be subtracted
    # off before normalization occurs.
    if cli_args.get('inputs_need_diff'):
        step_ri('Taking the diff of the inputs')
        extra_vars = model.extra_vars
        if BASE_INT_FIELD not in list(extra_vars):
            terminate_with_message('Base field not present in raw ds info')
        inputs = inputs - extra_vars[BASE_INT_FIELD]

    if cli_args.get('inputs_need_norm'):
        step_ri('Normalizing the inputs')
        inputs = min_max_norm(
            inputs,
            norm_values[INPUT_MAX_MIN_DIFF],
            norm_values[INPUT_MIN_X],
        )

    step_ri('Calling the model and obtaining its outputs')
    outputs_model = model(torch.from_numpy(inputs))

    step_ri('Denormalizing the outputs')
    # Denormalize the outputs
    outputs_model = min_max_denorm(
        outputs_model,
        norm_values[OUTPUT_MAX_MIN_DIFF],
        norm_values[OUTPUT_MIN_X],
    )
    # Testing output data should already be denormalized
    outputs_truth = testing_dataset.get_outputs()

    step_ri('Computing the MAE and MSE')
    mae_val = mae(outputs_truth, outputs_model)
    mse_val = mse(outputs_truth, outputs_model)
    print(f'Model MAE: {mae_val}')
    print(f'Model MSE: {mse_val}')

    step_ri('Writing results to HDF')
    out_file_path = f'{analysis_path}/{RESULTS_F}'
    print(f'File location: {out_file_path}')
    out_data = {
        'outputs_truth': outputs_truth,
        'outputs_model': outputs_model,
        MAE: mae_val,
        MSE: mse_val,
    }
    HDFWriteModule(out_file_path).create_and_write_hdf_simple(out_data)

    plot_title = 'Neural Network'
    plot_identifier = f'{tag}, epoch {epoch}'

    scatter_plot = cli_args.get('scatter_plot')
    if scatter_plot is not None:
        step_ri('Generating scatter plot')
        n_rows, n_cols = [int(arg) for arg in scatter_plot]
        plot_comparison_scatter_grid(
            outputs_model,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            f'{analysis_path}/scatter.png',
        )

    if cli_args.get('zernike_plots'):
        nrows = outputs_truth.shape[0]
        zernike_count = len(zernike_terms)
        if nrows % zernike_count != 0:
            terminate_with_message('Data is in the incorrect shape for '
                                   'the Zernike plot(s)')

        def _split(data):
            # Split the data so that each group (first dim) consists of all
            # the Zernike terms perturbed by a given amount
            return np.array(np.split(data, nrows // zernike_count))

        # Groups will have the shape (rms pert, zernike terms, zernike terms)
        outputs_truth_gr = _split(outputs_truth)
        outputs_model_gr = _split(outputs_model)

        # It is assumed that the truth terms all have the same perturbation
        # for each group and that there are only perturbations along the main
        # diagonal. Therefore, each group (first dim) should be equivalent to
        # `perturbation * identity matrix`. Due to this, we can simply obtain
        # the list of all RMS perturbations.
        perturbation_grid = outputs_truth_gr[:, 0, 0]

        step_ri('Generating a Zernike response plot')
        plot_zernike_response(
            zernike_terms,
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_response.png',
        )

        step_ri('Generating a Zernike total cross coupling plot')
        plot_zernike_total_cross_coupling(
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/total_cross_coupling.png',
        )

        step_ri('Generating a Zernike cross coupling animation')
        plot_zernike_cross_coupling_animation(
            zernike_terms,
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_cross_coupling.gif',
        )

        step_ri('Generating a Zernike cross coupling matrix animation')
        plot_zernike_cross_coupling_mat_animation(
            zernike_terms,
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_cross_coupling_mat.gif',
        )
