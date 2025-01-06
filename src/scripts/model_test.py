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
from utils.constants import (ANALYSIS_P, EXTRA_VARS_F, MAE, MSE, PROC_DATA_P,
                             RESULTS_F, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.model import Model
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa
from utils.plots.plot_comparison_scatter_grid_v2 import plot_comparison_scatter_grid_v2  # noqa
from utils.plots.plot_zernike_cross_coupling_animation import plot_zernike_cross_coupling_animation  # noqa
from utils.plots.plot_zernike_cross_coupling_mat_animation import plot_zernike_cross_coupling_mat_animation  # noqa
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.plots.plot_zernike_total_cross_coupling import plot_zernike_total_cross_coupling  # noqa
from utils.printing_and_logging import step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.stats_and_error import mae, mse, rss
from utils.terminate_with_message import terminate_with_message
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_test_parser(subparsers):
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
    subparser.add_argument(
        '--print-outputs',
        action='store_true',
        help='print out the truth and model outputs',
    )
    subparser.add_argument(
        '--take-rss-model-outputs',
        action='store_true',
        help='print out the RSS of the model outputs',
    )
    shared_argparser_args(subparser, ['force_cpu'])


def model_test(cli_args):
    title('Model test script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']

    model = Model(tag, epoch, force_cpu=cli_args.get('force_cpu'))
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
        if model.base_field is None:
            terminate_with_message('Base field not present in extra variables')
        inputs = inputs - model.base_field

    if cli_args.get('inputs_need_norm'):
        step_ri('Normalizing the inputs')
        inputs = model.norm_data(inputs)

    step_ri('Calling the model and obtaining its outputs')
    outputs_model = model(inputs)

    step_ri('Denormalizing the outputs')
    # Denormalize the outputs
    outputs_model = model.denorm_data(outputs_model)
    # Testing output data should already be denormalized
    outputs_truth = testing_dataset.get_outputs()

    # Print the results to the console
    if cli_args.get('print_outputs'):
        step_ri('Results')
        print('Outputs truth:')
        print(outputs_truth)
        print('Outputs model:')
        print(np.array2string(outputs_model, separator=', ', precision=3))

    # This only makes sense if a test is being done on a row without aberrations
    if cli_args.get('take_rss_model_outputs'):
        print('Model outputs RSS: ', rss(outputs_model))

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
        # plot_comparison_scatter_grid(
        #     outputs_model,
        #     outputs_truth,
        #     n_rows,
        #     n_cols,
        #     plot_title,
        #     plot_identifier,
        #     f'{analysis_path}/scatter.png',
        # )
        # step_ri('Generating density scatter plot')
        # plot_comparison_scatter_grid(
        #     outputs_model,
        #     outputs_truth,
        #     n_rows,
        #     n_cols,
        #     plot_title,
        #     plot_identifier,
        #     f'{analysis_path}/density_scatter.png',
        #     plot_density=True,
        # )

        step_ri('Generating density scatter plot')
        plot_comparison_scatter_grid_v2(
            outputs_model,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            2,
            f'{analysis_path}/density_scatter_v2.png',
            plot_density=15,
            filter_value=7e-8,
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
