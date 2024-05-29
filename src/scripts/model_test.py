"""
This script tests a model's performance against a testing dataset.

Any prior results for a given epoch will be deleted.

When using any of the Zernike-related plots, it is expected that the
`testing_ds` was simulated with the `sim_data` script using the
`--fixed-amount-per-zernike-range` arg and preprocessed with the
`preprocess_data_bare` script.
"""

import numpy as np
import torch
from utils.constants import (ANALYSIS_P, DS_RAW_INFO_F, INPUT_MAX_MIN_DIFF,
                             INPUT_MIN_X, MAE, MSE, OUTPUT_MAX_MIN_DIFF,
                             OUTPUT_MIN_X, PROC_DATA_P, RESULTS_F,
                             ZERNIKE_TERMS)
from utils.error import mae, mse
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_load
from utils.model import Model
from utils.norm import min_max_denorm, min_max_norm
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.printing_and_logging import step_ri, title
from utils.response_matrix import ResponseMatrix
from utils.shared_argparser_args import shared_argparser_args
from utils.terminate_with_message import terminate_with_message
from utils.torch_hdf_ds_loader import DSLoaderHDF


def model_test_parser(subparsers):
    """
    Example commands:
        python3 main.py model_test v1a last test_fixed_10nm_gl
        python3 main.py model_test fixed_10nm_gl last test_rand_50nm_s_gl \
            --scatter-plot 5 5 --zernike-response-gridded-plot
        python3 main.py model_test fixed_10nm_gl last \
            fixed_50nm_range_processed --zernike-response-gridded-plot \
            --inputs-need-norm \
            --response-matrix fixed_40nm
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
        '--inputs-need-norm',
        action='store_true',
        help='the inputs need to be normalized',
    )
    subparser.add_argument(
        '--response-matrix',
        help=('tag of the response matrix, the Zernike terms must align '
              'with the neural network model and testing dataset'),
    )
    subparser.add_argument(
        '--scatter-plot',
        nargs=2,
        metavar=('[n_rows]', '[n_cols]'),
        help='generate a scatter plot',
    )
    subparser.add_argument(
        '--zernike-response-gridded-plot',
        action='store_true',
        help='generate a Zernike response plot',
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
    testing_ds_tag = cli_args['testing_ds']
    testing_dataset = DSLoaderHDF(testing_ds_tag)
    inputs = testing_dataset.get_inputs()
    raw_info = json_load(f'{PROC_DATA_P}/{testing_ds_tag}/{DS_RAW_INFO_F}')
    zernike_terms = raw_info[ZERNIKE_TERMS]
    print(f'Using zernike terms: {zernike_terms}')

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

    mae_val = mae(outputs_truth, outputs_model, 'all')
    mse_val = mse(outputs_truth, outputs_model, 'all')
    print(f'Model MAE: {mae_val}')
    print(f'Model MSE: {mse_val}')

    response_matrix = cli_args.get('response_matrix')
    if response_matrix:
        response_matrix_obj = ResponseMatrix(response_matrix)
        # Ensure the Zernike terms matchup
        zernike_terms_resp_mat = response_matrix_obj.get_zernike_terms()
        if not np.array_equal(zernike_terms, zernike_terms_resp_mat):
            terminate_with_message('Zernike terms in response matrix do not '
                                   'match terms in dataset')
        # The response matrix does not work on normalized data
        inputs_no_norm = min_max_denorm(
            inputs,
            norm_values[INPUT_MAX_MIN_DIFF],
            norm_values[INPUT_MIN_X],
        )
        # Need to flatten out the pixels
        inputs_no_norm = inputs_no_norm.reshape(inputs_no_norm.shape[0], -1)
        outputs_resp_mat = response_matrix_obj(inputs_no_norm)

    step_ri('Writing results to HDF')
    out_file_path = f'{analysis_path}/{RESULTS_F}'
    print(f'File location: {out_file_path}')
    HDFWriteModule(out_file_path).create_and_write_hdf_simple({
        'outputs_truth': outputs_truth,
        'outputs_model': outputs_model,
        'outputs_response_matrix': outputs_resp_mat,
        # MAE and MSE are based on the neural network output, NOT the response
        # matrix output
        MAE: mae_val,
        MSE: mse_val,
    })

    scatter_plot = cli_args['scatter_plot']
    if scatter_plot is not None:
        step_ri('Generating scatter plot')
        n_rows, n_cols = [int(arg) for arg in scatter_plot]
        plot_comparison_scatter_grid(
            outputs_model,
            outputs_truth,
            n_rows,
            n_cols,
            get_abs_path(f'{analysis_path}/comparisons.png'),
        )

    if cli_args.get('zernike_response_gridded_plot'):
        zernike_count = len(zernike_terms)
        nrows = outputs_model.shape[0]
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
        outputs_resp_mat_gr = _split(outputs_resp_mat)

    if cli_args.get('zernike_response_gridded_plot'):
        step_ri('Generating a Zernike response plot')

        def _plot_zernike_response_wr(output_groups, title, name):
            out_path = get_abs_path(f'{analysis_path}/{name}.png')
            plot_zernike_response(zernike_terms, outputs_truth_gr,
                                  output_groups, title, out_path)

        _plot_zernike_response_wr(outputs_model_gr, 'Neural Network',
                                  'zernike_response')
        if response_matrix:
            _plot_zernike_response_wr(outputs_resp_mat_gr, 'Response Matrix',
                                      'zernike_response_resp_mat')
