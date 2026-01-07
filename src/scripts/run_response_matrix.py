"""
This script tests a response matrix's performance against a testing dataset.

Any prior results for a given response matrix will be deleted.

This script can run DH RMs with the `--dh-rm` argument.

For the Zernie plots, it is expected that the `testing_ds` was simulated with
the `sim_data` script using the `--fixed-amount-per-zernike-range` arg and
preprocessed with the `preprocess_data_bare` script.

This code is very similar to `model_test`.

The response matrix runs on denormalized inputs.
"""

import numpy as np
from utils.constants import (ANALYSIS_P, EXTRA_VARS_F, INPUT_MAX_MIN_DIFF,
                             INPUT_MIN_X, MAE, MSE, NORM_RANGE_ONES,
                             PROC_DATA_P, RESULTS_F, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.norm import min_max_denorm, sum_to_one
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa: E501
from utils.plots.plot_zernike_cross_coupling_animation import plot_zernike_cross_coupling_animation  # noqa: E501
from utils.plots.plot_zernike_cross_coupling_mat_animation import plot_zernike_cross_coupling_mat_animation  # noqa: E501
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.plots.plot_zernike_total_cross_coupling import plot_zernike_total_cross_coupling  # noqa: E501
from utils.plots.paper_plots.total_crosstalk import paper_plot_total_crosstalk  # noqa
from utils.plots.paper_plots.model_scatters import paper_plot_model_scatters  # noqa
from utils.plots.paper_plots.zernike_response import paper_plot_zernike_response  # noqa
from utils.printing_and_logging import step_ri, title
from utils.response_matrix import ResponseMatrix
from utils.stats_and_error import mae, mse
from utils.terminate_with_message import terminate_with_message
from utils.torch_hdf_ds_loader import DSLoaderHDF


def run_response_matrix_parser(subparsers):
    subparser = subparsers.add_parser(
        'run_response_matrix',
        help='test a trained model',
    )
    subparser.set_defaults(main=run_response_matrix)
    subparser.add_argument(
        'response_matrix',
        help='tag of the response matrix',
    )
    subparser.add_argument(
        'testing_ds',
        help=('name of the testing dataset, values must have been generated '
              'along a fixed grid; data must have no normalization'),
    )
    subparser.add_argument(
        '--scatter-plot',
        nargs=5,
        metavar=('[n_rows]', '[n_cols]', '[starting_zernike]',
                 '[filter_value]', '[plot_density]'),
        help=('generate a scatter plot; takes the args: number of rows, '
              'number of cols, first Zernike the model outputs, filter value '
              '(0 for no filtering, otherwise filtering with shared axes), '
              'points per pixel to use for the density plot'),
    )
    subparser.add_argument(
        '--zernike-plots',
        action='store_true',
        help='generate the Zernike plots',
    )
    subparser.add_argument(
        '--inputs-need-denorm',
        action='store_true',
        help='the inputs need to be denormalized',
    )
    subparser.add_argument(
        '--inputs-are-diff',
        action='store_true',
        help='the inputs are the delta from the base field',
    )
    subparser.add_argument(
        '--wfs-need-sum-to-one',
        action='store_true',
        help=('the input wavefronts need to be sum to one normalized; will '
              'occur after the differential wavefront happens; a differential '
              'wavefront cannot be passed in'),
    )
    subparser.add_argument(
        '--change-base-field',
        nargs='*',
        help=('raw datafile containing an updated base field to use to form '
              'the differential wavefronts; additional arguments can be '
              'repeated as many times as necessary and should specify '
              '<base field index> <starting row> <ending row>; this requires '
              'that the `--inputs-are-diff` arg is not set'),
    )
    subparser.add_argument(
        '--print-outputs',
        action='store_true',
        help='print out the truth and response matrix outputs',
    )
    subparser.add_argument(
        '--enable-paper-plots',
        type=int,
        help='plot the paper plots too',
    )
    subparser.add_argument(
        '--dh-rm',
        action='store_true',
        help='this is a DH RM (not for LLOWFS Zernikes)',
    )


def run_response_matrix(cli_args):
    title('Run response matrix script')

    step_ri('Loading in the response matrix')
    response_matrix = cli_args.get('response_matrix')
    response_matrix_obj = ResponseMatrix(response_matrix)

    dh_rm = cli_args['dh_rm']
    if dh_rm:
        step_ri('This is a DH response matrix')

    step_ri('Creating the analysis directory')
    testing_ds_tag = cli_args['testing_ds']
    analysis_path = f'{ANALYSIS_P}/{testing_ds_tag}/resp_mat_{response_matrix}'
    analysis_path = get_abs_path(analysis_path)
    delete_dir(analysis_path, quiet=True)
    make_dir(analysis_path)

    step_ri('Loading in the testing dataset')
    testing_dataset = DSLoaderHDF(testing_ds_tag)
    inputs = testing_dataset.get_inputs()
    outputs_truth = testing_dataset.get_outputs()
    base_path = f'{PROC_DATA_P}/{testing_ds_tag}'
    extra_vars = read_hdf(f'{base_path}/{EXTRA_VARS_F}')
    if not dh_rm:
        zernike_terms = extra_vars[ZERNIKE_TERMS]
        print(f'Using zernike terms: {zernike_terms}')

    if cli_args.get('inputs_need_denorm'):
        step_ri('Denormalizing the input values')
        inputs = min_max_denorm(
            inputs,
            extra_vars[INPUT_MAX_MIN_DIFF],
            extra_vars[INPUT_MIN_X],
            extra_vars[NORM_RANGE_ONES][()]
            if NORM_RANGE_ONES in extra_vars else False,
        )

    if not dh_rm:
        step_ri('Validating Zernike terms')
        # Ensure the Zernike terms matchup
        zernike_terms_resp_mat = response_matrix_obj.zernike_terms
        if not np.array_equal(zernike_terms, zernike_terms_resp_mat):
            terminate_with_message('Zernike terms in response matrix do not '
                                   'match terms in dataset')

    step_ri('Flattening inputs')
    # Need to flatten out the pixels before calling the response matrix
    inputs_reshaped = inputs.reshape(inputs.shape[0], -1)

    if dh_rm:
        step_ri('Removing inactive actuators')
        nonzero_acts = (inputs_reshaped != 0).any(axis=0)
        # Idxs where there is >= 1 pixel with a nonzero value
        active_idxs = np.where(nonzero_acts)[0]
        # Filter out the inactive actuators
        inputs_reshaped = inputs_reshaped[:, active_idxs]

    wfs_need_sum_to_one = cli_args.get('wfs_need_sum_to_one')
    if wfs_need_sum_to_one:
        step_ri('Summing the inputs to one')
        inputs_reshaped = sum_to_one(inputs_reshaped, (1))

    step_ri('Running the response matrix')
    change_base_field = cli_args.get('change_base_field')
    if dh_rm:
        print('Passing in the electric field')
        outputs_resp_mat = response_matrix_obj(ef=inputs_reshaped)
    elif cli_args.get('inputs_are_diff'):
        print('Passing in the difference of the intensity field')
        outputs_resp_mat = response_matrix_obj(diff_int_field=inputs_reshaped)
    elif change_base_field:
        print('Using a different base field')
        base_field_tag, *base_field_args = change_base_field
        base_field, _, _, _ = load_raw_sim_data_chunks(base_field_tag)
        base_field = base_field.reshape(base_field.shape[0], -1)
        if wfs_need_sum_to_one:
            print('Making pixel values in the base field(s) sum to 1')
            base_field = sum_to_one(base_field, (1))
        elements = len(base_field_args)
        if elements % 3 != 0:
            terminate_with_message('Incorrect number of mapping arguments')
        for arg_idx in range(elements // 3):
            starting_arg = arg_idx * 3
            base_field_idx = int(base_field_args[starting_arg])
            idx_low = int(base_field_args[starting_arg + 1])
            idx_high = int(base_field_args[starting_arg + 2])
            print(f'Using base field at index {base_field_idx} on '
                  f'rows {idx_low} - {idx_high}')
            inputs_reshaped[idx_low:idx_high] -= base_field[base_field_idx]
        outputs_resp_mat = response_matrix_obj(diff_int_field=inputs_reshaped)
    else:
        print('Passing in the total intensity field')
        outputs_resp_mat = response_matrix_obj(total_int_field=inputs_reshaped)

    # Print the results to the console
    if cli_args.get('print_outputs'):
        step_ri('Printing outputs')

        def _print_outputs(vals):
            print(np.array2string(vals, separator=', ', precision=3))

        step_ri('Results')
        print('Truth (nm):')
        print(outputs_truth * 1e9)
        print('Model (nm):')
        _print_outputs(outputs_resp_mat * 1e9)
        print('Absolute difference (nm):')
        abs_diff = np.abs(outputs_truth - outputs_resp_mat)
        _print_outputs(abs_diff * 1e9)
        print('Percent error:')
        percent_errors = abs_diff / outputs_truth * 100
        _print_outputs(percent_errors)
        print('Average percent error:')
        print(np.sum(percent_errors) / len(zernike_terms))

    step_ri('Computing the MAE and MSE')
    mae_val = mae(outputs_truth, outputs_resp_mat)
    mse_val = mse(outputs_truth, outputs_resp_mat)
    print(f'Model MAE: {mae_val}')
    print(f'Model MSE: {mse_val}')

    step_ri('Writing results to HDF')
    out_file_path = f'{analysis_path}/{RESULTS_F}'
    print(f'File location: {out_file_path}')
    out_data = {
        'outputs_truth': outputs_truth,
        'outputs_response_matrix': outputs_resp_mat,
        MAE: mae_val,
        MSE: mse_val,
    }
    HDFWriteModule(out_file_path).create_and_write_hdf_simple(out_data)

    plot_title = 'Response Matrix'
    plot_identifier = response_matrix

    # Enable paper specific plots
    enable_paper_plots = cli_args.get('enable_paper_plots') is not None
    if enable_paper_plots:
        paper_plot_model_idx = cli_args.get('enable_paper_plots')

    scatter_plot = cli_args.get('scatter_plot')
    if scatter_plot is not None:
        step_ri('Generating scatter plot and density scatter plot')
        (n_rows, n_cols, starting_zernike, plot_density) = [
            int(arg) for arg in [*scatter_plot[:3], scatter_plot[4]]
        ]
        filter_value = float(scatter_plot[3])
        print(f'Using {n_rows} rows and {n_cols} cols.')
        print(f'Starting Zernike: {starting_zernike}.')
        if filter_value:
            print(f'Filtering between: [-{filter_value},{filter_value}].')
        print(f'Point per pixel for density plot: {plot_density}.')
        plot_comparison_scatter_grid(
            outputs_resp_mat,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            starting_zernike,
            filter_value,
            f'{analysis_path}/scatter.png',
        )
        plot_comparison_scatter_grid(
            outputs_resp_mat,
            outputs_truth,
            n_rows,
            n_cols,
            plot_title,
            plot_identifier,
            starting_zernike,
            filter_value,
            f'{analysis_path}/density_scatter.png',
            plot_density=plot_density,
        )
        if enable_paper_plots:
            paper_plot_model_scatters(
                outputs_resp_mat,
                outputs_truth,
                starting_zernike,
                f'{analysis_path}/paper_scatter.png',
                paper_plot_model_idx,
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
        outputs_resp_mat_gr = _split(outputs_resp_mat)

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
            outputs_resp_mat_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_response.png',
        )
        if enable_paper_plots:
            paper_plot_zernike_response(
                zernike_terms,
                perturbation_grid,
                outputs_resp_mat_gr,
                f'{analysis_path}/paper_zernike_response.png',
                paper_plot_model_idx,
            )

        step_ri('Generating a Zernike total cross coupling plot')
        plot_zernike_total_cross_coupling(
            zernike_terms,
            perturbation_grid,
            outputs_resp_mat_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/total_cross_coupling.png',
        )
        if enable_paper_plots:
            paper_plot_total_crosstalk(
                zernike_terms,
                perturbation_grid,
                outputs_resp_mat_gr,
                f'{analysis_path}/paper_total_cross_coupling.png',
                paper_plot_model_idx,
            )

        step_ri('Generating a Zernike cross coupling animation')
        plot_zernike_cross_coupling_animation(
            zernike_terms,
            perturbation_grid,
            outputs_resp_mat_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_cross_coupling.gif',
        )

        step_ri('Generating a Zernike cross coupling matrix animation')
        plot_zernike_cross_coupling_mat_animation(
            zernike_terms,
            perturbation_grid,
            outputs_resp_mat_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/zernike_cross_coupling_mat.gif',
        )
