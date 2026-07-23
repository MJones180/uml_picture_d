"""
This script tests a model's performance against a testing dataset.

Any prior results for a given epoch will be deleted.

When generating any of the Zernike-related plots, it is expected that
the `testing_ds` was simulated with the `sim_data` script using the
`--fixed-amount-per-zernike-range` arg and preprocessed with the
`preprocess_data_bare` script.
"""

import numpy as np
from utils.constants import (ANALYSIS_P, EXTRA_VARS_F, MAE, MSE,
                             NORM_STABILITY_VALUE, PROC_DATA_P, RESULTS_F,
                             ZERNIKE_TERMS)
from utils.group_data_from_list import group_data_from_list
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.load_raw_sim_data import (load_raw_sim_data_chunks,
                                     raw_sim_data_chunk_paths)
from utils.model import Model
from utils.path import delete_dir, get_abs_path, make_dir
from utils.plots.plot_coeff_comparison import plot_coeff_comparison
from utils.plots.plot_comparison_scatter_grid import plot_comparison_scatter_grid  # noqa: E501
from utils.plots.plot_gamma_bars import plot_gamma_bars
from utils.plots.plot_model_loss import plot_model_loss
from utils.plots.plot_zernike_cross_coupling_animation import plot_zernike_cross_coupling_animation  # noqa: E501
from utils.plots.plot_zernike_cross_coupling_mat_animation import plot_zernike_cross_coupling_mat_animation  # noqa: E501
from utils.plots.plot_zernike_crosstalk_grid import plot_zernike_crosstalk_grid
from utils.plots.plot_zernike_response import plot_zernike_response
from utils.plots.plot_zernike_total_cross_coupling import plot_zernike_total_cross_coupling  # noqa: E501
from utils.plots.paper_plots.total_crosstalk import paper_plot_total_crosstalk  # noqa
from utils.plots.paper_plots.model_scatters import paper_plot_model_scatters  # noqa
from utils.plots.paper_plots.zernike_response import paper_plot_zernike_response  # noqa
from utils.printing_and_logging import dec_print_indent, step, step_ri, title
from utils.shared_argparser_args import shared_argparser_args
from utils.stats_and_error import (mae, mse,
                                   symmetric_mean_absolute_percentage_error)
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
        '--change-base-field',
        nargs='*',
        help=('raw datafile containing an updated base field to use to form '
              'the differential wavefronts (should not have any sum one '
              'normalization); additional arguments can be repeated as many '
              'times as necessary and should specify <base field index> '
              '<starting row> <ending row>; this requires that both the '
              '`--inputs-need-norm` and `--inputs-need-diff` args are set'),
    )
    subparser.add_argument(
        '--change-base-field-corresponding',
        help=('raw datafile containing the base fields to use; '
              'every row of data has its own corresponding base field'),
    )
    subparser.add_argument(
        '--outputs-no-denorm',
        action='store_true',
        help='the outputs do not need to be denormalized',
    )
    subparser.add_argument(
        '--norm-stability-value',
        type=float,
        default=NORM_STABILITY_VALUE,
        help='the stability constant to use for normalization',
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
        '--disable-zernike-gif-plots',
        action='store_true',
        help='disable generating the Zernike GIF plots',
    )
    subparser.add_argument(
        '--zernike-crosstalk-grid',
        action='store_true',
        help='generate a Zernike crosstalk grid',
    )
    subparser.add_argument(
        '--print-outputs',
        action='store_true',
        help='print out the truth and model outputs',
    )
    subparser.add_argument(
        '--print-actuator-height-error',
        nargs='*',
        help=('convert the coefficients to actuator heights and compute '
              'the MAE and MSE; three arguments should be passed for each '
              'group: tag of the raw dataset containing the basis modes, '
              'the table name, and the number of coefficients in the group; '
              'all the output coefficients should be covered by the modes'),
    )
    subparser.add_argument(
        '--print-actuator-height-error-transpose-modes',
        action='store_true',
        help=('should be used with the `--print-actuator-height-error` arg; '
              'transpose the modes'),
    )
    subparser.add_argument(
        '--max-rows-per-model-call',
        type=int,
        help='limit the number of rows per model call',
    )
    subparser.add_argument(
        '--enable-paper-plots',
        type=int,
        help=('plot the paper plots too; the index passed is the model '
              'being used as determined by the plotting script'),
    )
    subparser.add_argument(
        '--plot-coeff-mae-smape',
        type=int,
        nargs='+',
        help=('plot the MAE and sMAPE for the output coefficients; the passed '
              'arguments should be the upper index for each coeff group'),
    )
    subparser.add_argument(
        '--plot-layerscale-gamma',
        nargs='+',
        help=('plot the mean LayerScale gamma from every layer; the passed '
              'argument should specify the variable name (probably `gamma`); '
              'additional args can be passed if there are multiple heads, '
              'each arg represents the depth of a head'),
    )
    subparser.add_argument(
        '--plot-loss-curves',
        action='store_true',
        help='plot the training and validation loss curves',
    )
    shared_argparser_args(subparser, ['force_cpu'])


def model_test(cli_args):
    title('Model test script')

    tag = cli_args['tag']
    epoch = cli_args['epoch']

    model = Model(
        tag,
        epoch,
        force_cpu=cli_args.get('force_cpu'),
        max_rows_per_model_call=cli_args.get('max_rows_per_model_call'),
    )
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
    zernike_terms = read_hdf(extra_vars_path).get(ZERNIKE_TERMS)
    if zernike_terms is not None:
        print(f'Using zernike terms: {zernike_terms}')

    norm_stability_value = cli_args.get('norm_stability_value')
    if norm_stability_value is None:
        norm_stability_value = NORM_STABILITY_VALUE
    step_ri(f'Norm stability value: {norm_stability_value}')

    if cli_args.get('inputs_need_norm'):
        step_ri('Preprocessing the inputs')
        inputs_need_diff = cli_args.get('inputs_need_diff')

        def _norm_inputs(input_chunk, updated_base_field=None):
            if updated_base_field is not None:
                model.change_base_field(updated_base_field)
            return model.preprocess_data(
                input_chunk,
                sub_basefield=inputs_need_diff,
                sum_dims=(1, 2, 3),
                norm_stability_constant=norm_stability_value,
            )

        if cli_args.get('change_base_field'):
            print('Changing the base field')
            bf_tag, *base_field_args = cli_args.get('change_base_field')
            base_fields, _, _, _ = load_raw_sim_data_chunks(bf_tag)
            if model.inputs_sum_to_one:
                print('Making pixel values in the base field(s) sum to 1')
                base_fields = model.sum_inputs_to_one(base_fields, (1, 2))
            for group_args in group_data_from_list(base_field_args, 3):
                base_field_idx = int(group_args[0])
                idx_low = int(group_args[1])
                idx_high = int(group_args[2])
                print(f'Using base field at index {base_field_idx} on '
                      f'rows {idx_low} - {idx_high}')
                inputs[idx_low:idx_high] = _norm_inputs(
                    inputs[idx_low:idx_high],
                    base_fields[base_field_idx],
                )
        elif cli_args.get('change_base_field_corresponding'):
            print('Changing the base field')
            base_fields, _, _, _ = load_raw_sim_data_chunks(
                cli_args.get('change_base_field_corresponding'))
            if model.inputs_sum_to_one:
                print('Making pixel values in the base field(s) sum to 1')
                base_fields = model.sum_inputs_to_one(base_fields, (1, 2))
            print('Each row has its own base field')
            for idx in range(inputs.shape[0]):
                inputs[idx] = _norm_inputs(inputs[idx][:, None],
                                           base_fields[idx])
        else:
            inputs = _norm_inputs(inputs)

    step_ri('Calling the model and obtaining its outputs')
    outputs_model = model(inputs)

    if not cli_args.get('outputs_no_denorm'):
        step_ri('Denormalizing the outputs')
        outputs_model = model.denorm_data(
            outputs_model,
            norm_stability_constant=norm_stability_value,
        )
    # Testing output data should already be denormalized
    outputs_truth = testing_dataset.get_outputs()

    # Print the results to the console
    if cli_args.get('print_outputs'):
        step_ri('Printing outputs')

        def _print_outputs(vals):
            print(np.array2string(vals, separator=', ', precision=3))

        step_ri('Results')
        print('Truth (nm):')
        print(outputs_truth * 1e9)
        print('Model (nm):')
        _print_outputs(outputs_model * 1e9)
        print('Absolute difference (nm):')
        abs_diff = np.abs(outputs_truth - outputs_model)
        _print_outputs(abs_diff * 1e9)
        print('Percent error:')
        percent_errors = abs_diff / outputs_truth * 100
        _print_outputs(percent_errors)
        if zernike_terms is not None:
            print('Average percent error:')
            print(np.sum(percent_errors) / len(zernike_terms))

    step_ri('Computing the MAE and MSE')
    mae_val = mae(outputs_truth, outputs_model)
    mse_val = mse(outputs_truth, outputs_model)
    print(f'Model MAE: {mae_val}')
    print(f'Model MSE: {mse_val}')

    print_actuator_height_error = cli_args.get('print_actuator_height_error')
    if print_actuator_height_error is not None:
        step_ri('Determining actuator height error')
        heights_truth_all = []
        heights_model_all = []
        lower_idx = 0
        trans_modes = cli_args['print_actuator_height_error_transpose_modes']
        for group_args in group_data_from_list(print_actuator_height_error, 3):
            modes_tag = group_args[0]
            table_name = group_args[1]
            number_coeffs = int(group_args[2])
            upper_idx = lower_idx + number_coeffs
            step(f'Basis modes for {modes_tag}')
            print(f'Table name: {table_name}')
            print(f'Number of coeffs: {number_coeffs}')
            modes_path = raw_sim_data_chunk_paths(modes_tag)[0]
            modes_data = read_hdf(modes_path)[table_name][:]
            print(f'Modes shape: {modes_data.shape}')
            if trans_modes:
                modes_data = np.transpose(modes_data)
            # Grab just the used modes
            modes_data = modes_data[:number_coeffs]
            # Compute the actuator heights from these modes
            heights_truth = outputs_truth[:, lower_idx:upper_idx] @ modes_data
            heights_model = outputs_model[:, lower_idx:upper_idx] @ modes_data
            heights_truth_all.append(heights_truth)
            heights_model_all.append(heights_model)
            print(f'Actuator heights shape: {np.array(heights_truth).shape}')
            print(f'Actuator height MAE: {mae(heights_truth, heights_model)}')
            print(f'Actuator height MSE: {mse(heights_truth, heights_model)}')
            lower_idx = upper_idx
            dec_print_indent()
        rows = len(outputs_truth)
        heights_truth = np.swapaxes(heights_truth_all, 0, 1).reshape(rows, -1)
        heights_model = np.swapaxes(heights_model_all, 0, 1).reshape(rows, -1)
        print(f'Actuator heights shape: {heights_truth.shape}')
        print(f'Overall MAE: {mae(heights_truth, heights_model)}')
        print(f'Overall MSE: {mse(heights_truth, heights_model)}')

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
            outputs_model,
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
            outputs_model,
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
                outputs_model,
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
        if enable_paper_plots:
            paper_plot_zernike_response(
                zernike_terms,
                perturbation_grid,
                outputs_model_gr,
                f'{analysis_path}/paper_zernike_response.png',
                paper_plot_model_idx,
            )

        step_ri('Generating a Zernike total cross coupling plot')
        plot_zernike_total_cross_coupling(
            zernike_terms,
            perturbation_grid,
            outputs_model_gr,
            plot_title,
            plot_identifier,
            f'{analysis_path}/total_cross_coupling.png',
        )
        if enable_paper_plots:
            paper_plot_total_crosstalk(
                zernike_terms,
                perturbation_grid,
                outputs_model_gr,
                f'{analysis_path}/paper_total_cross_coupling.png',
                paper_plot_model_idx,
            )

        if not cli_args.get('disable_zernike_gif_plots'):
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

        if cli_args.get('zernike_crosstalk_grid'):
            step_ri('Generating a Zernike crosstalk grid')
            plot_zernike_crosstalk_grid(
                zernike_terms,
                perturbation_grid,
                outputs_model_gr,
                f'{analysis_path}/crosstalk_grid.png',
            )

    plot_coeff_mae_smape = cli_args.get('plot_coeff_mae_smape')
    if plot_coeff_mae_smape is not None:
        step_ri('Plotting coefficient MAE and sMAPE')
        print(f'Coeff group idxs: {plot_coeff_mae_smape}')
        plot_coeff_comparison(
            plot_coeff_mae_smape,
            mae(outputs_truth, outputs_model, 0),
            symmetric_mean_absolute_percentage_error(outputs_truth,
                                                     outputs_model, 0),
            'MAE',
            'sMAPE',
            f'{analysis_path}/mae_and_smape.png',
        )

    plot_layerscale_gamma = cli_args.get('plot_layerscale_gamma')
    if plot_layerscale_gamma:
        step_ri('Plotting LayerScale gamma')
        gamma_name, *head_depths = plot_layerscale_gamma
        print(f'Gamma variable name: {gamma_name}')
        if len(head_depths) > 0:
            head_depths = [int(val) for val in head_depths]
            print(f'Head split points: {head_depths}')
        gamma_magnitudes = [
            param.data.abs().mean().item()
            for name, param in model.model.named_parameters()
            if gamma_name in name
        ]
        if len(gamma_magnitudes) == 0:
            print(f'No layers with {gamma_name} found')
        else:
            print(f'Gamma Mean: {np.mean(gamma_magnitudes)}')
            print(f'Gamma Min: {np.min(gamma_magnitudes)}')
            print(f'Gamma Max: {np.max(gamma_magnitudes)}')
            plot_gamma_bars(
                gamma_magnitudes,
                f'{analysis_path}/layerscale_gamma.png',
                head_depths,
            )

    if cli_args.get('plot_loss_curves'):
        step_ri('Plotting loss curves')
        plot_model_loss(
            # Columns: epoch, train_loss, val_loss
            model.training_loss,
            model.training_args['loss'],
            f'{analysis_path}/loss.png',
        )
