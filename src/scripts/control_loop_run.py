"""
Run a control loop to flatten out a wavefront.

A diagram of how the control loop works can be found at
`diagrams/wfs_control_loop.png`.

The step file should be generated with the `gen_zernike_time_steps` script.
"""

import numpy as np
from utils.constants import CONTROL_LOOP_RESULTS_P, CONTROL_LOOP_STEPS_P
from utils.iterate_simulated_control_loop import iterate_simulated_control_loop
from utils.path import make_dir
from utils.plots.plot_control_loop_zernikes import plot_control_loop_zernikes
from utils.plots.plot_control_loop_zernikes_subplots import plot_control_loop_zernikes_subplots  # noqa: E501
from utils.printing_and_logging import step_ri, title


def control_loop_run_parser(subparsers):
    subparser = subparsers.add_parser(
        'control_loop_run',
        help='run a control loop to flatten out a wavefront',
    )
    subparser.set_defaults(main=control_loop_run)
    subparser.add_argument(
        'step_file',
        help='name of the CSV file containing the control loop steps',
    )
    subparser.add_argument(
        'K_p',
        type=float,
        help='proportional gain (range -1 to 0)',
    )
    subparser.add_argument(
        'K_i',
        type=float,
        help='integral gain (range -1 to 0)',
    )
    subparser.add_argument(
        'K_d',
        type=float,
        help='derivative gain (range -1 to 0)',
    )
    subparser.add_argument(
        'train_name',
        help='name of the optical train',
    )
    subparser.add_argument(
        'ref_wl',
        type=float,
        help='reference wavelength in meters for the image simulations',
    )
    subparser.add_argument(
        '--grid-points',
        type=int,
        default=1024,
        help='number of grid points for the image simulations',
    )
    model_group = subparser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--neural-network',
        nargs=2,
        metavar=('[tag]', '[epoch]'),
        help='use a neural network',
    )
    model_group.add_argument(
        '--response-matrix',
        metavar='[tag]',
        help='use a response matrix',
    )


def control_loop_run(cli_args):
    title('Control loop run script')

    # ====================
    # Load in the CLI args
    # ====================

    step_ri('Loading CLI args')
    step_file = cli_args['step_file']
    K_p = cli_args['K_p']
    K_i = cli_args['K_i']
    K_d = cli_args['K_d']
    train_name = cli_args['train_name']
    ref_wl = cli_args['ref_wl']
    grid_points = cli_args['grid_points']
    neural_network = cli_args['neural_network']
    response_matrix = cli_args['response_matrix']

    # ==============================
    # Load in the control loop steps
    # ==============================

    step_ri('Loading in the control loop steps')
    step_file_path = f'{CONTROL_LOOP_STEPS_P}/{step_file}.csv'
    step_data = np.loadtxt(step_file_path, delimiter=',')
    total_steps = step_data.shape[0]
    print(f'Total steps: {total_steps}')
    with open(step_file_path) as file:
        # The first three columns must be the row index, cumulative time,
        # and delta time. The first character (`#`, comment) and last character
        # (`\n`, new line) must be chopped off.
        header_line = file.readline()[1:-1].split(',')
    # Grab only the Zernike terms
    zernike_terms = header_line[3:]
    # Parse off the space and `Z` character
    zernike_terms = [int(term[2:]) for term in zernike_terms]
    print(f'Zernike terms: {zernike_terms}')

    # ====================
    # Run the control loop
    # ====================

    (_, cumulative_time, true_error_history,
     meas_error_history) = iterate_simulated_control_loop(
         step_data,
         zernike_terms,
         K_p,
         K_i,
         K_d,
         train_name,
         ref_wl,
         grid_points=grid_points,
         enable_logs=True,
         use_nn=neural_network,
         use_rm=response_matrix,
     )

    # ======================
    # Create the output dirs
    # ======================

    step_ri('Creating the output directories')

    def _make_dir(base, folder):
        path = f'{base}/{folder}'
        make_dir(path)
        return path

    # Set the model string
    if neural_network:
        tag, epoch = neural_network
        model_str = f'NN_{tag}_{epoch}'
    elif response_matrix:
        model_str = f'RM_{response_matrix}'

    # All of the output directories
    out_dir = _make_dir(CONTROL_LOOP_RESULTS_P,
                        f'{step_file}_{model_str}_{K_p}_{K_i}_{K_d}')
    output_path_hist_data = _make_dir(out_dir, 'history_data')
    output_path_ts_same = _make_dir(out_dir, 'time_series_plots/same_plot')
    output_path_ts_sub = _make_dir(out_dir, 'time_series_plots/subplots')
    output_path_psd_same = _make_dir(out_dir, 'psd_plots/same_plot')
    output_path_psd_sub = _make_dir(out_dir, 'psd_plots/subplots')

    # =====================
    # Write out the history
    # =====================

    step_ri('Writing out the history')

    # Write out the history CSV
    def _write_hist(history_data, hist_type):
        print(f'Saving {hist_type} error history.')
        history_path = f'{output_path_hist_data}/{hist_type}_error.csv'
        np.savetxt(history_path, history_data, delimiter=',', fmt='%.12f')

    _write_hist(true_error_history, 'true')
    _write_hist(meas_error_history, 'meas')

    # ==================
    # Generate the plots
    # ==================

    step_ri('Generating the plots')

    # Title that will be displayed on the plots
    base_title = (f'Control Loop, Step File={step_file}, '
                  f'Timesteps={total_steps}\n')
    base_title_ext = base_title + (
        f'Model={model_str}, K_PID={(K_p, K_i, K_d)}, ')
    # Determine the number of rows and columns for the subplot plots
    n_rows = n_cols = int(np.ceil(np.sqrt(len(zernike_terms))))

    def _create_plots(history_data, hist_type, additional_info):
        print(f'Saving {hist_type} error plots (x4).')
        plot_title = base_title_ext + f'{hist_type} error ({additional_info})'
        plot_file = f'{hist_type}_error.png'

        def _zernike_same_plot(output_dir, plot_psd):
            plot_path = f'{output_dir}/{plot_file}'
            plot_control_loop_zernikes(zernike_terms, history_data, plot_title,
                                       cumulative_time, plot_path, plot_psd)

        def _zernike_subplots(output_dir, plot_psd):
            plot_path = f'{output_dir}/{plot_file}'
            plot_control_loop_zernikes_subplots(zernike_terms, history_data,
                                                plot_title, cumulative_time,
                                                n_rows, n_cols, plot_path,
                                                plot_psd)

        _zernike_same_plot(output_path_ts_same, False)  # Time series
        _zernike_same_plot(output_path_psd_same, True)  # PSD
        _zernike_subplots(output_path_ts_sub, False)  # Time series
        _zernike_subplots(output_path_psd_sub, True)  # PSD

    _create_plots(true_error_history, 'true', 'input aberrations')
    _create_plots(meas_error_history, 'meas', 'model outputs')

    print('Saving input signal plots (x2).')
    input_signal_data = step_data[:, 3:]
    plot_title = f'{base_title}Input Signal'
    plot_path = f'{output_path_ts_same}/input_signal.png'
    plot_control_loop_zernikes(zernike_terms, input_signal_data, plot_title,
                               cumulative_time, plot_path)
    plot_path = f'{output_path_ts_sub}/input_signal.png'
    plot_control_loop_zernikes_subplots(zernike_terms, input_signal_data,
                                        plot_title, cumulative_time, n_rows,
                                        n_cols, plot_path)

    print('Saving input signal vs meas error plots (x2).')

    def _zernike_subplots(output_dir, plot_psd):
        plot_title = base_title_ext + 'input signal vs meas error'
        plot_path = f'{output_dir}/input_vs_meas.png'
        plot_control_loop_zernikes_subplots(zernike_terms, input_signal_data,
                                            plot_title, cumulative_time,
                                            n_rows, n_cols, plot_path,
                                            plot_psd, meas_error_history,
                                            ['Input Signal', 'Meas Error'])

    _zernike_subplots(output_path_ts_sub, False)
    _zernike_subplots(output_path_psd_sub, True)

    print('Saving true error vs meas error plots (x2).')

    def _zernike_subplots(output_dir, plot_psd):
        plot_title = base_title_ext + 'true error vs meas error'
        plot_path = f'{output_dir}/true_vs_meas.png'
        plot_control_loop_zernikes_subplots(zernike_terms, true_error_history,
                                            plot_title, cumulative_time,
                                            n_rows, n_cols, plot_path,
                                            plot_psd, meas_error_history,
                                            ['True Error', 'Meas Error'])

    _zernike_subplots(output_path_ts_sub, False)
    _zernike_subplots(output_path_psd_sub, True)
