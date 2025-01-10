"""
Run a control loop to flatten out a wavefront.

A diagram of how the control loop works can be found at
`diagrams/wfs_control_loop.png`.

The step file should be generated with the `gen_zernike_time_steps` script.
"""

import numpy as np
from utils.constants import CONTROL_LOOP_STEPS_P, RANDOM_P
from utils.load_optical_train import load_optical_train
from utils.model import Model
from utils.path import make_dir
from utils.plots.plot_control_loop_zernikes import plot_control_loop_zernikes
from utils.plots.plot_control_loop_zernikes_psd import plot_control_loop_zernikes_psd  # noqa: E501
from utils.printing_and_logging import step_ri, title
from utils.response_matrix import ResponseMatrix
from utils.sim_prop_wf import sim_prop_wf
from utils.terminate_with_message import terminate_with_message


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
    subparser.add_argument(
        '--plot-with-zernike-numbers',
        action='store_true',
        help=('add extra scatter points containing the Zernike term numbers '
              'on top of each point for the plots'),
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

    step_ri('Loading CLI args')
    step_file = cli_args['step_file']
    K_p = cli_args['K_p']
    K_i = cli_args['K_i']
    K_d = cli_args['K_d']
    train_name = cli_args['train_name']
    ref_wl = cli_args['ref_wl']
    grid_points = cli_args['grid_points']
    plot_with_zernike_numbers = cli_args['plot_with_zernike_numbers']
    neural_network = cli_args['neural_network']
    response_matrix = cli_args['response_matrix']

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
    zerinke_count = len(zernike_terms)
    print(f'Zernike terms: {zernike_terms}')

    def _verify_gain(name, var):
        print(f'{name}: {var}')
        if not (-1 <= var <= 0):
            terminate_with_message(f'{name} must be between -1 and 0')

    step_ri('Verifying gain values')
    _verify_gain('K_p', K_p)
    _verify_gain('K_i', K_i)
    _verify_gain('K_d', K_d)

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, camera_pixels,
     camera_sampling) = load_optical_train(train_name)

    # Load in the model
    if neural_network:
        step_ri('Loading in the neural network')
        tag, epoch = neural_network
        model = Model(tag, epoch, force_cpu=cli_args.get('force_cpu'))
        model_str = f'NN_{tag}_{epoch}'

        def call_model(inputs):
            # Subtract off the base field so that we have the delta field
            # intensity and then normalize the data
            preprocessed_inputs = model.norm_data(inputs - model.base_field)
            # We need to add an extra dimension to represents the batch size
            preprocessed_inputs = preprocessed_inputs[None, :, :, :]
            # Since we are only passing in one row, we only need to grab the
            # first row from the batch size dimension
            output = model(preprocessed_inputs)[0]
            # Denormalize the data
            return model.denorm_data(output)
    elif response_matrix:
        step_ri('Loading in the response matrix')
        response_matrix_obj = ResponseMatrix(response_matrix)
        model_str = f'RM_{response_matrix}'

        def call_model(inputs):
            # A 1D array should be passed in
            return response_matrix_obj(total_int_field=inputs.reshape(-1))
    else:
        terminate_with_message('Neural network or response matrix required')

    step_ri('Preparing to run control loop')
    print('Initializing corrections to all zero')
    corrections = np.zeros(zerinke_count)
    # We need the cumulative sum for each Zernike term to compute integral grain
    running_zernike_sum = np.zeros(zerinke_count)
    # Keep track of the history for plotting and writing out
    true_error_history = []
    meas_error_history = []

    step_ri('Running the control loop')
    for step in step_data:
        row_idx, cumulative_time, delta_time, *zernike_coeffs = step
        print(f'Step: {int(row_idx + 1)}/{total_steps}')
        # Aberrations should be the sum of the signal and the correction
        aberrations = zernike_coeffs + corrections
        true_error_history.append(aberrations)
        # Simulate the camera image that represents these Zernike coeffs
        camera_image, _, _ = sim_prop_wf(
            init_beam_d,
            ref_wl,
            beam_ratio,
            optical_train,
            camera_pixels,
            camera_sampling,
            zernike_terms,
            aberrations,
            grid_points,
        )
        # This will output the model's coefficients (nn or response matrix)
        model_output = call_model(camera_image)
        meas_error_history.append(model_output)
        # The new set of corrections are in addition to the corrections from the
        # last time step, so we can just add each term as we go.
        corrections += K_p * model_output  # proportional term
        # Update the running sum to compute the integral term.
        running_zernike_sum += model_output * delta_time
        corrections += K_i * running_zernike_sum  # integral term
        # Do not calculate the time derivative if this is the first step
        if len(meas_error_history) > 1:
            dzdt = (model_output - meas_error_history[-2]) / delta_time
            corrections += K_d * dzdt  # derivative term
    # Now, we can plot our history over time
    true_error_history = np.array(true_error_history)
    meas_error_history = np.array(meas_error_history)
    print('Finished running the control loop')

    step_ri('Creating the output directory')
    output_path = f'{RANDOM_P}/{step_file}_{model_str}_{K_p}_{K_i}_{K_d}'
    print(f'Output path: {output_path}')
    make_dir(output_path)

    step_ri('Saving history and generating plots')

    def _write_hist_and_plots(history_data, hist_type, additional_info):
        base_path = f'{output_path}/{hist_type}_error_history'
        # Write out the history CSV
        history_path = f'{base_path}.csv'
        print(f'Saving {hist_type} error history.')
        np.savetxt(history_path, history_data, delimiter=',', fmt='%.12f')
        # Generate the plot
        plot_path = f'{base_path}.png'
        plot_path_psd = f'{base_path}_psd.png'
        title_info = (f'Model={model_str}, K_PID={(K_p, K_i, K_d)}, '
                      f'{hist_type} error ({additional_info})')
        print(f'Saving {hist_type} error plots.')
        plot_control_loop_zernikes(
            zernike_terms,
            history_data,
            step_file,
            title_info,
            cumulative_time,
            plot_with_zernike_numbers,
            plot_path,
        )
        plot_control_loop_zernikes_psd(
            zernike_terms,
            history_data,
            step_file,
            title_info,
            delta_time,
            plot_path_psd,
        )

    _write_hist_and_plots(true_error_history, 'true', 'input aberrations')
    _write_hist_and_plots(meas_error_history, 'meas', 'model outputs')

    plot_path = f'{output_path}/input_signal.png'
    print('Saving input signal plot.')
    plot_control_loop_zernikes(
        zernike_terms,
        step_data[:, 3:],
        step_file,
        'Input Signal',
        cumulative_time,
        plot_with_zernike_numbers,
        plot_path,
    )
