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
from utils.plots.plot_control_loop_zernikes import plot_control_loop_zernikes
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
        model_str = f'nn_{tag}_{epoch}'

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
        model_str = f'rm_{response_matrix}'

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
    # Keep track of the model ouputs over time so they can be plotted at the end
    model_output_history = []

    step_ri('Running the control loop')
    for step in step_data:
        row_idx, cumulative_time, delta_time, *zernike_coeffs = step
        print(f'Step: {int(row_idx + 1)}/{total_steps}')
        # Aberrations should be the sum of the step signal and the correction
        aberrations = zernike_coeffs + corrections
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
        model_output_history.append(model_output)
        # The new set of corrections are the corrections from the last time step
        # in addition to the PID gains. Therefore, we can just add on each gain
        # term as we go.
        corrections += K_p * model_output  # K_p term
        # Yes, the running sum could be calculated each time from the
        # `model_output_history`, but this seems quite a bit more efficient
        running_zernike_sum += model_output
        corrections += K_i * running_zernike_sum  # K_i term
        # Do not calculate the time derivative if this is the first step
        if len(model_output_history) > 1:
            dzdt = (model_output - model_output_history[-2]) / delta_time
            corrections += K_d * dzdt  # K_d term
    # Now, we can plot our model outputs over time
    model_output_history = np.array(model_output_history)
    print('Finished running the control loop')

    step_ri('Generating plots')
    plot_path = f'{RANDOM_P}/{step_file}_{model_str}_{K_p}_{K_i}_{K_d}.png'
    print(f'Saving plot to {plot_path}')
    plot_control_loop_zernikes(
        zernike_terms,
        model_output_history,
        model_str,
        (K_p, K_i, K_d),
        cumulative_time,
        plot_path,
    )
