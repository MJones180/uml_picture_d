"""
Generate a CSV file with time steps containing different Zernike coefficients.
This can then be plugged into the `control_loop_run` script.

Note: the CSV files generated with this script have a max precision of
12 decimal places.

This file works well for simulating a single control loop and running it with
the `control_loop_run` script. However, if many, static aberration control loop
steps need to be run, then it may be easier to create an aberrations file with
the `sim_data` script and then run those using the
`control_loop_static_wavefronts` script.

While this script currently only has options to generate static aberrations,
it does have the potential to create ones as complex as needed.
"""

import numpy as np
from utils.constants import CONTROL_LOOP_STEPS_P
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def gen_zernike_time_steps_parser(subparsers):
    subparser = subparsers.add_parser(
        'gen_zernike_time_steps',
        help='generate a CSV with timesteps and Zernike coefficients',
    )
    subparser.set_defaults(main=gen_zernike_time_steps)
    subparser.add_argument(
        'name',
        help='name of the output CSV file (excluding the file extension)',
    )
    subparser.add_argument(
        'delta_time',
        type=float,
        help='delta time between timesteps (in seconds)',
    )
    subparser.add_argument(
        'timesteps',
        type=int,
        help='total number of timesteps',
    )
    subparser.add_argument(
        'zernike_low',
        type=int,
        help='lowest Zernike term',
    )
    subparser.add_argument(
        'zernike_high',
        type=int,
        help='highest Zernike term',
    )

    gen_method_group = subparser.add_mutually_exclusive_group()
    gen_method_group.add_argument(
        '--single-zernike-constant-value',
        nargs=2,
        metavar=('[zernike term]', '[rms error in meters]'),
        help='add a constant amplitude on a single Zernike term',
    )
    gen_method_group.add_argument(
        '--all-zernikes-constant-value',
        metavar='[rms error in meters]',
        help='add a constant amplitude on all Zernike terms',
    )


def gen_zernike_time_steps(cli_args):
    title('Generate zernike time steps script')

    step_ri('Loading CLI args')
    name = cli_args['name']
    delta_time = cli_args['delta_time']
    timesteps = cli_args['timesteps']
    zernike_low = cli_args['zernike_low']
    zernike_high = cli_args['zernike_high']
    single_zernike_constant_value = cli_args['single_zernike_constant_value']
    all_zernikes_constant_value = cli_args['all_zernikes_constant_value']

    step_ri('Config information')
    print(f'Timesteps: {timesteps}')
    print(f'Delta time: {delta_time} (s)')
    print(f'Total time: {timesteps * delta_time} (s)')
    zernike_terms = np.arange(zernike_low, zernike_high + 1)
    print(f'Zernike terms: {zernike_terms}')

    step_ri('Column information')
    # This will be the first row in the CSV file with the name of each column
    column_names = ('ROW_IDX, CUMULATIVE_TIME, DELTA_TIME, ' +
                    ', '.join([f'Z{term}' for term in zernike_terms]))
    print(column_names)
    # The number of columns ignoring the Zernike coeff columns
    base_column_count = 3
    # The number of columns in the header row
    total_column_count = base_column_count + len(zernike_terms)
    print(f'Total columns: {total_column_count}')

    step_ri('Constructing the output data array')
    output_data = np.zeros((timesteps, total_column_count))
    timestep_values = np.arange(timesteps)
    delta_time_values = np.full(timesteps, delta_time)
    # Have the time start at 0
    cumulative_time_values = np.concatenate(
        ([0], np.cumsum(delta_time_values[1:])))
    # Index column
    output_data[:, 0] = timestep_values
    # Cumulative time column
    output_data[:, 1] = cumulative_time_values
    # Delta time column
    output_data[:, 2] = delta_time_values

    if single_zernike_constant_value:
        step_ri('Using a single Zernike term with a constant RMS error')
        zernike_term, rms_error = single_zernike_constant_value
        print(f'Zernike term: {zernike_term}, RMS error: {rms_error} (m)')
        idx = base_column_count + int(zernike_term) - zernike_low
        term_values = np.full(timesteps, float(rms_error))
        output_data[:, idx] = term_values
    elif all_zernikes_constant_value:
        step_ri('Using all Zernike terms with a constant RMS error')
        rms_error = all_zernikes_constant_value
        print(f'RMS error: {rms_error} (m)')
        term_values = np.full(timesteps, float(rms_error))
        for idx in range(base_column_count, total_column_count):
            output_data[:, idx] = term_values
    else:
        terminate_with_message('No method for generating the data chosen')

    output_file = f'{CONTROL_LOOP_STEPS_P}/{name}.csv'
    step_ri('Saving CSV file')
    print(f'File path: {output_file}')
    np.savetxt(
        output_file,
        output_data,
        delimiter=',',
        header=column_names,
        fmt='%.12f',
    )
