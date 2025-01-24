"""
Analyze the results produced by running the `control_loop_static_wavefronts`
script. This script will give the total number of static wavefronts that ended
up converging after running many simulated control loops.
"""

from glob import glob
import numpy as np
from utils.constants import (CONTROL_LOOP_RESULTS_P, DATA_F,
                             MEAS_ERROR_HISTORY, TRUE_ERROR_HISTORY)
from utils.hdf_read_and_write import read_hdf
from utils.printing_and_logging import step_ri, title


def analyze_static_wavefront_convergence_parser(subparsers):
    subparser = subparsers.add_parser(
        'analyze_static_wavefront_convergence',
        help='analyze the number of rows that converged',
    )
    subparser.set_defaults(main=analyze_static_wavefront_convergence)
    subparser.add_argument(
        'tag',
        help=('tag of the directory containing the control loop history '
              'located in the `control_loop_results` folder'),
    )
    subparser.add_argument(
        'convergence_threshold',
        type=float,
        help=('all Zernike coefficients must be in the range [-threshold, '
              'threshold] to be considered converged'))


def analyze_static_wavefront_convergence(cli_args):
    title('Analyze static wavefront convergence script')

    # =====================
    # Load in the histories
    # =====================

    # All history datafiles should follow the format [chunk]_[DATA_F]
    step_ri('Loading in datafiles')
    tag = cli_args['tag']
    paths = glob(f'{CONTROL_LOOP_RESULTS_P}/{tag}/*_{DATA_F}')
    true_error_history = []
    meas_error_history = []
    # Combine the histories from all the datafiles together
    for idx, path in enumerate(paths):
        print(f'Path: {path}')
        data = read_hdf(path)
        true_error_history.extend(data[TRUE_ERROR_HISTORY][:])
        meas_error_history.extend(data[MEAS_ERROR_HISTORY][:])
    # Turn back into a np array, has the shape of:
    #   (control loop runs, time steps, Zernike coefficients)
    true_error_history = np.array(true_error_history)
    meas_error_history = np.array(meas_error_history)
    # The number of wavefronts that had control loops ran on them
    nrows = true_error_history.shape[0]

    # =================================================
    # Calculate the number of wavefronts that converged
    # =================================================

    step_ri('Calculating converged control loop rows')
    convergence_threshold = cli_args['convergence_threshold']
    print(f'Threshold range: [-{convergence_threshold}, '
          f'{convergence_threshold}]')

    def _history_converged(history):
        # Take the absolute value of only the last timestep and see if the
        # Zernike coefficients are considered converged
        mask = np.abs(history[:, -1, :]) <= convergence_threshold
        # All coefficients in a control loop must be converged for the control
        # loop itself to be considered converged
        return np.all(mask, axis=1)

    # The true and measured error must both be converged for the control loop
    # to be considered converged
    rows_converged = (_history_converged(true_error_history)
                      & _history_converged(meas_error_history))

    # =================
    # Print the results
    # =================

    step_ri('Convergence results')
    total_converged = np.sum(rows_converged)
    percentage = (total_converged / nrows) * 100
    print(f'Converged: {total_converged}/{nrows} ({percentage:0.2f}%).')
