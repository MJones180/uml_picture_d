"""
Run many aberration rows from a dataset's aberration file and see if they
converge within a certain number of timesteps. The aberration file must be
created with the `--save-aberrations-csv` arg in the `sim_data` script.

This script only works on static aberrations - they cannot change at each
timestep. If this feature is needed, then use the `control_loop_run` script
in conjunction with the `gen_zernike_time_steps` script.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.iterate_simulated_control_loop import iterate_simulated_control_loop
from utils.load_raw_sim_data import load_raw_sim_data_aberrations_file
from utils.printing_and_logging import step_ri, title


def control_loop_ds_capture_parser(subparsers):
    subparser = subparsers.add_parser(
        'control_loop_dataset_capture',
        help='run a control loop on each row of a dataset to test capture',
    )
    subparser.set_defaults(main=control_loop_dataset_capture)
    subparser.add_argument(
        'data_tag',
        help='tag of the raw dataset containing the aberrations file',
    )
    subparser.add_argument(
        'max_steps',
        type=int,
        help='the max number of steps in each control loop',
    )
    subparser.add_argument(
        'delta_time',
        type=float,
        help='time between timesteps',
    )
    subparser.add_argument(
        'convergence_threshold',
        type=float,
        help=('value that all Zernike coefficients must be below for the '
              'wavefront to be considered captured'),
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
        '--cores',
        default=1,
        type=int,
        help='number of cores to split the control loop runs across',
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


def control_loop_dataset_capture(cli_args):
    title('Control loop dataset capture script')

    # ====================
    # Load in the CLI args
    # ====================

    step_ri('Loading CLI args')
    data_tag = cli_args['data_tag']
    max_steps = cli_args['max_steps']
    delta_time = cli_args['delta_time']
    convergence_threshold = cli_args['convergence_threshold']
    K_p = cli_args['K_p']
    K_i = cli_args['K_i']
    K_d = cli_args['K_d']
    train_name = cli_args['train_name']
    ref_wl = cli_args['ref_wl']
    cores = cli_args['cores']
    grid_points = cli_args['grid_points']
    neural_network = cli_args['neural_network']
    response_matrix = cli_args['response_matrix']

    # =======================
    # Create the process pool
    # =======================

    step_ri('Creating the process pool')
    print(f'Using {cores} core(s)')
    pool = ProcessPool(ncpus=cores)

    # =======================
    # Load in the aberrations
    # =======================

    step_ri('Loading the aberrations')
    aberrations, zernike_terms = load_raw_sim_data_aberrations_file(data_tag)
    print(f'Zernike terms: {zernike_terms}')
    nrows = aberrations.shape[0]
    print(f'Aberration row count: {nrows}')

    # ==============================
    # Setup the control loop columns
    # ==============================

    step_ri('Setting up the shared columns')
    idx_col = np.arange(max_steps)
    delta_time_col = np.full(max_steps, delta_time)
    cumulative_time_col = np.concatenate(([0], np.cumsum(delta_time_col[1:])))
    # Turn all of the row vectors into 2D column vectors
    idx_col = idx_col[:, None]
    delta_time_col = delta_time_col[:, None]
    cumulative_time_col = cumulative_time_col[:, None]

    # ==============================================
    # Worker code to iterate over many control loops
    # ==============================================

    # The control loops will be iterated by this function (will be called
    # independently by each worker)
    def worker_iterate_control_loops(worker_idx, aberration_chunk):
        count = aberration_chunk.shape[0]
        worker_str = f'Worker [{worker_idx}]'
        if count == 0:
            print(f'{worker_str} not assigned any control loops')
            return
        print(f'{worker_str} assigned {count} control loop(s)')
        # An array consisting of the tuples (end_idx, converged) where
        # `converged` is True if the control loop did converge
        control_loop_convergence = []
        for idx in range(count):
            print(f'[{worker_idx}] Control Loop, {idx + 1}/{count}')
            # Need to create a copy of the aberrations for each time step
            coeff_steps = np.tile(aberration_chunk[idx][:, None], max_steps).T
            # Columns: row index, cumulative time, delta time, *zernike coeffs
            control_loop_steps = np.concatenate(
                (idx_col, cumulative_time_col, delta_time_col, coeff_steps),
                axis=1,
            )
            # Iterate over the control loop steps
            (end_idx, _, true_error_history,
             meas_error_history) = iterate_simulated_control_loop(
                 control_loop_steps,
                 zernike_terms,
                 K_p,
                 K_i,
                 K_d,
                 train_name,
                 ref_wl,
                 grid_points=grid_points,
                 enable_logs=False,
                 use_nn=neural_network,
                 use_rm=response_matrix,
                 early_stopping=convergence_threshold,
             )

            # Both the true and meas error must both converge
            converge_vals = [*true_error_history[-1], *meas_error_history[-1]]
            # A bool on whether the control loop converged or not
            converged = np.all(np.abs(converge_vals) <= convergence_threshold)
            control_loop_convergence.append((end_idx, converged))
        return control_loop_convergence

    # =================================================
    # Split the aberrations into chunks for each worker
    # =================================================

    step_ri('Creating the chunks for the workers')
    print(f'Splitting {nrows} control loops across {cores} core(s)')
    # Allow identification of individual workers
    worker_indexes = np.arange(cores)
    # Split the rows into chunks to pass to each worker
    aberration_chunks = np.array_split(aberrations, cores)

    # =====================
    # Run the control loops
    # =====================

    step_ri('Beginning to run control loops')
    results = pool.map(
        worker_iterate_control_loops,
        worker_indexes,
        aberration_chunks,
    )
    print('Finished running the control loops')

    # ======================================
    # Merge the worker results back together
    # ======================================

    step_ri('Merging results back together from workers')
    # Each worker has a list of tuples, so we want one big list of tuples
    merged_results = results[0]
    for result_tuple in results[1:]:
        if result_tuple is None:
            continue
        merged_results.extend(result_tuple)

    # =================
    # Print the results
    # =================

    step_ri('Convergence results')
    convergence = [bool(converged) for end_idx, converged in merged_results]
    total_captured = np.sum(convergence)
    percentage = (total_captured / nrows) * 100
    print(f'Threshold of {convergence_threshold}.')
    print(f'Captured: {total_captured}/{nrows} ({percentage:0.2f}%).')
