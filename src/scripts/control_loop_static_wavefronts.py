"""
Run many aberration rows from a dataset's aberration file. The aberration file
must be created with the `--save-aberrations-csv` arg in the `sim_data` script.

This script only works on static aberrations - they cannot change at each
timestep. If this feature is needed, then use the `control_loop_run` script
in conjunction with the `gen_zernike_time_steps` script.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.constants import (ARGS_F, CONTROL_LOOP_RESULTS_P, DATA_F,
                             MEAS_ERROR_HISTORY, TRUE_ERROR_HISTORY)
from utils.hdf_read_and_write import HDFWriteModule
from utils.iterate_simulated_control_loop import iterate_simulated_control_loop
from utils.json import json_write
from utils.load_raw_sim_data import load_raw_sim_data_aberrations_file
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title


def control_loop_static_wf_parser(subparsers):
    subparser = subparsers.add_parser(
        'control_loop_static_wavefronts',
        help='run a control loop on each row (static aberrations) of a dataset',
    )
    subparser.set_defaults(main=control_loop_static_wavefronts)
    subparser.add_argument(
        'data_tag',
        help='tag of the raw dataset containing the aberrations file',
    )
    subparser.add_argument(
        'steps',
        type=int,
        help='the number of steps in each control loop',
    )
    subparser.add_argument(
        'delta_time',
        type=float,
        help='time between timesteps',
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
        '--output-write-batch',
        type=int,
        default=50,
        help='number of control loops to run before writing out per worker',
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


def control_loop_static_wavefronts(cli_args):
    title('Control loop static wavefronts script')

    # ====================
    # Load in the CLI args
    # ====================

    step_ri('Loading CLI args')
    data_tag = cli_args['data_tag']
    steps = cli_args['steps']
    delta_time = cli_args['delta_time']
    K_p = cli_args['K_p']
    K_i = cli_args['K_i']
    K_d = cli_args['K_d']
    train_name = cli_args['train_name']
    ref_wl = cli_args['ref_wl']
    output_write_batch = cli_args['output_write_batch']
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
    idx_col = np.arange(steps)
    delta_time_col = np.full(steps, delta_time)
    cumulative_time_col = np.concatenate(([0], np.cumsum(delta_time_col[1:])))
    # Turn all of the row vectors into 2D column vectors
    idx_col = idx_col[:, None]
    delta_time_col = delta_time_col[:, None]
    cumulative_time_col = cumulative_time_col[:, None]

    # =====================
    # Create the output dir
    # =====================

    step_ri('Creating the output directory')
    # Set the model string
    if neural_network:
        tag, epoch = neural_network
        model_str = f'NN_{tag}_{epoch}'
    elif response_matrix:
        model_str = f'RM_{response_matrix}'
    out_dir = (f'{CONTROL_LOOP_RESULTS_P}/'
               f'{data_tag}_{model_str}_{K_p}_{K_i}_{K_d}')
    print(f'Path: {out_dir}')
    make_dir(out_dir)

    # ====================
    # Writing out the args
    # ====================

    step_ri('Saving all CLI args')
    json_write(f'{out_dir}/{ARGS_F}', cli_args)

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
        # Store the history so that it can be written out
        histories = {TRUE_ERROR_HISTORY: [], MEAS_ERROR_HISTORY: []}

        def write_history():
            print(f'Worker [{worker_idx}] writing out histories')
            out_file = f'{out_dir}/{worker_idx}_{DATA_F}'
            HDFWriteModule(out_file).create_and_write_hdf_simple(histories)

        for idx in range(count):
            print(f'[{worker_idx}] Control Loop, {idx + 1}/{count}')
            # Need to create a copy of the aberrations for each time step
            coeff_steps = np.tile(aberration_chunk[idx][:, None], steps).T
            # Columns: row index, cumulative time, delta time, *zernike coeffs
            control_loop_steps = np.concatenate(
                (idx_col, cumulative_time_col, delta_time_col, coeff_steps),
                axis=1,
            )
            # Iterate over the control loop steps
            (_, _, true_error_history,
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
             )
            histories[TRUE_ERROR_HISTORY].append(true_error_history)
            histories[MEAS_ERROR_HISTORY].append(meas_error_history)
            # See if it is time to do another write
            if (idx + 1) % output_write_batch == 0:
                write_history()
        # Do one final write out
        write_history()

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
    pool.map(worker_iterate_control_loops, worker_indexes, aberration_chunks)
    print('Finished running the control loops')
