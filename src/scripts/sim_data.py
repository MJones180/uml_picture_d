"""
This script simulates data using PROPER.

A datafile will be outputted for every worker.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
import proper
from utils.constants import (ARGS_F, CCD_INTENSITY, CCD_SAMPLING, DATA_F,
                             FULL_INTENSITY, FULL_SAMPLING,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.downsample_data import downsample_data
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.load_optical_train import load_optical_train
from utils.path import get_abs_path, make_dir
from utils.plots.plot_intensity_field import plot_intensity_field
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw
from utils.terminate_with_message import terminate_with_message


def sim_data_parser(subparsers):
    subparser = subparsers.add_parser(
        'sim_data',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=sim_data)
    subparser.add_argument(
        'tag',
        help='tag for this simulated data',
    )
    subparser.add_argument(
        'train_name',
        help='name of the optical train',
    )
    subparser.add_argument(
        'ref_wl',
        help='reference wavelength in meters',
    )
    subparser.add_argument(
        '--output-write-batch',
        type=int,
        default=50,
        help='number of simulations to run before writing them out per worker',
    )
    subparser.add_argument(
        '--grid-points',
        type=int,
        default=1024,
        help='number of grid points',
    )
    subparser.add_argument(
        '--save-plots',
        action='store_true',
        help=('save plots at each step of the train, NOTE: should only do '
              'this for a few rows since the plots take extra time and space'),
    )
    subparser.add_argument(
        '--save-full-intensity',
        action='store_true',
        help='save the full intensity and not just the rebinned CCD version',
    )
    subparser.add_argument(
        '--cores',
        default=1,
        type=int,
        help=('number of cores to split the simulations between, more cores '
              'means faster but more memory consumption'),
    )
    subparser.add_argument(
        '--append-no-aberrations-row',
        action='store_true',
        help='append a row at the end with no aberrations',
    )

    aberrations_group = subparser.add_mutually_exclusive_group()
    aberrations_group.add_argument(
        '--no-aberrations',
        action='store_true',
        help='will simulate one row with no aberrations',
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike',
        nargs=3,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters]'),
        help=('will simulate `([zernike term high] - [zernike term low])` '
              'rows by injecting a fixed RMS error for each '
              'Zernike term (the terms must be sequential)'),
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike-range',
        nargs=5,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[rms error points]'),
        help=('will simulate `([zernike term high] - [zernike term low]) * '
              '[rms error points]` rows by injecting a fixed RMS error in the '
              'RMS range for each Zernike term (the terms must be sequential) '
              '| NOTE: for any negative values, write them as \" -number\"'),
    )
    aberrations_group.add_argument(
        '--rand-amount-per-zernike',
        nargs=5,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[nrows]'),
        help=('will simulate `nrows` by injecting a random uniform RMS error '
              'between the two bounds for each Zernike term in each '
              'simulation (the terms must be sequential) | NOTE: for any '
              'negative values, write them as \" -number\"'),
    )
    aberrations_group.add_argument(
        '--rand-amount-per-zernike-single',
        nargs=5,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[nrows]'),
        help=('will simulate `nrows` by injecting a random uniform RMS error '
              'between the two bounds for a single, random Zernike term in '
              'each simulation (the terms must be sequential) | NOTE: for any '
              'negative values, write them as \" -number\"'),
    )
    aberrations_group.add_argument(
        '--rand-amount-per-zernike-single-each',
        nargs=5,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[nrows]'),
        help=('will simulate `nrows * zernike_terms` by injecting a random '
              'uniform RMS error between the two bounds for a single Zernike '
              'term in each simulation and a simulation for each Zernike term '
              '(the terms must be sequential) | NOTE: for any negative '
              'values, write them as \" -number\"'),
    )


def sim_data(cli_args):
    title('Simulate data script')

    step_ri('Creating the process pool')
    cores = cli_args['cores']
    print(f'Using {cores} core(s)')
    pool = ProcessPool(ncpus=cores)

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    step_ri('Loading CLI args')
    tag = cli_args['tag']
    train_name = cli_args['train_name']
    ref_wl = float(cli_args['ref_wl'])
    output_write_batch = cli_args['output_write_batch']
    grid_points = cli_args['grid_points']
    save_plots = cli_args['save_plots']
    save_full_intensity = cli_args['save_full_intensity']
    append_no_aberrations_row = cli_args.get('append_no_aberrations_row')
    no_aberrations = cli_args['no_aberrations']
    fixed_amount_per_zernike = cli_args['fixed_amount_per_zernike']
    fixed_amount_per_zernike_range = cli_args['fixed_amount_per_zernike_range']
    rand_amount_per_zernike = cli_args['rand_amount_per_zernike']
    rand_amount_per_zernike_single = cli_args['rand_amount_per_zernike_single']
    rand_amount_per_zernike_single_each = cli_args[
        'rand_amount_per_zernike_single_each']

    step_ri('Creating output directory')
    output_path = f'{RAW_SIMULATED_DATA_P}/{tag}'
    make_dir(output_path)

    step_ri('Saving all CLI args')
    json_write(f'{output_path}/{ARGS_F}', cli_args)

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, ccd_pixels,
     ccd_sampling) = load_optical_train(train_name)

    step_ri('Figuring out aberrations')

    # Generate Zernike terms in order between two bounds
    def _zernike_terms_list(low, high):
        terms = np.arange(int(low), int(high) + 1)
        return terms, len(terms)

    # Setup for aberration simulations that perturb between two bounds
    def _pert_range_setup(idx_low, idx_high, perturb_low, perturb_high, rows):
        rows = int(rows)
        perturb_low = float(perturb_low)
        perturb_high = float(perturb_high)
        print(f'Perturbation range: {perturb_low} to {perturb_high} meters')
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        return rows, perturb_low, perturb_high, zernike_terms, col_count

    # These variables must be defined: `zernike_terms`, `aberrations`
    if no_aberrations:
        print('Will not use any aberrations')
        zernike_terms = np.array([1])
        # The aberrations must be a 2D array
        aberrations = np.array([[0]])
    elif fixed_amount_per_zernike:
        idx_low, idx_high, perturb = fixed_amount_per_zernike
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each zernike term independently
        aberrations = np.identity(col_count) * float(perturb)
        print(f'Each row will consist of a Zernike term with an RMS error of '
              f'{perturb} meters')
        print('A row will also be at the end with no Zernike aberrations')
    elif fixed_amount_per_zernike_range:
        # This function is technically for the random perturbations, but the
        # code is the same, just the argument meanings are slightly different
        (rms_points, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*fixed_amount_per_zernike_range)
        rms_vals = np.linspace(perturb_low, perturb_high, rms_points)
        print(f'The following RMS error values will be used ({rms_points}):')
        print(rms_vals)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in rms_vals:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        aberrations = np.vstack(aberrations)
    elif rand_amount_per_zernike:
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*rand_amount_per_zernike)
        aberrations = np.random.uniform(perturb_low, perturb_high,
                                        (rows, col_count))
        print('Each row will consist of Zernike terms with random uniform '
              'RMS error')
    elif rand_amount_per_zernike_single:
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*rand_amount_per_zernike_single)
        # One Zernike term per row
        coeffs = np.random.uniform(perturb_low, perturb_high, rows)
        # Pick a random column for each term
        rand_cols = np.random.randint(0, col_count, rows)
        aberrations = np.zeros((rows, col_count))
        aberrations[np.arange(rows), rand_cols] = coeffs
        print('Each row will consist of a random Zernike term with a random '
              'RMS error')
    elif rand_amount_per_zernike_single_each:
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*rand_amount_per_zernike_single_each)
        # The random amount to perturb each set of Zernikes by
        coeffs = np.random.uniform(perturb_low, perturb_high, rows)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in coeffs:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        aberrations = np.vstack(aberrations)
        print('Each row will consist of a Zernike term with a random RMS '
              'error and a row for each Zernike term in the range')
    else:
        terminate_with_message('No aberration procedure chosen')
    if zernike_terms.shape[0]:
        print(f'Zernike term range: {zernike_terms[0]} to {zernike_terms[-1]}')
    if append_no_aberrations_row:
        print('Adding a blank row of zeros at the end')
        # Add a blank row of zeros at the end
        aberrations = np.vstack((aberrations, np.zeros(col_count)))
    nrows = aberrations.shape[0]
    print(f'Total rows being simulated: {nrows}')

    # Data will be simulated and written out by this function (will be called
    # independently by each worker)
    def worker_sim_and_write(worker_idx, aberrations_chunk):
        sim_count = aberrations_chunk.shape[0]
        worker_str = f'Worker [{worker_idx}]'
        if sim_count == 0:
            print(f'{worker_str} not assigned any simulations')
            return
        print(f'{worker_str} assigned {sim_count} simulation(s)')
        # Ignore all proper logs
        proper.print_it = False
        # The data that will be written out
        simulation_data = {
            # Noll zernike term indices that are being used for aberrations
            ZERNIKE_TERMS: zernike_terms,
            # The rms error in meters associated with each of the zernike terms
            ZERNIKE_COEFFS: aberrations_chunk,
            CCD_INTENSITY: [],
            CCD_SAMPLING: ccd_sampling,
        }
        if save_full_intensity:
            simulation_data[FULL_INTENSITY] = []
            simulation_data[FULL_SAMPLING] = []

        def _write_data():
            print(f'{worker_str} writing out data')
            out_file = f'{output_path}/{worker_idx}_{DATA_F}'
            HDFWriteModule(out_file).create_and_write_hdf_simple(
                simulation_data)

        plot_idx = 0

        def _plot_intensity(wf_or_intensity, title, reset=False):
            if not save_plots:
                return
            nonlocal plot_idx
            plot_path = f'{output_path}/plots/w_{worker_idx}_sim_{sim_idx}'
            linear_path = f'{plot_path}/linear'
            log_path = f'{plot_path}/log'
            # Needs to be done for each simulation
            if reset:
                plot_idx = 0
                make_dir(linear_path)
                make_dir(log_path)
            # If it is a NP array, then it is the final intensity on the CCD,
            # otherwise it is a PROPER wavefront object
            if isinstance(wf_or_intensity, np.ndarray):
                intensity = wf_or_intensity
                plot_sampling = ccd_sampling
            else:
                intensity = proper.prop_get_amplitude(wf_or_intensity)**2
                plot_sampling = proper.prop_get_sampling(wf_or_intensity)

            def _get_plot_path(sub_dir):
                return get_abs_path(f'{sub_dir}/step_{plot_idx}.png')

            plot_intensity_field(intensity, plot_sampling, title,
                                 _get_plot_path(linear_path))
            plot_intensity_field(intensity, plot_sampling, title,
                                 _get_plot_path(log_path), True)
            plot_idx += 1

        for sim_idx in range(sim_count):
            print(f'[{worker_idx}] Simulation, {sim_idx + 1}/{sim_count}')
            # Create the wavefront that will be passed through the train
            wavefront = proper.prop_begin(init_beam_d, ref_wl, grid_points,
                                          beam_ratio)
            # Define the initial aperture
            proper.prop_circular_aperture(wavefront, init_beam_d / 2)
            # Set this as the entrance to the train
            proper.prop_define_entrance(wavefront)
            # Add in the aberrations to the wavefront
            proper.prop_zernikes(wavefront, zernike_terms,
                                 aberrations_chunk[sim_idx])
            _plot_intensity(wavefront, 'Entrance', reset=True)
            # Loop through the train
            for step in optical_train:
                # Nested list mean that the step should be eligible for plotting
                if type(step) is list:
                    step[1](wavefront)
                    _plot_intensity(wavefront, step[0])
                else:
                    step(wavefront)
            # The final wavefront intensity and sampling of its grid
            (wavefront_intensity, sampling) = proper.prop_end(wavefront)
            # Downsample to the CCD
            wf_int_ds = downsample_data(wavefront_intensity, sampling,
                                        ccd_sampling, ccd_pixels)
            # Plot the downsampled CCD intensity
            _plot_intensity(wf_int_ds, 'CCD Resampled')
            # Add the data to the output arrays
            simulation_data[CCD_INTENSITY].append(wf_int_ds)
            if save_full_intensity:
                simulation_data[FULL_INTENSITY].append(wavefront_intensity)
                simulation_data[FULL_SAMPLING].append(sampling)
            # Potentially write out the data now if a full batch is done
            if (sim_idx + 1) % output_write_batch == 0:
                _write_data()
        # Do one final write at the end
        _write_data()

    step_ri('Creating the chunks for the workers')
    print(f'Splitting {nrows} simulations across {cores} core(s)')
    # Allow identification of individual workers
    worker_indexes = np.arange(cores)
    # Split the rows into chunks to pass to each worker
    aberrations_chunks = np.array_split(aberrations, cores)

    step_ri('Beginning to run simulations')
    # Since each worker writes out its own data, no need to aggregate at the end
    pool.map(worker_sim_and_write, worker_indexes, aberrations_chunks)

    step_ri('Simulations completed')
    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
