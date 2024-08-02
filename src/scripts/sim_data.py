"""
This script simulates data using PROPER.

A datafile will be outputted for every worker.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.constants import (ARGS_F, CCD_INTENSITY, CCD_SAMPLING, DATA_F,
                             FULL_INTENSITY, FULL_SAMPLING,
                             RAW_SIMULATED_DATA_P, ZERNIKE_COEFFS,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.load_optical_train import load_optical_train
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw
from utils.sim_prop_wf import multi_worker_sim_prop_many_wf
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
    subparser.add_argument(
        '--use-only-aberration-map',
        action='store_true',
        help='do not propagate a wavefront and just store the aberration map',
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
        '--fixed-amount-per-zernike-all',
        nargs=3,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters]'),
        help=('will simulate 1 row by injecting a fixed RMS error for each '
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
    append_no_aberrations_row = cli_args['append_no_aberrations_row']
    use_only_aberration_map = cli_args['use_only_aberration_map']
    no_aberrations = cli_args['no_aberrations']
    fixed_amount_per_zernike = cli_args['fixed_amount_per_zernike']
    fixed_amount_per_zernike_all = cli_args['fixed_amount_per_zernike_all']
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
        print('Each row will consist of a Zernike term with an RMS error of '
              f'{perturb} meters')
    elif fixed_amount_per_zernike_all:
        idx_low, idx_high, perturb = fixed_amount_per_zernike_all
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        aberrations = np.full((1, col_count), float(perturb))
        print('A single row where each term will have an RMS error of '
              f'{perturb} meters')
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

    def write_cb(worker_idx, simulation_data):
        print(f'Worker [{worker_idx}] writing out data')
        out_file = f'{output_path}/{worker_idx}_{DATA_F}'
        HDFWriteModule(out_file).create_and_write_hdf_simple(simulation_data)

    def batch_write_cb(worker_idx, sim_idx, simulation_data):
        if (sim_idx + 1) % output_write_batch == 0:
            write_cb(worker_idx, simulation_data)

    # Run all the simulations and save the results
    multi_worker_sim_prop_many_wf(
        pool,
        cores,
        init_beam_d,
        ref_wl,
        beam_ratio,
        optical_train,
        ccd_pixels,
        ccd_sampling,
        zernike_terms,
        aberrations,
        save_full_intensity=save_full_intensity,
        grid_points=grid_points,
        plot_path=f'{output_path}/plots/' if save_plots else None,
        use_only_aberration_map=use_only_aberration_map,
        sim_post_cb=batch_write_cb,
        worker_post_cb=write_cb,
    )

    step_ri('Simulations completed')
    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
