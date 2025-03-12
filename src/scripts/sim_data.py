"""
This script simulates data using PROPER.

A datafile will be outputted for every worker.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.cli_args import save_cli_args
from utils.constants import ABERRATIONS_F, DATA_F, RAW_SIMULATED_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
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
        help='save the full intensity and not just the rebinned camera version',
    )
    subparser.add_argument(
        '--save-aberrations-csv',
        action='store_true',
        help='save a text file containing just the aberrations',
    )
    subparser.add_argument(
        '--save-aberrations-csv-quit',
        action='store_true',
        help='save a text file containing just the aberrations and then quit',
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
        '--fixed-amount-per-zernike-all',
        nargs=3,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters]'),
        help=('will simulate 1 row by injecting a fixed RMS error on all '
              'Zernike terms (the terms must be sequential)'),
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike',
        nargs=3,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters]'),
        help=('will simulate `([zernike term high] - [zernike term low])` '
              'rows by injecting a fixed RMS error for each Zernike term '
              'where each wavefront contains only one Zernike at a time '
              '(the terms must be sequential)'),
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike-pm',
        nargs=3,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters]'),
        help=('will simulate `([zernike term high] - [zernike term low])` '
              'rows by injecting a fixed RMS error for each Zernike term '
              'where each wavefront contains only one Zernike at a time '
              '(the terms must be sequential) for both the +/- RMS errors'),
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike-range',
        nargs=5,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[rms error points]'),
        help=('will simulate `([zernike term high] - [zernike term low]) * '
              '[rms error points]` rows by injecting a fixed RMS error in the '
              'RMS range for each Zernike term where each wavefront contains '
              'only only Zernike at a time (the terms must be sequential) '
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
        '--rand-amount-per-zernike-groups',
        nargs='+',
        metavar=('[nrows] [zernike term low] [zernike term high] '
                 '[rms error in meters low] [rms error in meters high]'),
        help=('will simulate `nrows` by injecting a random uniform RMS error '
              'between the two bounds for each Zernike term in each '
              'simulation | the bounds can be different for different Zernike '
              'terms, the four arguments can be repeated as many times as '
              'necessary to cover all the groupings, the Zernike terms just '
              'have to be sequential and have no overlap | NOTE: for any '
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
    aberrations_group.add_argument(
        '--rand-amount-per-zernike-row-then-gaussian-pert',
        nargs=6,
        metavar=('[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[pert_nrows]', '[std]'),
        help=('will simulate 1 base row by injecting a random uniform RMS '
              'error between the two bounds for each Zernike term (the terms '
              'must be sequential), then will simulate `pert_nrows` with '
              '`std` (in meters) normal error around the base row | NOTE: for '
              'any negative values, write them as \" -number\"'),
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
    save_aberrations_csv = cli_args['save_aberrations_csv']
    save_aberrations_csv_quit = cli_args['save_aberrations_csv_quit']
    append_no_aberrations_row = cli_args['append_no_aberrations_row']
    use_only_aberration_map = cli_args['use_only_aberration_map']
    no_aberrations = cli_args['no_aberrations']
    fixed_amount_per_zernike_all = cli_args['fixed_amount_per_zernike_all']
    fixed_amount_per_zernike = cli_args['fixed_amount_per_zernike']
    fixed_amount_per_zernike_pm = cli_args['fixed_amount_per_zernike_pm']
    fixed_amount_per_zernike_range = cli_args['fixed_amount_per_zernike_range']
    rand_amount_per_zernike = cli_args['rand_amount_per_zernike']
    rand_amount_per_zernike_groups = cli_args['rand_amount_per_zernike_groups']
    rand_amount_per_zernike_single = cli_args['rand_amount_per_zernike_single']
    rand_amount_per_zernike_single_each = cli_args[
        'rand_amount_per_zernike_single_each']
    rand_amount_per_zernike_row_then_gaussian_pert = cli_args[
        'rand_amount_per_zernike_row_then_gaussian_pert']

    step_ri('Creating output directory')
    output_path = f'{RAW_SIMULATED_DATA_P}/{tag}'
    make_dir(output_path)

    step_ri('Saving all CLI args')
    save_cli_args(output_path, cli_args, 'sim_data')

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, camera_pixels,
     camera_sampling) = load_optical_train(train_name)

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

    # Create a random number generator
    rng = np.random.default_rng()
    # These variables must be defined: `zernike_terms`, `aberrations`
    if no_aberrations:
        print('Will not use any aberrations')
        zernike_terms = np.array([1])
        # The aberrations must be a 2D array
        aberrations = np.array([[0]])
    elif fixed_amount_per_zernike_all:
        idx_low, idx_high, perturb = fixed_amount_per_zernike_all
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        aberrations = np.full((1, col_count), float(perturb))
        print('A single row where each term will have an RMS error of '
              f'{perturb} meters')
    elif fixed_amount_per_zernike:
        idx_low, idx_high, perturb = fixed_amount_per_zernike
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each zernike term independently
        aberrations = np.identity(col_count) * float(perturb)
        print('Each row will consist of a Zernike term with an RMS error of '
              f'{perturb} meters')
    elif fixed_amount_per_zernike_pm:
        idx_low, idx_high, perturb = fixed_amount_per_zernike_pm
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each Zernike term independently for both the +/- perturbations
        m_aberrations = np.identity(col_count) * -float(perturb)
        p_aberrations = np.identity(col_count) * float(perturb)
        aberrations = np.concatenate((m_aberrations, p_aberrations), axis=0)
        print('Each Zernike term will have two rows with the RMS errors of  '
              f'{perturb} and -{perturb} meters')
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
        aberrations = rng.uniform(perturb_low, perturb_high, (rows, col_count))
        print('Each row will consist of Zernike terms with random uniform '
              'RMS error')
    elif rand_amount_per_zernike_groups:
        rows = int(rand_amount_per_zernike_groups[0])
        group_args = rand_amount_per_zernike_groups[1:]
        # Verify there are the correct number of arguments
        if len(group_args) % 4 != 0:
            terminate_with_message('Each group must have four arguments')
        # The data that needs to be aggregated for each group
        zernike_terms = []
        aberrations = []
        col_count = 0
        for group_idx in range(len(group_args) // 4):
            print(f'Group: {group_idx}')
            groups_args = group_args[group_idx * 4:(group_idx + 1) * 4]
            group_zernikes, cols = _zernike_terms_list(*groups_args[:2])
            zernike_terms.extend(group_zernikes)
            col_count += cols
            print(f'    Zernike terms ({cols}): {group_zernikes}')
            perturb_low = float(groups_args[2])
            perturb_high = float(groups_args[3])
            print(f'    Perturbation range: {perturb_low} to {perturb_high}')
            aberrations.append(
                rng.uniform(perturb_low, perturb_high, (rows, cols)))
        # Convert the datatype back to native int
        zernike_terms = np.array([int(v) for v in zernike_terms])
        # Join together all the aberrations
        aberrations = np.concatenate(aberrations, axis=1)
        print('Each row will consist of Zernike terms with random uniform '
              'RMS error between the group bounds')
    elif rand_amount_per_zernike_single:
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*rand_amount_per_zernike_single)
        # One Zernike term per row
        coeffs = rng.uniform(perturb_low, perturb_high, rows)
        # Pick a random column for each term
        rand_cols = rng.integers(0, col_count, rows)
        aberrations = np.zeros((rows, col_count))
        aberrations[np.arange(rows), rand_cols] = coeffs
        print('Each row will consist of a random Zernike term with a random '
              'RMS error')
    elif rand_amount_per_zernike_single_each:
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*rand_amount_per_zernike_single_each)
        # The random amount to perturb each set of Zernikes by
        coeffs = rng.uniform(perturb_low, perturb_high, rows)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in coeffs:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        aberrations = np.vstack(aberrations)
        print('Each row will consist of a Zernike term with a random RMS '
              'error and a row for each Zernike term in the range')
    elif rand_amount_per_zernike_row_then_gaussian_pert:
        *initial_args, std = rand_amount_per_zernike_row_then_gaussian_pert
        (rows, perturb_low, perturb_high, zernike_terms,
         col_count) = _pert_range_setup(*initial_args)
        std = float(std)
        # Create the base row
        base_row = rng.uniform(perturb_low, perturb_high, (1, col_count))
        # Figure out the Gaussian perturbations about the base row
        perturb_amounts = rng.normal(0, std, size=(rows, col_count))
        perturbed_rows = base_row + perturb_amounts
        aberrations = np.concatenate((base_row, perturbed_rows))
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

    if save_aberrations_csv or save_aberrations_csv_quit:
        step_ri('Saving the aberrations')
        aber_path = f'{output_path}/{ABERRATIONS_F}'
        header = ', '.join([str(v) for v in zernike_terms])
        np.savetxt(
            aber_path,
            aberrations,
            delimiter=',',
            fmt='%.12f',
            header=header,
        )
        print(f'Saved to {aber_path}')
        if save_aberrations_csv_quit:
            quit()

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
        camera_pixels,
        camera_sampling,
        zernike_terms,
        aberrations,
        save_full_intensity=save_full_intensity,
        grid_points=grid_points,
        plot_path=f'{output_path}/plots/' if save_plots else None,
        use_only_aberration_map=use_only_aberration_map,
        sim_post_cb=batch_write_cb,
        worker_post_cb=write_cb,
        do_not_return_data=True,
    )

    step_ri('Simulations completed')
    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
