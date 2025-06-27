"""
This script simulates data using PROPER.

A datafile will be outputted for every worker.
"""

import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.cli_args import save_cli_args
from utils.constants import (ABERRATIONS_F, DATA_F, DM_ACTUATOR_HEIGHTS,
                             DM_MASK, OT_DM_LIST, PLOTTING_LINEAR_INT,
                             PLOTTING_LINEAR_PHASE,
                             PLOTTING_LINEAR_PHASE_NON0_INT, PLOTTING_LOG_INT,
                             PLOTTING_PATH, RAW_DATA_P)
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
        nargs=4,
        metavar=('[linear intensity bool]', '[log intensity bool]',
                 '[linear phase bool]',
                 '[linear phase where int nonzero bool]'),
        help=('save plots at each step of the train, the arguments are '
              'bools on what should be plotted; NOTE: should only do this for '
              'a few rows since the plots take extra time and space'),
    )
    subparser.add_argument(
        '--save-full-ef',
        action='store_true',
        help='save the final, full electric field',
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
    subparser.add_argument(
        '--aberrations-and-dm-union',
        action='store_true',
        help=('by default, when there are DMs in the optical train, every '
              'combination of specified aberrations (N) and DM actuator '
              'heights (M) will be formed for a total of (N * M) simulated '
              'rows; when this option is passed, N == M is required and there '
              'will be a 1-to-1 mapping, for a total of N simulated rows'),
    )

    aberrations_group = subparser.add_mutually_exclusive_group()
    aberrations_group.add_argument(
        '--no-aberrations',
        nargs=2,
        metavar=('[zernike term low]', '[zernike term high]'),
        help='will simulate one row with no aberrations for given Zernikes',
    )
    aberrations_group.add_argument(
        '--explicit',
        nargs='+',
        metavar='[starting Zernike] [*RMS error on Zernike terms]',
        help='will simulate 1 row with explicit Zernike coefficient values',
    )
    aberrations_group.add_argument(
        '--fixed-amount-per-zernike-all',
        nargs='+',
        metavar='[zernike term low] [zernike term high] [rms error in meters]',
        help=('will simulate 1 row by injecting a fixed RMS error on each '
              'Zernike term | the aberration can be different for different '
              'Zernike terms, the three arguments can be repeated as many '
              'times as necessary to cover all the groupings, the Zernike '
              'terms just have to be sequential and have no overlap'),
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
        '--rand-amount-per-zernike-normal',
        nargs='+',
        metavar=('[nrows] [zernike term low] [zernike term high] '
                 '[distribution center] [standard deviation]'),
        help=('the same as `--rand-amount-per-zernike` except with random '
              'normal instead of random uniform error'),
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
        metavar=('[std]', '[zernike term low]', '[zernike term high]',
                 '[rms error in meters low]', '[rms error in meters high]',
                 '[pert_nrows]'),
        help=('will simulate 1 base row by injecting a random uniform RMS '
              'error between the two bounds for each Zernike term (the terms '
              'must be sequential), then will simulate `pert_nrows` with '
              '`std` (in meters) normal error around the base row | NOTE: for '
              'any negative values, write them as \" -number\"'),
    )

    dm_group = subparser.add_mutually_exclusive_group()
    dm_group.add_argument(
        '--explicit-actuator-heights',
        nargs='+',
        metavar='[dm idx (1D)] [actuator idx] [actuator height in meters]',
        help=('set specific actuators from their index (only active locations '
              'in the mask can be chosen); the three arguments can be '
              'repeated as many times as necessary'),
    )
    dm_group.add_argument(
        '--single-actuator-pokes',
        nargs=2,
        metavar=('[poke amount in meters]',
                 '[max actuators per DM or 0 for all]'),
        help='will poke each actuator one at a time across all DMs',
    )
    dm_group.add_argument(
        '--rand-actuator-heights',
        nargs='+',
        metavar='[nrows] [stroke low in meters] [stroke high in meters]',
        help=('will set a uniformly random height on each actuator that is '
              'between the valid stroke range; the two arguments that specify '
              'the stroke range must be set for each DM (in order)'),
    )


def sim_data(cli_args):
    title('Simulate data script')

    # ==========================================================================
    # INITIAL SETUP
    # ==========================================================================

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
    save_full_ef = cli_args['save_full_ef']
    save_full_intensity = cli_args['save_full_intensity']
    save_aberrations_csv = cli_args['save_aberrations_csv']
    save_aberrations_csv_quit = cli_args['save_aberrations_csv_quit']
    append_no_aberrations_row = cli_args['append_no_aberrations_row']
    use_only_aberration_map = cli_args['use_only_aberration_map']
    aberrations_and_dm_union = cli_args['aberrations_and_dm_union']

    step_ri('Creating output directory')
    output_path = f'{RAW_DATA_P}/{tag}'
    make_dir(output_path)

    step_ri('Saving all CLI args')
    save_cli_args(output_path, cli_args, 'sim_data')

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, camera_pixels, camera_sampling,
     optical_train_module) = load_optical_train(train_name)

    # ==========================================================================
    # ADD IN ABERRATIONS
    # ==========================================================================

    step_ri('Determining aberrations')
    zernike_terms = None
    col_count = None

    # Set the Zernike terms and there count to the local scope
    def _set_zernike_terms(terms):
        nonlocal zernike_terms, col_count
        zernike_terms = terms
        col_count = len(terms)

    # Create a list with the Zernike terms in order between two bounds
    def _gen_zernike_terms(low, high):
        terms = np.arange(int(low), int(high) + 1)
        _set_zernike_terms(terms)
        return terms, len(terms)

    # Setup for aberration simulations that perturb between two bounds
    def _pert_range_setup(idx_low, idx_high, perturb_low, perturb_high, rows):
        rows = int(rows)
        perturb_low = float(perturb_low)
        perturb_high = float(perturb_high)
        print(f'Perturbation range: {perturb_low} to {perturb_high} meters')
        _gen_zernike_terms(idx_low, idx_high)
        return rows, perturb_low, perturb_high

    # Break a list of arguments into the individual groups; the first two
    # arguments of every group are expected to be the Zernike range
    def _arg_groups(all_args, args_per_group):
        if len(all_args) % args_per_group != 0:
            terminate_with_message(f'Each group must have {args_per_group} '
                                   'arguments')
        zernike_terms = []
        groups = []
        for group_idx in range(len(all_args) // args_per_group):
            idx_low = group_idx * args_per_group
            groups_args = all_args[idx_low:idx_low + args_per_group]
            zernikes, cols = _gen_zernike_terms(*groups_args[:2])
            zernike_terms.extend(zernikes)
            group_string = (f'Group: {group_idx}\n\t'
                            f'Zernike terms ({cols}): {zernikes}\n\t')
            other_args = [float(arg) for arg in groups_args[2:]]
            groups.append([group_string, cols, *other_args])
        # Convert the datatype back to native int
        zernike_terms = np.array([int(v) for v in zernike_terms])
        _set_zernike_terms(zernike_terms)
        return groups

    # Create a random number generator
    rng = np.random.default_rng()

    def no_aberrations(idx_low, idx_high):
        print('A single row with no aberrations')
        _gen_zernike_terms(idx_low, idx_high)
        return np.full((1, col_count), 0)

    def explicit(idx_low, *coeffs):
        print('A single row with explicitly given Zernike coefficients')
        idx_low = int(idx_low)
        coeffs = [float(coeff) for coeff in coeffs]
        _gen_zernike_terms(idx_low, idx_low + len(coeffs) - 1)
        return np.array([float(coeff) for coeff in coeffs])[None]

    def fixed_amount_per_zernike_all(*all_groups):
        print('A single row where each term can have a different '
              'fixed RMS error')
        # The data that needs to be aggregated for each group
        row_aberrations = []
        for group in _arg_groups(all_groups, 3):
            group_desc, cols, aberration_amount = group
            print(group_desc + f'Aberration amount: {aberration_amount}')
            row_aberrations.extend(np.full(cols, aberration_amount))
        return np.array([row_aberrations])

    def fixed_amount_per_zernike(idx_low, idx_high, perturb):
        print('Each row will consist of a Zernike term with an RMS error of '
              f'{perturb} meters')
        _gen_zernike_terms(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each zernike term independently
        return np.identity(col_count) * float(perturb)

    def fixed_amount_per_zernike_pm(idx_low, idx_high, perturb):
        print('Each Zernike term will have two rows with the RMS errors of  '
              f'{perturb} and -{perturb} meters')
        _gen_zernike_terms(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each Zernike term independently for both the +/- perturbations
        m_aberrations = np.identity(col_count) * -float(perturb)
        p_aberrations = np.identity(col_count) * float(perturb)
        return np.concatenate((m_aberrations, p_aberrations), axis=0)

    def fixed_amount_per_zernike_range(*args):
        print('Each row will consist of a single Zernike term with '
              'aberrations between two bounds along a fixed grid')
        # This function is technically for the random perturbations, but the
        # code is the same, just the argument meanings are slightly different
        rms_points, perturb_low, perturb_high = _pert_range_setup(*args)
        rms_vals = np.linspace(perturb_low, perturb_high, rms_points)
        print(f'The following RMS error values will be used ({rms_points}):')
        print(rms_vals)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in rms_vals:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        return np.vstack(aberrations)

    def rand_amount_per_zernike(rows, *group_args):
        print('Each row will consist of Zernike terms with random uniform '
              'RMS error between the group bounds')
        rows = int(rows)
        aberrations = []
        for group in _arg_groups(group_args, 4):
            group_desc, cols, pert_low, pert_high = group
            print(group_desc +
                  f'Perturbation range: {pert_low} to {pert_high}')
            aberrations.append(rng.uniform(pert_low, pert_high, (rows, cols)))
        # Join together all the aberrations
        return np.concatenate(aberrations, axis=1)

    def rand_amount_per_zernike_normal(rows, *group_args):
        print('Each row will consist of Zernike terms with random normal '
              'RMS error')
        rows = int(rows)
        aberrations = []
        for group in _arg_groups(group_args, 4):
            group_desc, cols, dist_center, dist_std = group
            print(group_desc +
                  f'Standard deviation of {dist_std} around {dist_center}')
            aberrations.append(rng.normal(dist_center, dist_std, (rows, cols)))
        # Join together all the aberrations
        return np.concatenate(aberrations, axis=1)

    def rand_amount_per_zernike_single(*args):
        print('Each row will consist of a random Zernike term with a random '
              'RMS error')
        rows, perturb_low, perturb_high = _pert_range_setup(*args)
        # One Zernike term per row
        coeffs = rng.uniform(perturb_low, perturb_high, rows)
        # Pick a random column for each term
        rand_cols = rng.integers(0, col_count, rows)
        aberrations = np.zeros((rows, col_count))
        aberrations[np.arange(rows), rand_cols] = coeffs
        return aberrations

    def rand_amount_per_zernike_single_each(*args):
        print('Each row will consist of a Zernike term with a random RMS '
              'error and a row for each Zernike term in the range')
        rows, perturb_low, perturb_high = _pert_range_setup(*args)
        # The random amount to perturb each set of Zernikes by
        coeffs = rng.uniform(perturb_low, perturb_high, rows)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in coeffs:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        return np.vstack(aberrations)

    def rand_amount_per_zernike_row_then_gaussian_pert(std, *initial_args):
        print('A single base row with Gaussian perturbations')
        rows, perturb_low, perturb_high = _pert_range_setup(*initial_args)
        base_row = rng.uniform(perturb_low, perturb_high, (1, col_count))
        # Figure out the Gaussian perturbations about the base row
        perturb_amounts = rng.normal(0, float(std), size=(rows, col_count))
        return np.concatenate((base_row, base_row + perturb_amounts))

    for key in (
            'no_aberrations',
            'explicit',
            'fixed_amount_per_zernike_all',
            'fixed_amount_per_zernike',
            'fixed_amount_per_zernike_pm',
            'fixed_amount_per_zernike_range',
            'rand_amount_per_zernike',
            'rand_amount_per_zernike_normal',
            'rand_amount_per_zernike_single',
            'rand_amount_per_zernike_single_each',
            'rand_amount_per_zernike_row_then_gaussian_pert',
    ):
        if cli_args[key]:
            step_ri(f'Calling `{key}`')
            aberrations = locals()[key](*cli_args[key])
            break
    if zernike_terms is None:
        terminate_with_message('No aberration procedure chosen')
    print(f'Zernike term range: {zernike_terms[0]} to {zernike_terms[-1]}')
    if append_no_aberrations_row:
        print('Adding a blank row of zeros at the end')
        # Add a blank row of zeros at the end
        aberrations = np.vstack((aberrations, np.zeros(aberrations.shape[1])))
    aberration_row_count = aberrations.shape[0]
    print(f'Total aberration rows being simulated: {aberration_row_count}')

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

    # ==========================================================================
    # ADD IN DM COMMANDS
    # ==========================================================================

    extra_params = {}
    dm_spec = getattr(optical_train_module, OT_DM_LIST, None)
    if dm_spec and dm_spec != {}:
        step_ri('Loading in DMs from the optical train')
        dm_masks = []
        for dm_idx in sorted(dm_spec.keys()):
            dm_mask = dm_spec[dm_idx][DM_MASK].astype(bool)
            print(f'DM {dm_idx} - Grid: {dm_mask.shape} - '
                  f'Actuators: {dm_mask.sum()}')
            dm_masks.append(dm_mask)

        def explicit_actuator_heights(*actuator_heights):
            print('Setting explicit actuator heights')
            if len(actuator_heights) % 3 != 0:
                terminate_with_message('Each group must contain three args')
            # Store the heights for each DM
            all_dm_heights = []
            # Will contain 2D sublists which give the active actuator indexes
            dm_valid_idxs = []
            # Loop through each DM
            for dm_mask in dm_masks:
                # Grid of actuators at 0
                all_dm_heights.append(np.zeros_like(dm_mask).astype(float))
                # The active actuator indexes
                dm_valid_idxs.append(np.transpose(np.nonzero(dm_mask)))
            # Now set the explicit actuator heights
            for group_idx in range(len(actuator_heights) // 3):
                dm_idx = int(actuator_heights[group_idx * 3])
                actuator_idx = int(actuator_heights[group_idx * 3 + 1])
                actuator_height = float(actuator_heights[group_idx * 3 + 2])
                # The index for the active actuators is passed, but the actual
                # index inside of the grid is needed
                actual_actuator_idx = dm_valid_idxs[dm_idx][actuator_idx]
                print(f'[DM {dm_idx}] Actuator {actuator_idx} '
                      f'({actual_actuator_idx}): {actuator_height}')
                all_dm_heights[dm_idx][*actual_actuator_idx] = actuator_height
            # Each set of DM heights needs to be 3D
            all_dm_heights = [arr[None, :, :] for arr in all_dm_heights]
            return all_dm_heights

        def single_actuator_pokes(poke_amount, max_actuators_per_dm):
            print('Each actuator will be poked one at a time')
            # Typecast the args
            poke_amount = float(poke_amount)
            max_actuators_per_dm = int(max_actuators_per_dm)
            # If this has a value of 0, it means use all possible DM actuators
            if max_actuators_per_dm == 0:
                max_actuators_per_dm = np.inf
            else:
                print(f'Each DM will have {max_actuators_per_dm} max pokes')
            # Figure out the total number of rows that will be simulated
            total_rows = np.sum(
                [min(arr.sum(), max_actuators_per_dm) for arr in dm_masks])
            # Store the heights for each DM
            all_dm_heights = []
            # Keep track of which simulation it is
            sim_idx = 0
            # Loop through each DM
            for dm_idx, dm_mask in enumerate(dm_masks):
                # Create a grid of all zeros based on the size of the mask
                all_zeros = np.zeros_like(dm_mask).astype(float)
                # Copy the grid so there is a copy for every simulation
                dm_heights = np.repeat(all_zeros[None], total_rows, axis=0)
                # Keep track of how many rows this DM has used incase the
                # `max_actuators_per_dm` variable is set
                rows_for_dm = 0
                # Grab a list of all indexes in the mask that have True values
                for true_idx in np.argwhere(dm_mask):
                    # Poke the next actuator in the next simulation
                    dm_heights[sim_idx, *true_idx] = poke_amount
                    # Increment the counters
                    sim_idx += 1
                    rows_for_dm += 1
                    # End early if max actuators hit
                    if rows_for_dm == max_actuators_per_dm:
                        break
                # Store the actuator heights for the simulations
                all_dm_heights.append(dm_heights)
            return all_dm_heights

        def rand_actuator_heights(rows, *dm_stroke_ranges):
            print('Each actuator will be poked with a uniformly random height '
                  'from the valid stroke range')
            rows = int(rows)
            if len(dm_stroke_ranges) != len(dm_masks) * 2:
                terminate_with_message('Must provide stroke range for each DM')
            # Store the heights for each DM
            all_dm_heights = []
            # Loop through each DM
            for dm_idx, dm_mask in enumerate(dm_masks):
                # Set the stroke range for each DM
                stroke_low = float(dm_stroke_ranges[dm_idx * 2])
                stroke_high = float(dm_stroke_ranges[dm_idx * 2 + 1])
                print(f'DM {dm_idx} stroke: [{stroke_low}, {stroke_high}]')
                # Randomly set the actuator heights
                dm_heights = rng.uniform(stroke_low, stroke_high,
                                         (rows, *dm_mask.shape))
                # Mask out any actuators that should not be active
                dm_heights[:, ~dm_mask] = 0
                all_dm_heights.append(dm_heights)
            return all_dm_heights

        # Call the correction procedure to set the DM heights
        for key in (
                'explicit_actuator_heights',
                'single_actuator_pokes',
                'rand_actuator_heights',
        ):
            if cli_args[key]:
                step_ri(f'Calling `{key}`')
                dm_act_heights = locals()[key](*cli_args[key])
                break

        # Verify actuator heights are set for each DM
        if len(dm_act_heights) != len(dm_masks):
            terminate_with_message('No DM actuator height procedure chosen')
        dm_row_counts = [heights.shape[0] for heights in dm_act_heights]
        # Verify each DM has the same number of actuator height rows
        if len(set(dm_row_counts)) > 1:
            terminate_with_message('All DMs must have the same number of rows')
        dm_height_row_count = dm_row_counts[0]
        print(f'Total DM rows being simulated: {dm_height_row_count}')

        step_ri('Matching up aberration rows with DM actuator height rows')
        # Simulations are technically based on the number of aberrations rows.
        # Therefore, a 1-to-1 mapping needs to be made between DM actuator
        # heights and aberration coefficients. If `aberrations_and_dm_union` is
        # passed, then there should be the same number of actuator height rows
        # and aberration rows, in which case the two will be paired up 1-to-1.
        # Otherwise, the product between the two row counts will be taken and
        # each row will each be matched up to every other row.
        if aberrations_and_dm_union:
            print('Mapping each row 1-to-1 between the two')
            total_rows = aberration_row_count
            if aberration_row_count != dm_height_row_count:
                terminate_with_message('Must have the same number of '
                                       'aberration rows and DM height rows '
                                       f'({aberration_row_count} != '
                                       f'{dm_height_row_count})')
        else:
            print('Taking the product between the two')
            total_rows = aberration_row_count * dm_height_row_count
            # Repeat the aberrations across the total number of DM heights.
            # Repeat of 2 for [A B] produces [A A B B].
            # This is a 2D array repeating on axis 0.
            aberrations = np.repeat(aberrations, dm_height_row_count, axis=0)
            # Tile the DM heights across the total number of aberrations.
            # Tile of 2 for [A B] produces [A B A B].
            # This is a 3D array tiling on axis 0.
            for dm_idx, dm_heights in enumerate(dm_act_heights):
                dm_act_heights[dm_idx] = np.tile(
                    dm_heights,
                    (aberration_row_count, 1, 1),
                )
        print(f'Total rows being simulated: {total_rows}')

        # Store the actuator height arrays so they are passed to the simulations
        for idx, heights in enumerate(dm_act_heights):
            extra_params[DM_ACTUATOR_HEIGHTS(idx)] = heights

    # ==========================================================================
    # SIMULATIONS
    # ==========================================================================

    def write_cb(worker_idx, simulation_data):
        print(f'Worker [{worker_idx}] writing out data')
        out_file = f'{output_path}/{worker_idx}_{DATA_F}'
        HDFWriteModule(out_file).create_and_write_hdf_simple(simulation_data)

    def batch_write_cb(worker_idx, sim_idx, simulation_data):
        if (sim_idx + 1) % output_write_batch == 0:
            write_cb(worker_idx, simulation_data)

    plotting = {}
    if save_plots is not None:
        plotting = {
            PLOTTING_PATH: f'{output_path}/plots/',
            PLOTTING_LINEAR_INT: save_plots[0].lower() == 'true',
            PLOTTING_LOG_INT: save_plots[1].lower() == 'true',
            PLOTTING_LINEAR_PHASE: save_plots[2].lower() == 'true',
            PLOTTING_LINEAR_PHASE_NON0_INT: save_plots[3].lower() == 'true',
        }

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
        extra_params=extra_params,
        save_full_ef=save_full_ef,
        save_full_intensity=save_full_intensity,
        grid_points=grid_points,
        plotting=plotting,
        use_only_aberration_map=use_only_aberration_map,
        sim_post_cb=batch_write_cb,
        worker_post_cb=write_cb,
        do_not_return_data=True,
    )

    # ==========================================================================
    # DONE
    # ==========================================================================

    step_ri('Simulations completed')
    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()
