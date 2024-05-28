"""
This script simulates data using PROPER.
"""

import numpy as np
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
    """
    Example commands:
        python3 main.py sim_data ds_no_aberrations v84 600e-9 \
            --no-aberrations --save-plots
        python3 main.py sim_data fixed_10nm v84 600e-9 \
            --output-write-batch 10 \
            --fixed-amount-per-zernike 2 24 10e-9
        python3 main.py sim_data fixed_50nm_range v84 600e-9 \
            --output-write-batch 10 \
            --fixed-amount-per-zernike-range 2 24 " -50e-9" 50e-9 21
        python3 main.py sim_data random_50nm v84 600e-9 \
            --output-write-batch 10 \
            --rand-amount-per-zernike 2 24 " -50e-9" 50e-9 500
        python3 main.py sim_data random_50nm_single v84 600e-9 \
            --output-write-batch 10 \
            --rand-amount-per-zernike-single 2 24 " -50e-9" 50e-9 500
    """
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
        help='number of simulations to run before writing them out',
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
        '--enable-proper-logs',
        action='store_true',
        help='enable PROPER logs',
    )
    subparser.add_argument(
        '--save-full-intensity',
        action='store_true',
        help='save the full intensity and not just the rebinned CCD version',
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


def sim_data(cli_args):
    title('Simulate data script')

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    step_ri('Loading CLI args')
    tag = cli_args['tag']
    train_name = cli_args['train_name']
    ref_wl = float(cli_args['ref_wl'])
    output_write_batch = cli_args['output_write_batch']
    grid_points = cli_args['grid_points']
    save_plots = cli_args['save_plots']
    enable_proper_logs = cli_args['enable_proper_logs']
    save_full_intensity = cli_args['save_full_intensity']
    no_aberrations = cli_args['no_aberrations']
    fixed_amount_per_zernike = cli_args['fixed_amount_per_zernike']
    fixed_amount_per_zernike_range = cli_args['fixed_amount_per_zernike_range']
    rand_amount_per_zernike = cli_args['rand_amount_per_zernike']
    rand_amount_per_zernike_single = cli_args['rand_amount_per_zernike_single']

    if not enable_proper_logs:
        step_ri('Switching off PROPER logging')
        # Ignore all proper logs
        proper.print_it = False

    step_ri('Creating output directory')
    output_path = f'{RAW_SIMULATED_DATA_P}/{tag}'
    make_dir(output_path)

    step_ri('Saving all CLI args')
    json_write(f'{output_path}/{ARGS_F}', cli_args)

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, ccd_pixels,
     ccd_sampling) = load_optical_train(train_name)

    # The plotting function
    def _plot_intensity(wf_or_intensity, title, plot_path, plot_idx):
        # If it is a NP array, then it is the final intensity on the CCD,
        # otherwise it is a PROPER wavefront object
        if isinstance(wf_or_intensity, np.ndarray):
            intensity = wf_or_intensity
            plot_sampling = ccd_sampling
        else:
            intensity = proper.prop_get_amplitude(wf_or_intensity)**2
            plot_sampling = proper.prop_get_sampling(wf_or_intensity)

        def _get_plot_path(sub_dir):
            return get_abs_path(f'{plot_path}/{sub_dir}/step_{plot_idx}.png')

        plot_intensity_field(intensity, plot_sampling, title,
                             _get_plot_path('linear'))
        plot_intensity_field(intensity, plot_sampling, title,
                             _get_plot_path('log'), True)

    step_ri('Figuring out aberrations')

    # Generate Zernike terms in order between two bounds
    def _zernike_terms_list(low, high):
        terms = np.arange(int(low), int(high) + 1)
        return terms, len(terms)

    # Setup for aberration simulations that perturb between two bounds
    def _perturb_range_setup(idx_low, idx_high, perturb_low, perturb_high,
                             nrows):
        nrows = int(nrows)
        perturb_low = float(perturb_low)
        perturb_high = float(perturb_high)
        print(f'Perturbation range: {perturb_low} to {perturb_high} meters')
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        return nrows, perturb_low, perturb_high, zernike_terms, col_count

    # These variables must be defined: `nrows`, `zernike_terms`, `aberrations`
    if no_aberrations:
        nrows = 1
        print('Will not use any aberrations')
        # These three variables must be defined, but they will not be used of
        # course if there are no aberrations
        zernike_terms = np.array([])
        aberrations = []
    elif fixed_amount_per_zernike:
        idx_low, idx_high, perturb = fixed_amount_per_zernike
        zernike_terms, col_count = _zernike_terms_list(idx_low, idx_high)
        # For this we just need an identity matrix to represent perturbing
        # each zernike term independently
        aberrations = np.identity(col_count) * float(perturb)
        # Add a blank row of zeros at the end
        aberrations = np.vstack((aberrations, np.zeros(col_count)))
        nrows = aberrations.shape[0]
        print(f'Each row will consist of a Zernike term with an RMS error of '
              f'{perturb} meters')
        print('A row will also be at the end with no Zernike aberrations')
    elif fixed_amount_per_zernike_range:
        # This function is technically for the random perturbations, but the
        # code is the same, just the argument meanings are slightly different
        (rms_points, perturb_low, perturb_high, zernike_terms,
         col_count) = _perturb_range_setup(*fixed_amount_per_zernike_range)
        rms_vals = np.linspace(perturb_low, perturb_high, rms_points)
        print(f'The following RMS error values will be used ({rms_points}):')
        print(rms_vals)
        aberrations = []
        # Calculate the identity RMS error for each point
        for rms_val in rms_vals:
            aberrations.append(np.identity(col_count) * rms_val)
        # Stack them all so the shape is (Zernike cols * points, Zernike cols)
        aberrations = np.vstack(aberrations)
        nrows = aberrations.shape[0]
    elif rand_amount_per_zernike:
        (nrows, perturb_low, perturb_high, zernike_terms,
         col_count) = _perturb_range_setup(*rand_amount_per_zernike)
        aberrations = np.random.uniform(perturb_low, perturb_high,
                                        (nrows, col_count))
        print('Each row will consist of Zernike terms with random uniform '
              'RMS error')
    elif rand_amount_per_zernike_single:
        (nrows, perturb_low, perturb_high, zernike_terms,
         col_count) = _perturb_range_setup(*rand_amount_per_zernike_single)
        # One Zernike term per row
        coeffs = np.random.uniform(perturb_low, perturb_high, nrows)
        # Pick a random column for each term
        rand_cols = np.random.randint(0, col_count, nrows)
        aberrations = np.zeros((nrows, col_count))
        aberrations[np.arange(nrows), rand_cols] = coeffs
        print('Each row will consist of a random Zernike term with a random '
              'RMS error')
    else:
        terminate_with_message('No aberration procedure chosen')
    if zernike_terms.shape[0]:
        print(f'Zernike term range: {zernike_terms[0]} to {zernike_terms[-1]}')
    print(f'Total rows being simulated: {nrows}')

    # The data that will be written out
    simulation_data = {
        # Noll zernike term indices that are being used for aberrations
        ZERNIKE_TERMS: zernike_terms,
        # The rms error in meters associated with each of the zernike terms
        ZERNIKE_COEFFS: aberrations,
        CCD_INTENSITY: [],
        CCD_SAMPLING: ccd_sampling,
    }
    if save_full_intensity:
        simulation_data[FULL_INTENSITY] = []
        simulation_data[FULL_SAMPLING] = []

    def _write_data():
        out_file = f'{output_path}/{DATA_F}'
        HDFWriteModule(out_file).create_and_write_hdf_simple(simulation_data)

    step_ri('Beginning to run simulations')
    for sim_idx in range(nrows):
        print(f'Simulation {sim_idx + 1}/{nrows}')
        # Create the wavefront that will be passed through the train
        wavefront = proper.prop_begin(
            init_beam_d,
            ref_wl,
            grid_points,
            beam_ratio,
        )
        # Define the initial aperture
        proper.prop_circular_aperture(wavefront, init_beam_d / 2)
        # Set this as the entrance to the train
        proper.prop_define_entrance(wavefront)
        # If there are aberrations then it will be a np array, otherwise it will
        # be just a native list
        if isinstance(aberrations, np.ndarray):
            proper.prop_zernikes(wavefront, zernike_terms,
                                 aberrations[sim_idx])
        if save_plots:
            plot_path = f'{output_path}/plots/sim_{sim_idx}'
            make_dir(f'{plot_path}/linear')
            make_dir(f'{plot_path}/log')
            _plot_intensity(wavefront, 'Entrance', plot_path, 0)
        # Loop through the train
        for plot_idx, step in enumerate(optical_train, 1):
            # Nested lists mean that the step should be eligible for plotting
            if type(step) is list:
                step[1](wavefront)
                if save_plots:
                    _plot_intensity(wavefront, step[0], plot_path, plot_idx)
            else:
                step(wavefront)
        # The final wavefront intensity and sampling of its grid
        (wavefront_intensity, sampling) = proper.prop_end(wavefront)
        # Downsample to the CCD
        wf_int_ds = downsample_data(wavefront_intensity, sampling,
                                    ccd_sampling, ccd_pixels)
        if save_plots:
            # Plot the downsampled CCD intensity
            _plot_intensity(wf_int_ds, 'CCD Resampled', plot_path,
                            plot_idx + 1)
        # Add the data to the output arrays
        simulation_data[CCD_INTENSITY].append(wf_int_ds)
        if save_full_intensity:
            simulation_data[FULL_INTENSITY].append(wavefront_intensity)
            simulation_data[FULL_SAMPLING].append(sampling)
        # Potentially write out the data now if a full batch is done
        if (sim_idx + 1) % output_write_batch == 0:
            print('Writing out data')
            _write_data()

    step_ri('Simulations completed')
    print('Saving data one last time')
    _write_data()
