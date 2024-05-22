"""
This script simulates data using PROPER.
"""

import numpy as np
import proper
from utils.constants import ARGS_F, DATA_F, PROPER_SIM_DATA_P
from utils.downsample_data import downsample_data
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.load_optical_train import load_optical_train
from utils.path import get_abs_path, make_dir
from utils.plots.plot_intensity_field import plot_intensity_field
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw


def sim_data_parser(subparsers):
    """
    Example command:
        python3 main.py sim_data test_dataset v84 600e-9 --nrow 30 \
            --output-write-batch 10
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
        '--nrows',
        type=int,
        default=1,
        help='number of rows to simulate',
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


def sim_data(cli_args):
    title('Simulate data script')

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    tag = cli_args['tag']
    train_name = cli_args['train_name']
    ref_wl = float(cli_args['ref_wl'])
    nrows = cli_args['nrows']
    output_write_batch = cli_args['output_write_batch']
    grid_points = cli_args['grid_points']
    save_plots = cli_args['save_plots']
    enable_proper_logs = cli_args['enable_proper_logs']
    save_full_intensity = cli_args['save_full_intensity']

    if not enable_proper_logs:
        step_ri('Switching off PROPER logging')
        # Ignore all proper logs
        proper.print_it = False

    step_ri('Creating output directory')
    output_path = f'{PROPER_SIM_DATA_P}/{tag}'
    make_dir(output_path)

    step_ri('Saving all CLI args')
    json_write(f'{output_path}/{ARGS_F}', cli_args)

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, ccd_pixels,
     ccd_sampling) = load_optical_train(train_name)

    simulation_data = {
        'ccd_intensity': [],
        'ccd_sampling': [],
    }
    if save_full_intensity:
        simulation_data['full_intensity'] = []
        simulation_data['full_sampling'] = []

    def _write_data():
        out_file = f'{output_path}/{DATA_F}'
        HDFWriteModule(out_file).create_and_write_hdf_simple(simulation_data)

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
        simulation_data['ccd_intensity'].append(wf_int_ds)
        simulation_data['ccd_sampling'].append(ccd_sampling)
        if save_full_intensity:
            simulation_data['full_intensity'].append(wavefront_intensity)
            simulation_data['full_sampling'].append(sampling)
        # Potentially write out the data now if a full batch is done
        if (sim_idx + 1) % output_write_batch == 0:
            print('Writing out data')
            _write_data()

    step_ri('Simulations completed')
    print('Saving data one last time')
    _write_data()
