"""
This script simulates data using PROPER.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import proper
from utils.constants import ARGS_F, DATA_F, PROPER_SIM_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
from utils.idl_rainbow_cmap import idl_rainbow_cmap
from utils.json import json_write
from utils.load_optical_train import load_optical_train
from utils.path import make_dir
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

    def plot_wf_intensity(wf, title, plot_path, plot_idx):

        def _plot(use_log=False):
            # Reset the plot
            plt.clf()
            plt.title(title)
            intensity = proper.prop_get_amplitude(wf)**2
            if use_log:
                # Ignore divide by zero errors here if they occurr
                with np.errstate(divide='ignore'):
                    intensity = np.log10(intensity)
                vmin = -8
                intensity[intensity == -np.inf] = vmin
                cmap = idl_rainbow_cmap()
                plt.imshow(intensity, vmin=vmin, vmax=0, cmap=cmap)
            else:
                plt.imshow(intensity, cmap='Greys_r')

            plt.xlabel('X [mm]')
            plt.ylabel('Y [mm]')
            tick_count = 7
            tick_locations = np.linspace(0, grid_points, tick_count)
            # Half the width of the grid in mm (originally in meters)
            grid_rad_mm = 1e3 * proper.prop_get_sampling(wf) * grid_points / 2
            tick_labels = np.linspace(-grid_rad_mm, grid_rad_mm, tick_count)
            # Sometimes the middle tick likes to be negative
            tick_labels[3] = 0
            # Round to two decimal places
            tick_labels = [f'{label:.2f}' for label in tick_labels]
            plt.xticks(tick_locations, tick_labels)
            # The y ticks get plotted from top to bottom, so flip them
            plt.yticks(tick_locations, tick_labels[::-1])
            colorbar_label = 'log10(intensity)' if use_log else 'intensity'
            plt.colorbar(label=colorbar_label)
            path_dir = 'log' if use_log else 'linear'
            plot_path_complete = f'{plot_path}/{path_dir}/step_{plot_idx}.png'
            plt.savefig(plot_path_complete, dpi=300)

        _plot(use_log=False)
        _plot(use_log=True)

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
            plot_wf_intensity(wavefront, 'Entrance', plot_path, 0)
        # Loop through the train
        for plot_idx, step in enumerate(optical_train):
            # Nested lists mean that the step should be eligible for plotting
            if type(step) is list:
                plot_title, wf_func = step
                wf_func(wavefront)
                if save_plots:
                    plot_wf_intensity(wavefront, plot_title, plot_path,
                                      plot_idx + 1)
            else:
                step(wavefront)
        # The final wavefront intensity and sampling of its grid
        (wavefront_intensity, sampling) = proper.prop_end(wavefront)

        # Total grid size in meters
        grid_size = sampling * grid_points
        # Total grid size of the CCD in meters
        ccd_grid_size = ccd_sampling * ccd_pixels
        # How much bigger our current grid is than the CCD grid
        scale_factor = grid_size / ccd_grid_size
        # Instead of jumping right to performing the crops and resizings, we
        # perform an intermediary step with a crop first so that we do not
        # need to allocate so much memory (it is also faster).
        # Grab the number of points that cover the CCD grid with a tiny bit of
        # padding so that the full CCD grid is covered.
        point_count_crop = int(np.ceil(grid_points / scale_factor * 1.05))
        point_count_crop_half = point_count_crop // 2
        grid_points_half = int(grid_points / 2)
        lower = grid_points_half - point_count_crop_half
        upper = grid_points_half + point_count_crop_half
        # A cropped version of the intensity that contains mainly the CCD grid
        wf_int = wavefront_intensity[lower:upper, lower:upper]
        # When we resize to the larger array by this updated scale factor, the
        # `point_count_crop` points in the middle will be the points that are
        # inside of the CCD grid
        scale_factor *= point_count_crop / grid_points
        point_count_scaled = int(np.round(scale_factor * point_count_crop))
        # Convert the cropped intensity to a Pillow Image object
        wf_int = Image.fromarray(wf_int)
        # Resize to the larger interpolated intensity grid
        wf_int = wf_int.resize((point_count_scaled, point_count_scaled))
        # Crop out the points that correspond to the CCD
        point_count_scaled_half = point_count_scaled // 2
        lower = point_count_scaled_half - point_count_crop_half
        upper = point_count_scaled_half + point_count_crop_half
        wf_int = wf_int.crop([lower, lower, upper, upper])
        # Resize to the actual number of pixels in the CCD
        wf_int = wf_int.resize((ccd_pixels, ccd_pixels), Image.NEAREST)
        # Convert back to a numpy array
        wf_int = np.array(wf_int)
        # Add the data to the output arrays
        simulation_data['ccd_intensity'].append(wf_int)
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
