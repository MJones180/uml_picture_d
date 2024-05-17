"""
This script simulates data using PROPER.
"""

import proper
import matplotlib.pyplot as plt
import numpy as np
from utils.constants import ARGS_F, DATA_F, PROPER_SIM_DATA_P
from utils.hdf_read_and_write import HDFWriteModule
from utils.json import json_write
from utils.load_optical_train import load_optical_train
from utils.path import make_dir
from utils.printing_and_logging import (dec_print_indent, inc_print_indent,
                                        step_ri, title)
from utils.proper_use_fftw import proper_use_fftw


def sim_data_parser(subparsers):
    """
    Example command:
    python3 main.py sim_data test_dataset v84 600e-9 --nrow 30 --output-write-batch 10
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
        help='save plots at each step of the train',
    )
    subparser.add_argument(
        '--enable-proper-logs',
        action='store_true',
        help='enable PROPER logs',
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
    init_beam_d, beam_ratio, optical_train = load_optical_train(train_name)

    simulation_data = {
        'intensity': [],
        'sampling': [],
    }

    def _write_data():
        out_file = f'{output_path}/{DATA_F}'
        HDFWriteModule(out_file).create_and_write_hdf_simple(simulation_data)

    def plot_wf_intensity(wf, title, plot_path):
        intensity = proper.prop_get_amplitude(wf)**2
        colorbar_label = 'intensity'
        # if SHOW_LOG_PLOT:
        #     colorbar_label = 'log10(intensity)'
        #     intensity = np.log10(intensity)
        #     intensity[intensity == -np.inf] = 0
        plt.clf()
        plt.title(title)
        plt.imshow(intensity, cmap='Greys_r')
        plt.colorbar(label=colorbar_label)
        plt.savefig(plot_path, dpi=300)

    step_ri('Beginning to run simulations')
    for sim_idx in range(nrows):
        print(f'On simulation {sim_idx + 1}/{nrows}')
        wavefront = proper.prop_begin(
            init_beam_d,
            ref_wl,
            grid_points,
            beam_ratio,
        )
        if save_plots:
            plot_path = f'{output_path}/plots/{sim_idx}/'
            make_dir(plot_path)
        plot_idx = 0
        for step in optical_train:
            if type(step) is list:
                step[1](wavefront)
                if save_plots:
                    plot_wf_intensity(wavefront, step[0],
                                      f'{plot_path}/{plot_idx}.png')
                    plot_idx += 1
            else:
                step(wavefront)

        (wavefront_intensity, sampling) = proper.prop_end(wavefront)
        simulation_data['intensity'].append(wavefront_intensity)
        simulation_data['sampling'].append(sampling)

        if (sim_idx + 1) % output_write_batch == 0:
            print('Writing out data')
            _write_data()

    step_ri('Simulations completed')
    print('Saving data one last time')
    _write_data()
