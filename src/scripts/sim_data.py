"""
This script simulates data using PROPER.
"""

import proper
import matplotlib.pyplot as plt
import numpy as np
from utils.load_optical_train import load_optical_train
from utils.printing_and_logging import (dec_print_indent, inc_print_indent,
                                        step_ri, title)
from utils.proper_use_fftw import proper_use_fftw


def sim_data_parser(subparsers):
    """
    Example command:
    python3 main.py sim_data v84 600e-9
    """
    subparser = subparsers.add_parser(
        'sim_data',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=sim_data)
    subparser.add_argument(
        'train_name',
        help='name of the optical train',
    )
    subparser.add_argument(
        'ref_wl',
        help='reference wavelength in meters',
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


def sim_data(cli_args):
    title('Simulate data script')

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    step_ri('Calling the optical train')

    # def plot_wf_intensity(wf, title, plot_idx):
    #     amp = proper.prop_get_amplitude(wf)
    #     intensity = amp**2
    #     colorbar_label = 'intensity'
    #     # if SHOW_LOG_PLOT:
    #     #     colorbar_label = 'log10(intensity)'
    #     #     intensity = np.log10(intensity)
    #     #     intensity[intensity == -np.inf] = 0
    #     plt.clf()
    #     plt.title(title)
    #     plt.imshow(intensity, cmap='Greys_r')
    #     plt.colorbar(label=colorbar_label)
    #     plt.savefig(f'plot_output/{plot_idx}.png', dpi=300)

    train_name = cli_args['train_name']
    ref_wl = float(cli_args['ref_wl'])
    grid_points = cli_args['grid_points']
    save_plots = cli_args['save_plots']

    init_beam_d, beam_ratio, optical_train = load_optical_train(train_name)

    wavefront = proper.prop_begin(init_beam_d, ref_wl, grid_points, beam_ratio)
    plot_idx = 0
    for step in optical_train:
        if type(step) is list:
            step[1](wavefront)
            # if save_plots:
            #     plot_wf_intensity(wavefront, step[0], plot_idx)
            #     plot_idx += 1
        else:
            step(wavefront)

    (wf, sampling) = proper.prop_end(wavefront)

    print(sampling)
