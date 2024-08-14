"""
Each coefficient is perturbed using a Gaussian distribution. Then, a histogram
is plotted of the sum of the absolute differences between the target wavefront
and the propagated perturbed wavefronts. Additionally, representative wavefronts
from each bin of the histogram will be plotted out.

Commands to run this script:
    python3 main_stnp.py perturb_wavefront_v2 \
        no_aberrations all_10nm 1e-9 100 --save-plots --cores 4
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.constants import (ARGS_F, CCD_INTENSITY, FULL_INTENSITY, RANDOM_P,
                             RAW_SIMULATED_DATA_P)
from utils.json import json_load
from utils.load_optical_train import load_optical_train
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.path import make_dir
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw
from utils.sim_prop_wf import multi_worker_sim_prop_many_wf
from utils.stats_and_error import sum_of_abs


def perturb_wavefront_v2_parser(subparsers):
    subparser = subparsers.add_parser(
        'perturb_wavefront_v2',
        help='perturb the coefficients of a wavefront to see how it changes',
    )
    subparser.set_defaults(main=perturb_wavefront_v2)
    subparser.add_argument(
        'base_field_ds',
        help=('name of the dataset containing the base field, this should be '
              'a raw dataset and there should only be one row containing no '
              'aberrations (the `no_aberrations` simulation arg), this '
              'dataset will be used to load in the simulation parameters'),
    )
    subparser.add_argument(
        'target_ds',
        help=('name of the dataset to perturb the wavefront coefficients for, '
              'this should be a raw dataset'),
    )
    subparser.add_argument(
        'std',
        type=float,
        help='standard deviation in meters to perturb each coefficient by',
    )
    subparser.add_argument(
        'perturbation_rows',
        type=int,
        help='number of rows to perturb the coefficients for',
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='the row to perturb the wavefront coefficients for',
    )
    subparser.add_argument(
        '--bin-count',
        type=int,
        default=5,
        help='number of bins to use in the histogram',
    )
    subparser.add_argument(
        '--use-full-field',
        action='store_true',
        help='use the full field instead of the CCD field',
    )
    subparser.add_argument(
        '--save-plots',
        action='store_true',
        help='save plots instead of displaying them',
    )
    subparser.add_argument(
        '--cores',
        default=1,
        type=int,
        help=('number of cores to split the simulations between, more cores '
              'means faster but more memory consumption'),
    )


def perturb_wavefront_v2(cli_args):
    title('Perturb wavefront v2 script')

    step_ri('Loading in CLI args')
    base_field_ds = cli_args['base_field_ds']
    target_ds = cli_args['target_ds']
    std = cli_args['std']
    perturbation_rows = cli_args['perturbation_rows']
    row_idx = cli_args['row_idx']
    bin_count = cli_args['bin_count']
    use_full_field = cli_args['use_full_field']
    save_plots = cli_args['save_plots']
    cores = cli_args['cores']

    step_ri('Creating the process pool')
    cores = cli_args['cores']
    print(f'Using {cores} core(s)')
    pool = ProcessPool(ncpus=cores)

    step_ri('Ensuring FFTW is being used')
    proper_use_fftw()

    step_ri('Loading in simulation args')
    # Loading in the args used to simulate data from the base field
    bf_cli_args = json_load(f'{RAW_SIMULATED_DATA_P}/{base_field_ds}/{ARGS_F}')
    train_name = bf_cli_args['train_name']
    ref_wl = float(bf_cli_args['ref_wl'])
    grid_points = int(bf_cli_args['grid_points'])

    step_ri('Loading in the optical train')
    (init_beam_d, beam_ratio, optical_train, ccd_pixels,
     ccd_sampling) = load_optical_train(train_name)

    step_ri('Loading in the target dataset')
    target_ds_data = load_raw_sim_data_chunks(target_ds, use_full_field)
    zernike_terms = target_ds_data[2]
    zernike_count = len(zernike_terms)
    target_wavefront = target_ds_data[0][row_idx]
    base_zernike_coeffs = target_ds_data[1][row_idx]
    # This will either be full or CCD sampling
    plot_sampling = target_ds_data[3]
    # Sometimes the plot sampling ends up being a list with one element
    if not isinstance(plot_sampling, float):
        plot_sampling = plot_sampling[0]

    step_ri('Propagating wavefront vectors through optical setup')

    def forward_model_prop(coeffs_vectors):
        results = multi_worker_sim_prop_many_wf(
            pool,
            cores,
            init_beam_d,
            ref_wl,
            beam_ratio,
            optical_train,
            ccd_pixels,
            ccd_sampling,
            zernike_terms,
            coeffs_vectors,
            save_full_intensity=True,
            grid_points=grid_points,
        )
        fields = results[FULL_INTENSITY if use_full_field else CCD_INTENSITY]
        return fields

    # Perturb each coefficient based on a Gaussian distribution
    rng = np.random.default_rng()
    perturb_coeff = rng.normal(0, std, size=(perturbation_rows, zernike_count))
    coeff_vectors = base_zernike_coeffs + perturb_coeff
    # Propagate each perturbed wavefront through the optical setup
    perturb_fields = forward_model_prop(coeff_vectors)

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()

    plot_dir = f'{RANDOM_P}/{base_field_ds}_{target_ds}_row_{row_idx}_wf_pert/'
    make_dir(plot_dir)

    # Take the difference between the perturbed and truth wavefronts
    wavefront_diff = target_wavefront - perturb_fields
    # Compute the sum of the absolute differences
    wavefront_errors = sum_of_abs(wavefront_diff, axes=(1, 2))
    fig, ax = plt.subplots()
    _, bins, bars = plt.hist(
        wavefront_errors,
        bins=bin_count,
        rwidth=0.8,
        color='red',
        edgecolor='black',
    )
    # Put the counts over each of the bars
    plt.bar_label(bars)
    # Hide the y axis
    ax.axes.get_yaxis().set_visible(False)
    # Set the ticks to be at the edges of the bins
    ax.set_xticks(bins)
    # Set the tick labels to be formatted with 3 decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))
    plt.title(f'Gaussian std={std} m Per Term, rows={perturbation_rows}')
    plt.xlabel('Sum of Absolute (Truth WF - Perturbed Propagated WF)')
    if save_plots:
        fig.tight_layout()
        plot_path = f'{plot_dir}/hist.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    # Plot out wavefronts from each of the bins
    sorted_idxs = np.argsort(wavefront_errors)
    wavefront_errors = wavefront_errors[sorted_idxs]
    perturb_fields = perturb_fields[sorted_idxs]
    perturb_coeff = perturb_coeff[sorted_idxs]
    for idx in range(len(bins) - 1):
        bin_low = bins[idx]
        bin_high = bins[idx + 1]
        valid_wavefronts = np.where((wavefront_errors >= bin_low)
                                    & (wavefront_errors <= bin_high))[0]

        def _plot_wf_comp(chosen_idx, classifier):
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            coeff_diff = [
                float(f'{(coeff * 1e9):0.3f}')
                for coeff in perturb_coeff[chosen_idx]
            ]
            plt.suptitle(
                f'Bin {idx + 1} {classifier}, '
                f'Sum of Absolute={wavefront_errors[chosen_idx]:0.3f}\n'
                f'Coeff Diff (nm)={coeff_diff}',
                wrap=True,
            )
            ax[0].imshow(target_wavefront)
            ax[0].set_title('Original Wavefront')
            ax[1].imshow(perturb_fields[chosen_idx])
            ax[1].set_title('Perturbed Wavefront')
            im = ax[2].imshow(target_wavefront - perturb_fields[chosen_idx])
            ax[2].set_title('Difference')
            divider = make_axes_locatable(ax[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            if save_plots:
                fig.tight_layout()
                plot_path = f'{plot_dir}/wf_bin_{idx}_{classifier}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        if len(valid_wavefronts) > 0:
            _plot_wf_comp(valid_wavefronts[0], 'Low')
            _plot_wf_comp(valid_wavefronts[-1], 'High')
