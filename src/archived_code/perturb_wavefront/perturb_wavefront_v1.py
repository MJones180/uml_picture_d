"""
This script perturbs each coefficient in a wavefront by a fixed amount to see
how the propagated wavefront ends up changing. Each coefficient is perturbed
independently, so there are no coupled perturbations.

Commands to run this script:
    python3 main_scnp.py perturb_wavefront_v1 \
        no_aberrations all_10nm 1e-8 --save-plots --cores 4
    python3 main_scnp.py perturb_wavefront_v1 \
        no_aberrations all_10nm 1e-8 --save-plots --cores 4 --take-diff
"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from pathos.multiprocessing import ProcessPool
from utils.constants import (ARGS_F, CAMERA_INTENSITY, FULL_INTENSITY,
                             RANDOM_P, RAW_SIMULATED_DATA_P)
from utils.json import json_load
from utils.load_optical_train import load_optical_train
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.printing_and_logging import step_ri, title
from utils.proper_use_fftw import proper_use_fftw
from utils.sim_prop_wf import multi_worker_sim_prop_many_wf


def perturb_wavefront_v1_parser(subparsers):
    subparser = subparsers.add_parser(
        'perturb_wavefront_v1',
        help='perturb the coefficients of a wavefront to see how it changes',
    )
    subparser.set_defaults(main=perturb_wavefront_v1)
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
        'perturbation_amount',
        type=float,
        help='amount to perturb each coefficient by',
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='the row to perturb the wavefront coefficients for',
    )
    subparser.add_argument(
        '--use-full-field',
        action='store_true',
        help='use the full field instead of the camera field',
    )
    subparser.add_argument(
        '--take-diff',
        action='store_true',
        help='take the difference between the target and perturbed fields',
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


def perturb_wavefront_v1(cli_args):
    title('Perturb wavefront v1 script')

    step_ri('Loading in CLI args')
    base_field_ds = cli_args['base_field_ds']
    target_ds = cli_args['target_ds']
    perturbation_amount = cli_args['perturbation_amount']
    row_idx = cli_args['row_idx']
    use_full_field = cli_args['use_full_field']
    take_diff = cli_args['take_diff']
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
    (init_beam_d, beam_ratio, optical_train, camera_pixels,
     camera_sampling) = load_optical_train(train_name)

    step_ri('Loading in the target dataset')
    target_ds_data = load_raw_sim_data_chunks(target_ds, use_full_field)
    zernike_terms = target_ds_data[2]
    zernike_count = len(zernike_terms)
    target_wavefront = target_ds_data[0][row_idx]
    base_zernike_coeffs = target_ds_data[1][row_idx]
    # This will either be full or camera sampling
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
            camera_pixels,
            camera_sampling,
            zernike_terms,
            coeffs_vectors,
            save_full_intensity=True,
            grid_points=grid_points,
        )
        fields = results[
            FULL_INTENSITY if use_full_field else CAMERA_INTENSITY]
        return fields

    # Create a vector where each coefficient is perturbed
    coeff_vectors = np.repeat(base_zernike_coeffs[None], zernike_count, axis=0)
    coeff_vectors[np.diag_indices_from(coeff_vectors)] += perturbation_amount
    # Propagate each aberrated wavefront through the optical setup
    fields = forward_model_prop(coeff_vectors)

    # Close the pool to any new jobs and remove it
    pool.close()
    pool.clear()

    # The number of rows and columns will be the nearest nxn grid
    if take_diff:
        intensity_data = fields - target_wavefront
        col_count = zernike_count
    else:
        # Make the target wavefront the first field
        intensity_data = np.concatenate((target_wavefront[None], fields))
        col_count = zernike_count + 1
    count_sqrt = int(np.ceil(col_count**0.5))
    n_cols = n_rows = count_sqrt
    plot_grid_points = target_wavefront.shape[1]
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    plt.suptitle(f'Perturbations of {perturbation_amount}')
    current_col = 0
    for plot_row in range(n_rows):
        for plot_col in range(n_cols):
            if current_col >= col_count:
                fig.delaxes(axs[plot_row, plot_col])
                continue
            intensity = intensity_data[current_col]
            axs_cell = axs[plot_row, plot_col]
            if current_col == 0 and not take_diff:
                axs_cell.set_title('Unperturbed Wavefront')
            else:
                axs_cell.set_title(current_col + 1)
            im = axs_cell.imshow(intensity, cmap='Greys_r')
            tick_count = 3
            tick_locations = np.linspace(0, plot_grid_points, tick_count)
            # Half the width of the grid in mm (originally in meters)
            grid_rad_mm = 1e3 * plot_sampling * plot_grid_points / 2
            tick_labels = np.linspace(-grid_rad_mm, grid_rad_mm, tick_count)
            # Sometimes the middle tick likes to be negative
            tick_labels[1] = 0
            # Round to two decimal places
            tick_labels = [f'{label:.2f}' for label in tick_labels]
            axs_cell.set_xticks(tick_locations, tick_labels)
            # The y ticks get plotted from top to bottom, so flip them
            axs_cell.set_yticks(tick_locations, tick_labels[::-1])

            divider = make_axes_locatable(axs_cell)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
            current_col += 1
    for ax in axs.flat:
        ax.set(xlabel='X [mm]', ylabel='Y [mm]')
    fig.tight_layout()
    if save_plots:
        plot_path = f'{RANDOM_P}/{base_field_ds}_{target_ds}_row_{row_idx}_pert'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
