"""
This script does not work since the propagated basis vectors do not span the
whole space, so the wavefronts cannot be properly described.

This script takes in a dataset where each row has one term with a fixed RMS
error, these rows will be used as the basis terms. The coefficients of the
wavefront from a given row in another dataset will be calculated.
Plots will be stored in `output/random/`.

Commands to run this script:
    python3 main.py find_wavefront_coeffs \
        fixed_10nm fixed_10nm --use-full-field --save-plots \
        --row-idx -2 --remove-outside-for-circle
    python3 main.py find_wavefront_coeffs \
        fixed_10nm all_10nm --use-full-field --save-plots
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import RANDOM_P
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def find_wavefront_coeffs_parser(subparsers):
    subparser = subparsers.add_parser(
        'find_wavefront_coeffs',
        help='test the coefficients of a wavefront',
    )
    subparser.set_defaults(main=find_wavefront_coeffs)
    subparser.add_argument(
        'terms_ds',
        help=('name of the dataset containing each of the Zernike terms, '
              'this should be a raw dataset and the last row should have '
              'no aberrations'),
    )
    subparser.add_argument(
        'test_ds',
        help=('name of the dataset to find the wavefront coefficients for, '
              'this should be a raw dataset'),
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='the row to find the wavefront coefficients for',
    )
    subparser.add_argument(
        '--use-full-field',
        action='store_true',
        help='use the full field instead of the CCD field',
    )
    subparser.add_argument(
        '--remove-outside-for-circle',
        action='store_true',
        help='remove corner pixels that fall outside of a circle',
    )
    subparser.add_argument(
        '--save-plots',
        action='store_true',
        help='save plots instead of displaying them',
    )


def find_wavefront_coeffs(cli_args):
    title('Find wavefront coeffs script')

    step_ri('Loading in CLI args')
    terms_ds = cli_args['terms_ds']
    test_ds = cli_args['test_ds']
    row_idx = cli_args['row_idx']
    use_full_field = cli_args['use_full_field']
    remove_outside_for_circle = cli_args['remove_outside_for_circle']
    save_plots = cli_args['save_plots']

    step_ri('Loading in the terms dataset')
    (intensity_fields, zernike_amounts, zernike_terms,
     _) = load_raw_sim_data_chunks(terms_ds, use_full_field)
    # Verify that the last row has no aberrations
    if not np.all(zernike_amounts[-1] == 0):
        terminate_with_message('Last row not all zeros for Zernike coeffs')
    # Chop off the last row with no aberrations
    zernike_amounts = zernike_amounts[:-1]
    zernike_count = len(zernike_terms)
    base_rms_error = zernike_amounts[0, 0]
    # Ensure the perturbations match the array below
    compare_arr = np.identity(zernike_count) * base_rms_error
    if not np.array_equal(compare_arr, zernike_amounts):
        terminate_with_message('Every row must have the same fixed RMS error '
                               'for each term')
    # The last row represents the base field
    base_field = intensity_fields[-1]
    # All the data now consists of perturbed fields for each Zernike term
    basis_terms = intensity_fields[:-1] - base_field

    step_ri('Loading in the test dataset')
    test_ds_data = load_raw_sim_data_chunks(test_ds, use_full_field)
    if not np.array_equal(zernike_terms, test_ds_data[2]):
        terminate_with_message('Zernike terms are not the same across the '
                               'two datasets')
    wavefront = test_ds_data[0][row_idx] - base_field
    truth_zernike_coeffs = test_ds_data[1][row_idx]

    if remove_outside_for_circle:
        step_ri('Removing corner pixels to make a circle')
        pixel_count = wavefront.shape[0]
        x_grid = np.linspace(-1, 1, pixel_count)
        x_grid = np.repeat(x_grid[None], pixel_count, axis=0)
        y_grid = x_grid.T
        valid_mask = (x_grid**2 + y_grid**2)**0.5 <= 1
        basis_terms *= valid_mask
        wavefront *= valid_mask

    step_ri('Computing coefficients')

    # Our fields now represent our basis terms. Each field, when flattened, can
    # be considered a basis vector in our space. Normally, basis vectors fill
    # the columns of a matrix, but for this they are the rows.
    basis_terms = basis_terms.reshape(zernike_count, -1)

    # Our wavefront should also be flattened so that it is a vector
    wavefront = wavefront.reshape(-1)

    # Compute the dot products between each of the basis terms
    basis_terms_dot_products = basis_terms @ basis_terms.T

    # Compute the dot product of the wavefront with each of the basis terms
    wavefront_dot_products = basis_terms @ wavefront

    # The crosstalk between the wavefront with each of the basis terms. This is
    # computed by just solving for the system of equations.
    coeffs = np.linalg.solve(basis_terms_dot_products, wavefront_dot_products)

    step_ri('Plotting a bar plot of wavefront coefficients')
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    indices = np.arange(zernike_count)
    plt.plot(indices, np.zeros(zernike_count), color='black', linewidth=1)
    plt.bar(
        indices,
        # Needs to be put back into the same units
        coeffs * base_rms_error,
        width=bar_width,
        color='blue',
        alpha=0.5,
        label='Obtained',
    )
    plt.bar(
        indices,
        truth_zernike_coeffs,
        width=bar_width,
        linewidth=2,
        edgecolor='black',
        color='none',
        label='Truth',
    )
    plt.title(f'Basis Terms Normalized at {base_rms_error} nm')
    plt.ylabel('RMS Error (nm)')
    plt.xlabel('Zernike Terms')
    plt.xticks(indices, [term for term in zernike_terms])
    plt.legend()
    if save_plots:
        fig.tight_layout()
        path = f'{RANDOM_P}/{terms_ds}_{test_ds}_row_{row_idx}_bar_graph.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    reconstructed_wf = coeffs @ basis_terms
    pixels = int(wavefront.shape[0]**0.5)
    ax[0].imshow(wavefront.reshape(pixels, pixels))
    ax[0].set_title('Original Wavefront')
    ax[1].imshow(reconstructed_wf.reshape(pixels, pixels))
    ax[1].set_title('Reconstructed Wavefront')
    if save_plots:
        fig.tight_layout()
        path = f'{RANDOM_P}/{terms_ds}_{test_ds}_row_{row_idx}_wavefront.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
