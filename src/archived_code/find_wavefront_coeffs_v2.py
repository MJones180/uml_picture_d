"""
This file is copied from `find_wavefront_coeffs.py`. The difference is, instead
of solving a system of equations to find the coefficients, this script uses the
SciPy `minimize` function to fit the coefficients.

This script uses the propagated basis vectors to do the fitting. For the
propagated basis vectors, it is not possible to fit any given wavefront. In the
case of basis vectors propagated with 10 nm RMS error, it is not possible to fit
a wavefront that was propagated with 10 nm RMS error on each term. However, for
unpropagated orthogonal basis vectors containing the Zernike terms, these can
fit any unpropagated wavefront. I believe the reason for this is that the
propagated basis vectors do not span the full space, so they cannot represent
all potential wavefront vectors.

Commands to run this script:
    python3 main.py find_wavefront_coeffs_v2 \
        fixed_10nm all_10nm -50 50 --save-plots
    python3 main.py find_wavefront_coeffs_v2 \
        fixed_10nm_zernike_wf all_10nm_zernike_wf -50 50 --save-plots
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import RANDOM_P
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from scipy.optimize import minimize
from utils.printing_and_logging import step_ri, title
from utils.stats_and_error import sum_of_abs
from utils.terminate_with_message import terminate_with_message


def find_wavefront_coeffs_v2_parser(subparsers):
    subparser = subparsers.add_parser(
        'find_wavefront_coeffs_v2',
        help='test the coefficients of a wavefront',
    )
    subparser.set_defaults(main=find_wavefront_coeffs_v2)
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
        'init_coeff_bounds',
        type=int,
        nargs=2,
        help='initial bounds to pick first guess between',
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


def find_wavefront_coeffs_v2(cli_args):
    title('Find wavefront coeffs v2 script')

    step_ri('Loading in CLI args')
    terms_ds = cli_args['terms_ds']
    test_ds = cli_args['test_ds']
    init_coeff_bounds = cli_args['init_coeff_bounds']
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
    print('TRUTH: ', truth_zernike_coeffs)

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

    def minimize_func(coeffs):
        # The wavefront error can be calculated as the sum of the absolute
        # differences between the target wavefront and reconstructed wavefront
        wavefront_error = sum_of_abs(wavefront - (coeffs @ basis_terms))
        # The finite difference amount
        STEP_SIZE = 1e-8
        # To calculate the gradient, we need to perturb each coeff by itself
        coeff_vectors = np.repeat(coeffs[None], zernike_count, axis=0)
        coeff_vectors[np.diag_indices_from(coeff_vectors)] += STEP_SIZE
        # Calculate the error for each perturbed coefficient
        grad_wfe = sum_of_abs(wavefront - (coeff_vectors @ basis_terms), 1)
        # Calculate the gradient
        grad = (grad_wfe - wavefront_error) / STEP_SIZE
        return wavefront_error, grad

    minimization = minimize(
        minimize_func,
        # Initial coefficient guesses
        np.random.uniform(*init_coeff_bounds, zernike_count),
        jac=True,
        tol=1e-8,
        method='L-BFGS-B',
    )
    print(minimization)
    coeffs = minimization.x
    print(coeffs)

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
