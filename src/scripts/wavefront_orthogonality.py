"""
This script takes in a dataset where each row has one term with a fixed RMS
error, these rows will be used as the basis terms. Using these rows, the cross
coupling between basis terms will be calculated. Additionally, the orthogonality
of the wavefront from a given row in another dataset will be calculated.
"""

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import RANDOM_P
from utils.idl_rainbow_cmap import idl_rainbow_cmap
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def wavefront_orthogonality_parser(subparsers):
    subparser = subparsers.add_parser(
        'wavefront_orthogonality',
        help='test the orthogonality of a wavefront',
    )
    subparser.set_defaults(main=wavefront_orthogonality)
    subparser.add_argument(
        'terms_ds',
        help=('name of the dataset containing each of the Zernike terms, '
              'this should be a raw dataset and the last row should have '
              'no aberrations'),
    )
    subparser.add_argument(
        'test_ds',
        help=('name of the dataset to check the wavefront orthogonality for, '
              'this should be a raw dataset'),
    )
    subparser.add_argument(
        '--row-idx',
        type=int,
        default=0,
        help='the row to check the wavefront orthogonality for',
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


def wavefront_orthogonality(cli_args):
    title('Wavefront orthogonality script')

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
    term_fields = intensity_fields[:-1] - base_field

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
        term_fields = term_fields * valid_mask
        wavefront = wavefront * valid_mask

    step_ri('Computing the term normalization')
    # To properly normalize the term, compute the product of each term with
    # itself and sum over the entire field. Then, the RMS error of the field
    # needs to be divided by. This will properly output the correct RMS error
    # when the field is called on itself.
    normalizations = np.sum(term_fields**2, axis=(1, 2)) / base_rms_error

    step_ri('Computing coefficients')

    def _compute_coeffs(function):
        # Need to compute the product of the wavefront with each term and then
        # sum over the entire field. Each coefficient needs to be divided by the
        # normalization factor for that term.
        return np.sum(term_fields * function, axis=(1, 2)) / normalizations

    # Calculate the crosstalk of each basis term with the other basis terms
    term_ct_coeffs = np.array([_compute_coeffs(term) for term in term_fields])
    # Calculate the crosstalk of the wavefront with each of the basis terms
    wf_coeffs = _compute_coeffs(wavefront)

    # Put all coeffs in to nm
    wf_coeffs_nm = wf_coeffs * 1e9
    truth_zernike_coeffs_nm = truth_zernike_coeffs * 1e9
    term_ct_coeffs_nm = term_ct_coeffs * 1e9

    step_ri('Plotting cross-coupling between basis terms')
    # Display the crosstalk of each of the basis terms in a grid
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f'Cross-Coupling Between Basis Terms (at {base_rms_error} '
                 'nm RMS)')
    im = ax.imshow(term_ct_coeffs_nm, cmap=idl_rainbow_cmap())
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.colorbar(im, ax=ax, label='nm RMS')
    if save_plots:
        fig.tight_layout()
        base_out_path = f'{RANDOM_P}/{terms_ds}'
        plot_path = f'{base_out_path}_basis_term_matrix.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    step_ri('Plotting a bar plot of wavefront coefficients')
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    indices = np.arange(zernike_count)
    plt.plot(indices, np.zeros(zernike_count), color='black', linewidth=1)
    plt.bar(
        indices,
        wf_coeffs_nm,
        width=bar_width,
        color='blue',
        alpha=0.5,
        label='Obtained',
    )
    plt.bar(
        indices,
        truth_zernike_coeffs_nm,
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
        plot_path = f'{base_out_path}_{test_ds}_row_{row_idx}_bar_graph.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
