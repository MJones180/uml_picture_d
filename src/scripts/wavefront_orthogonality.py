"""
Notes on this script
"""

import matplotlib.pyplot as plt
import numpy as np
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


def wavefront_orthogonality(cli_args):
    title('Wavefront orthogonality script')

    step_ri('Loading in CLI args')
    terms_ds = cli_args['terms_ds']
    test_ds = cli_args['test_ds']
    row_idx = cli_args['row_idx']
    use_full_field = cli_args['use_full_field']

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

    pixels = wavefront.shape[0]
    x_grid = np.linspace(-1, 1, pixels)
    x_grid = np.repeat(x_grid[None], pixels, axis=0)
    y_grid = x_grid.T
    valid_mask = (x_grid**2 + y_grid**2)**0.5 <= 1
    wavefront = wavefront * valid_mask
    term_fields = term_fields * valid_mask

    step_ri('Computing the term normalization')
    # To properly normalize the term, compute the product of each term with
    # itself and sum over the entire field. Then, the RMS error of the field
    # needs to be divided by. This will properly output the correct RMS error
    # when the field is called on itself.
    normalizations = np.sum(term_fields**2, axis=(1, 2)) / base_rms_error

    step_ri('Computing the coefficients for each term')
    # Need to compute the product of the wavefront with each term and then sum
    # over the entire field. Each coefficient needs to be divided by the
    # normalization factor for that term.
    coeffs = np.sum(term_fields * wavefront, axis=(1, 2)) / normalizations

    step_ri('Plotting a bar plot')
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8
    indices = np.arange(zernike_count)
    plt.plot(indices, np.zeros(zernike_count), color='black', linewidth=1)
    plt.bar(
        indices,
        coeffs * 1e9,
        width=bar_width,
        color='blue',
        alpha=0.5,
        label='Obtained',
    )
    plt.bar(
        indices,
        truth_zernike_coeffs * 1e9,
        width=bar_width,
        linewidth=2,
        edgecolor='black',
        color='none',
        label='Truth',
    )
    plt.ylabel('RMS Error (nm)')
    plt.xlabel('Zernike Terms')
    plt.xticks(indices, [term for term in zernike_terms])
    plt.legend()
    plt.show()
    plt.show()
