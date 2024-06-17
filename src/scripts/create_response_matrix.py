"""
Create a response matrix.

The dataset should not be normalized.

The following symbols and terms will be used:
    I0 ≡ base intensity field (no Zernike aberrations)
    ΔI = total intensity field - I0
    M: response matrix
    Z: Zernike coefficients
    @: matrix multiplication (np notation)
An image can be described by `ΔI = M @ Z`.
The matrix M is the Jacobian of the fields with respect to different Zernike
polynomial terms, it can be solved for by using finite differencing.
The terms in M are given by `ΔI_i / ΔZ_j` where the intensity field has its
pixels flattened into the rows and the Zernike perturbations are the columns.
By inverting M, we can then solve for the Zernike coefficients present in a
given image by `Z = M_inv @ ΔI`.
"""

import numpy as np
from utils.constants import (BASE_INT_FIELD, PERTURBATION_AMOUNT,
                             RESPONSE_MATRICES_P, RESPONSE_MATRIX_INV,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.load_raw_sim_data_chunks import load_raw_sim_data_chunks
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def create_response_matrix_parser(subparsers):
    subparser = subparsers.add_parser(
        'create_response_matrix',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=create_response_matrix)
    subparser.add_argument(
        '--simulated-data-tag',
        help=('generate the response matrix from raw simulated data, the '
              'data should be simulated via the `sim_data` script with the '
              '`--fixed-amount-per-zernike` argument passed'),
    )


def create_response_matrix(cli_args):
    title('Create response matrix script')

    step_ri('Loading in the data')
    sim_data_tag = cli_args['simulated_data_tag']
    if sim_data_tag:
        (intensity, zernike_coeffs, zernike_terms,
         _) = load_raw_sim_data_chunks(sim_data_tag)
        # The shape of this data is (fields, pixels, pixels) and should be
        # converted to (flattened_pixels, fields)
        intensity = intensity.reshape(intensity.shape[0], -1).T
        # The last column of data is the intensity field without any Zernike
        # aberrations, so we will take our differences with respect to it
        base_field = intensity[:, -1]
        # All the columns now consist of perturbed fields for each Zernike term
        perturbation_fields = intensity[:, :-1]
        # Verify that the last row has no aberrations
        if not np.all(zernike_coeffs[-1] == 0):
            terminate_with_message('Last row not all zeros for Zernike coeffs')
        # For the perturbation amounts, the last row is for the base case,
        # so we can chop it off
        perturbation_amounts = zernike_coeffs[:-1]

    step_ri('Calculating M')
    # This is the delta intensity field portion
    field_changes = perturbation_fields - base_field[:, None]

    # Verify that all the perturbations are equal
    perturbation_amounts_diag = np.diag(perturbation_amounts)
    if not np.all(perturbation_amounts_diag == perturbation_amounts_diag[0]):
        terminate_with_message('All Zernike perturbation amounts must be '
                               'the same')
    else:
        perturbation_amount = perturbation_amounts_diag[0]

    # Verify that all off-diagonal elements are zero, so set all main diagonal
    # elements to zero to make the check easier
    perturbation_amounts[np.diag_indices(perturbation_amounts.shape[0])] = 0
    if not np.all(perturbation_amounts == 0):
        terminate_with_message('There are off diagonal Zernike perturbations')

    # The perturbation amount should be fixed for all terms
    M_matrix = field_changes / perturbation_amount

    step_ri('Calculating M_inv')
    M_matrix_inv = np.linalg.pinv(M_matrix)

    step_ri('Saving M_inv')
    output_path = f'{RESPONSE_MATRICES_P}/{sim_data_tag}.h5'
    print(f'Outputting to {output_path}')
    HDFWriteModule(output_path).create_and_write_hdf_simple({
        BASE_INT_FIELD: base_field,
        RESPONSE_MATRIX_INV: M_matrix_inv,
        PERTURBATION_AMOUNT: perturbation_amount,
        ZERNIKE_TERMS: zernike_terms,
    })
