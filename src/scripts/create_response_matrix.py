"""
Create a response matrix.

For an image, we can describe it by the following:
    I (image's intensity field) = M (response matrix) * Z (Zernike coefficients)
The response matrix, M, is the Jacobian which we can solve for by using finite
differencing. Each element is given by the equation:
    (delta intensity field) / (delta Zernike perturbation)
where the intensity field has its pixels flattened out into the rows and the
Zernike terms are each of the columns.
By inverting M, we can then solve for the Zernike coefficients present in a
given image by the following:
    Z = M_inv * I [NP: Z = M_inv @ I]
"""

import numpy as np
from utils.constants import (CCD_INTENSITY, DATA_F, PERTURBATION_AMOUNT,
                             RAW_SIMULATED_DATA_P, RESPONSE_MATRICES_P,
                             RESPONSE_MATRIX, ZERNIKE_COEFFS, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule, read_hdf
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def create_response_matrix_parser(subparsers):
    """
    Example command:
        python3 main.py create_response_matrix \
            --simulated-data-tag ds_fixed_10nm
    """
    subparser = subparsers.add_parser(
        'create_response_matrix',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=create_response_matrix)
    subparser.add_argument(
        '--simulated-data-tag',
        help=('generate the response matrix from simulated data, the data '
              'should be simulated via the `sim_data` script with the '
              '`--fixed-amount-per-zernike` argument passed'),
    )


def create_response_matrix(cli_args):
    title('Create response matrix script')

    step_ri('Loading in the data')
    simulated_data_tag = cli_args['simulated_data_tag']
    if simulated_data_tag:
        print(f'Grabbing the data from simulated dataset {simulated_data_tag}')
        datafile_path = f'{RAW_SIMULATED_DATA_P}/{simulated_data_tag}/{DATA_F}'
        data = read_hdf(datafile_path)

        # The shape of this data is (fields, pixels, pixels)
        intensity = data[CCD_INTENSITY][:]
        # The shape should be converted to (flattened_pixels, fields)
        intensity = intensity.reshape(intensity.shape[0], -1).T
        # The last column of data is the intensity field without any Zernike
        # aberrations, so we will take our differences with respect to it
        base_field = intensity[:, -1]
        # All the columns now consist of perturbed fields for each Zernike term
        perturbation_fields = intensity[:, :-1]
        # For the perturbation amounts, the last row is for the base case (it
        # has no aberrations), so we can chop it off
        perturbation_amounts = data[ZERNIKE_COEFFS][:-1]
        # The Zernike terms being used
        zernike_terms = data[ZERNIKE_TERMS][:]

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
    output_path = f'{RESPONSE_MATRICES_P}/{simulated_data_tag}.h5'
    print(f'Outputting to {output_path}')
    HDFWriteModule(output_path).create_and_write_hdf_simple({
        RESPONSE_MATRIX: M_matrix_inv,
        PERTURBATION_AMOUNT: perturbation_amount,
        ZERNIKE_TERMS: zernike_terms,
    })
