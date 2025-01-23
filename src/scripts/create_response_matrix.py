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
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.printing_and_logging import step_ri, title
from utils.terminate_with_message import terminate_with_message


def create_response_matrix_parser(subparsers):
    subparser = subparsers.add_parser(
        'create_response_matrix',
        help='simulate data using PROPER',
    )
    subparser.set_defaults(main=create_response_matrix)
    subparser.add_argument(
        '--simulated-data-tag-single',
        help=('generate the response matrix from raw simulated data, the '
              'data should be simulated via the `sim_data` script with the '
              '`--fixed-amount-per-zernike` argument passed'),
    )
    subparser.add_argument(
        '--simulated-data-tag-average',
        help=('generate the response matrix from raw simulated data, the '
              'data should be simulated via the `sim_data` script with either '
              'the `--rand-amount-per-zernike-single-each` or '
              '`--fixed-amount-per-zernike-range` arguments passed; '
              'the average of all the perturbations will be taken'),
    )


def create_response_matrix(cli_args):
    title('Create response matrix script')

    step_ri('Figuring out how the data should be loaded')
    sim_data_tag_single = cli_args['simulated_data_tag_single']
    sim_data_tag_avg = cli_args['simulated_data_tag_average']
    if sim_data_tag_single:
        print('Creating response matrix from a single RMS perturbation')
        data_tag = sim_data_tag_single
    elif sim_data_tag_avg:
        print('Creating response matrix by averaging multiple RMS '
              'perturbations together')
        data_tag = sim_data_tag_avg
    else:
        terminate_with_message('No method chosen to create response matrix')

    step_ri('Loading in the data')
    (intensity, zernike_amounts, zernike_terms,
     _) = load_raw_sim_data_chunks(data_tag)
    # The shape of this data is (fields, pixels, pixels) and should be
    # converted to (flattened_pixels, fields)
    intensity = intensity.reshape(intensity.shape[0], -1).T
    # The last column of data is the intensity field without any Zernike
    # aberrations, so we will take our differences with respect to it
    base_field = intensity[:, -1]
    # All the data now consists of perturbed fields for each Zernike term
    perturbation_fields = intensity[:, :-1]
    # Verify that the last row has no aberrations
    if not np.all(zernike_amounts[-1] == 0):
        terminate_with_message('Last row not all zeros for Zernike coeffs')
    # For the perturbation amounts, the last row is for the base case,
    # so we can chop it off
    perturbation_amounts = zernike_amounts[:-1]

    step_ri('Chunking the data')
    # Instead of creating the response matrix for a single RMS perturbation,
    # a bunch of RMS perturbations will be averaged together to create the
    # response matrix. Below, the data will be split into chunks where each
    # chunk represents a single RMS perturbation across Zernike terms.
    if sim_data_tag_avg:
        chunks = int(perturbation_amounts.shape[0] / len(zernike_terms))
        chunked_pert_fields = np.split(perturbation_fields, chunks, axis=1)
        chunked_pert_amounts = np.split(perturbation_amounts, chunks)
    else:
        # For the single RMS perturbation, wrap it all inside of a single chunk
        # to make the code the same
        chunks = 1
        chunked_pert_fields = [perturbation_fields]
        chunked_pert_amounts = [perturbation_amounts]

    step_ri('Calculating M')
    # Below is an iterative approach to calculate the M matrix, it ends up being
    # faster than the vectorized version.
    M_matrix = np.zeros_like(chunked_pert_fields[0])
    fields_and_amounts = zip(chunked_pert_fields, chunked_pert_amounts)
    for pert_fields, pert_amounts in fields_and_amounts:
        # This is the delta intensity field portion
        field_changes = pert_fields - base_field[:, None]
        # Verify that all the perturbations are equal
        pert_amounts_diag = np.diag(pert_amounts)
        if not np.all(pert_amounts_diag == pert_amounts_diag[0]):
            terminate_with_message('All perturbation amounts must be the same '
                                   'for a set of Zernike terms')
        perturbation_amount = pert_amounts_diag[0]
        # Verify that all off-diagonal elements are zero, so set all main
        # diagonal elements to zero to make the check easier
        pert_amounts[np.diag_indices(pert_amounts.shape[0])] = 0
        if not np.all(pert_amounts == 0):
            terminate_with_message('Off diagonal perturbations present')
        # The perturbation amount should be fixed for all terms
        M_matrix += field_changes / perturbation_amount
    M_matrix /= chunks
    # ==========================================================================
    # Below is a vectorized version to calculate the M matrix, but it ends up
    # being slower. Commenting it out for now, could be handy to have the code
    # around for later.
    # ==========================================================================
    # chunked_pert_fields = np.array(chunked_pert_fields)
    # chunked_pert_amounts = np.array(chunked_pert_amounts)
    # field_changes = chunked_pert_fields - base_field[:, None]
    # # Need to wrap in an array otherwise it is ends up being read only.
    # diags = np.array(np.diagonal(chunked_pert_amounts, 0, 1, 2))
    # if not np.all(diags[:, 1:] == diags[:, :-1]):
    #     terminate_with_message('All perturbation amounts must be the same '
    #                            'for a set of Zernike terms')
    # M_matrix = field_changes / chunked_pert_amounts[:, 0, 0][:, None, None]
    # chunked_pert_amounts[:, *np.diag_indices(len(zernike_terms))] = 0
    # if not np.all(chunked_pert_amounts == 0):
    #     terminate_with_message('Off diagonal perturbations present')
    # M_matrix = np.average(M_matrix, 0)
    # ==========================================================================

    step_ri('Calculating M_inv')
    M_matrix_inv = np.linalg.pinv(M_matrix)

    step_ri('Saving M_inv')
    output_path = f'{RESPONSE_MATRICES_P}/{data_tag}.h5'
    print(f'Outputting to {output_path}')
    HDFWriteModule(output_path).create_and_write_hdf_simple({
        BASE_INT_FIELD: base_field,
        RESPONSE_MATRIX_INV: M_matrix_inv,
        PERTURBATION_AMOUNT: perturbation_amount,
        ZERNIKE_TERMS: zernike_terms,
    })
