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
from utils.constants import (BASE_INT_FIELD, INPUTS_SUM_TO_ONE,
                             PERTURBATION_AMOUNTS, RESPONSE_MATRICES_P,
                             RESPONSE_MATRIX_INV, ZERNIKE_TERMS)
from utils.hdf_read_and_write import HDFWriteModule
from utils.load_raw_sim_data import load_raw_sim_data_chunks
from utils.norm import sum_to_one
from utils.printing_and_logging import step_ri, title
from utils.simulate_camera import simulate_camera
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
    subparser.add_argument(
        '--wfs-sum-to-one',
        action='store_true',
        help='make the pixel values in each wavefront sum to 1',
    )
    subparser.add_argument(
        '--base-field-tag',
        help=('if the base field is not the last row in the data, then it '
              'should be passed via this argument; this argument should not '
              'be used if the base field is already in the data'),
    )
    subparser.add_argument(
        '--base-field-mapping',
        nargs='*',
        type=int,
        help=('map specific base fields to different portions of the data; '
              'the arguments can be repeated as many times as necessary and '
              'should specify <base field index> <starting row> <ending row>'),
    )
    subparser.add_argument(
        '--outputs-in-surface-error',
        action='store_true',
        help=('the Zernike coefficients are in terms of surface error instead '
              'of wavefront error'),
    )
    subparser.add_argument(
        '--outputs-scaling-factor',
        type=float,
        help='multiply the Zernike coefficients by a scaling factor',
    )
    subparser.add_argument(
        '--camera-for-wfs',
        nargs=3,
        help=('simulate a camera which adds noise, discretized light levels, '
              'and saturation; requires three arguments: <cam name> '
              '<exposure time in s> <countrate>; noise will be added to the '
              'input PSFs for all the data including the base field'),
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
    # ==========================================================================
    camera_for_wfs = cli_args.get('camera_for_wfs')
    if camera_for_wfs:
        step_ri('Using a camera for the wfs')
        cam_name, exposure_time, countrate = camera_for_wfs
        exposure_time = float(exposure_time)
        countrate = float(countrate)
        print(f'Camera: {cam_name}')
        print(f'Exposure time: {exposure_time} s ({1/exposure_time} Hz)')
        print(f'Countrate: {countrate} photons/s')
        # Add noise to the PSFs
        intensity = simulate_camera(
            intensity,
            cam_name,
            exposure_time,
            countrate,
        )
    # ==========================================================================
    # The shape of this data is (fields, pixels, pixels) and should be
    # converted to (flattened_pixels, fields)
    intensity = intensity.reshape(intensity.shape[0], -1).T
    # ==========================================================================
    wfs_sum_to_one = cli_args.get('wfs_sum_to_one')
    if wfs_sum_to_one:
        step_ri('Making pixel values in each wavefront sum to 1')
        intensity = sum_to_one(intensity, (0))
    # ==========================================================================
    base_field_tag = cli_args.get('base_field_tag')
    # The base field is being passed in separately
    if base_field_tag:
        (base_field, _, _, _) = load_raw_sim_data_chunks(base_field_tag)
        # ======================================================================
        if camera_for_wfs:
            print('Using the camera for the base field too')
            # Add noise to the basefield
            base_field = simulate_camera(
                base_field,
                cam_name,
                exposure_time,
                countrate,
            )
            print(f'Max electrons in pixel: {np.max(base_field)}')
        # ======================================================================
        # Like above, the base field(s) must be flattened
        base_field = base_field.reshape(base_field.shape[0], -1).T
        # ======================================================================
        wfs_sum_to_one = cli_args.get('wfs_sum_to_one')
        if wfs_sum_to_one:
            step_ri('Making pixel values in each base field sum to 1')
            base_field = sum_to_one(base_field, (0))
        # ======================================================================
        differential_fields = intensity
        perturbation_amounts = zernike_amounts
        # All the rows may not use the same base field
        base_field_mapping = cli_args.get('base_field_mapping')
        if base_field_mapping:
            elements = len(base_field_mapping)
            if elements % 3 != 0:
                terminate_with_message('Incorrect number of mapping arguments')
            for arg_idx in range(elements // 3):
                starting_arg = arg_idx * 3
                base_field_idx = base_field_mapping[starting_arg]
                idx_low = base_field_mapping[starting_arg + 1]
                idx_high = base_field_mapping[starting_arg + 2]
                print(f'Using base field at index {base_field_idx} on '
                      f'rows {idx_low} - {idx_high}')
                differential_fields[:, idx_low:idx_high] -= (
                    base_field[:, base_field_idx][:, None])
            print('Creating an averaged base field that will be saved')
            base_field_idxs = np.array(base_field_mapping[::3])
            base_field = np.sum(base_field[:, base_field_idxs], axis=1)
            base_field /= len(base_field_idxs)
            base_field = base_field[None]
        else:
            differential_fields -= base_field
    else:
        # The last column of data is the intensity field without any Zernike
        # aberrations, so we will take our differences with respect to it
        base_field = intensity[:, -1]
        # All the data now consists of perturbed fields for each Zernike term
        perturbation_fields = intensity[:, :-1]
        # Form the differential wavefronts
        differential_fields = perturbation_fields - base_field[:, None]
        # Verify that the last row has no aberrations
        if not np.all(zernike_amounts[-1] == 0):
            terminate_with_message('Last row not all zeros for Zernike coeffs')
        # For the perturbation amounts, the last row is for the base case,
        # so we can chop it off
        perturbation_amounts = zernike_amounts[:-1]

    if cli_args['outputs_in_surface_error']:
        step_ri('Converting from surface error to wavefront error')
        print('Multiplying perturbation amounts by 2')
        perturbation_amounts *= 2

    outputs_scaling_factor = cli_args.get('outputs_scaling_factor')
    if outputs_scaling_factor:
        step_ri('Adding a scaling factor to the outputs')
        print(f'Multiplying perturbation amounts by {outputs_scaling_factor}')
        perturbation_amounts *= outputs_scaling_factor

    step_ri('Chunking the data')
    # Instead of creating the response matrix for a single RMS perturbation,
    # a bunch of RMS perturbations will be averaged together to create the
    # response matrix. Below, the data will be split into chunks where each
    # chunk represents a single RMS perturbation across Zernike terms.
    if sim_data_tag_avg:
        chunks = int(perturbation_amounts.shape[0] / len(zernike_terms))
        chunked_diff_fields = np.split(differential_fields, chunks, axis=1)
        chunked_pert_amounts = np.split(perturbation_amounts, chunks)
    else:
        # For the single RMS perturbation, wrap it all inside of a single chunk
        # to make the code the same
        chunks = 1
        chunked_diff_fields = [differential_fields]
        chunked_pert_amounts = [perturbation_amounts]

    step_ri('Calculating M')
    # Below is an iterative approach to calculate the M matrix, it ends up being
    # faster than the vectorized version.
    M_matrix = np.zeros_like(chunked_diff_fields[0])
    fields_and_amounts = zip(chunked_diff_fields, chunked_pert_amounts)
    perturbation_chunk_amounts = []
    for field_changes, pert_amounts in fields_and_amounts:
        # Verify that all the perturbations are equal
        pert_amounts_diag = np.diag(pert_amounts)
        if not np.all(pert_amounts_diag == pert_amounts_diag[0]):
            terminate_with_message('All perturbation amounts must be the same '
                                   'for a set of Zernike terms')
        perturbation_amount = pert_amounts_diag[0]
        # Keep track of the perturbation for each chunk
        perturbation_chunk_amounts.append(perturbation_amount)
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
    # field_changes = np.array(chunked_diff_fields)
    # chunked_pert_amounts = np.array(chunked_pert_amounts)
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
    if camera_for_wfs:
        data_tag += '_sim_cam'
    output_path = f'{RESPONSE_MATRICES_P}/{data_tag}.h5'
    print(f'Outputting to {output_path}')
    HDFWriteModule(output_path).create_and_write_hdf_simple({
        BASE_INT_FIELD: base_field,
        RESPONSE_MATRIX_INV: M_matrix_inv,
        PERTURBATION_AMOUNTS: perturbation_chunk_amounts,
        ZERNIKE_TERMS: zernike_terms,
        INPUTS_SUM_TO_ONE: wfs_sum_to_one,
    })
