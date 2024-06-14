from utils.constants import (BASE_INT_FIELD, PERTURBATION_AMOUNT,
                             RESPONSE_MATRIX_INV, RESPONSE_MATRICES_P,
                             ZERNIKE_TERMS)
from utils.hdf_read_and_write import read_hdf
from utils.path import path_exists
from utils.printing_and_logging import dec_print_indent, step
from utils.terminate_with_message import terminate_with_message


class ResponseMatrix():

    def __init__(self, tag):
        step('Loading in the response matrix')

        self.tag = tag

        path = f'{RESPONSE_MATRICES_P}/{tag}.h5'
        print(f'Response matrix path: {path}')
        if not path_exists(path):
            terminate_with_message(f'Response matrix not found at {path}')

        data = read_hdf(f'{RESPONSE_MATRICES_P}/{tag}.h5')
        self.base_int_field = data[BASE_INT_FIELD][:]
        self.resp_mat_inv = data[RESPONSE_MATRIX_INV][:]
        self.pert_amount = data[PERTURBATION_AMOUNT][()]
        self.zernike_terms = data[ZERNIKE_TERMS][:]

        dec_print_indent()

    def get_tag(self):
        return self.tag

    def get_perturbation_amount(self):
        return self.pert_amount

    def get_zernike_terms(self):
        return self.zernike_terms

    def call_response_matrix(
        self,
        total_int_field=None,
        diff_int_field=None,
    ):
        """
        Obtain the Zernike coefficients using a response matrix.

        Parameters
        ----------
        total_int_field : np.array, optional
            The total intensity field with the 2D dimensions of (rows, pixels).
        diff_int_field : np.array, optional
            The difference of the intensity field with the 2D dimensions of
            (rows, pixels). This represents the `delta_intensity` because the
            base field has already been subtracted from the aberrated field.

        Returns
        -------
        no.array
            2D array with dimensions of (rows, Zernike coefficients).
        """
        if total_int_field is not None:
            delta_intensity = total_int_field - self.base_int_field
        else:
            delta_intensity = diff_int_field
        # The response matrix is of the shape (Zernike terms, pixels), so need
        # to transpose the change in intensity so the dimensionality works out.
        return (self.resp_mat_inv @ delta_intensity.T).T

    def __call__(self, *args, **kwargs):
        return self.call_response_matrix(*args, **kwargs)
