import numpy as np
import torch
import torch.nn as nn
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import read_hdf
from utils.norm import z_score_denormalize


class JacobianEF(nn.Module):

    def __init__(
        self,
        device,
        output_z_score_mean,
        output_z_score_std,
        dm1_modes_tag=None,
        dm1_modes_table=None,
        dm1_modes_count=None,
        dm1_modes_transpose=None,
        dm2_modes_tag=None,
        dm2_modes_table=None,
        dm2_modes_count=None,
        dm2_modes_transpose=None,
        jacobian_tag=None,
        jacobian_table=None,
        lambda_scaling=None,
        apply_log_scaling=None,
    ):
        """The JacobianEF class.

        Parameters
        ----------
        device : str
            The torch device to use.
        output_z_score_mean : np.array
            The output Z-score mean values.
        output_z_score_std : np.array
            The output Z-score STD values.
        dm1_modes_tag : str
            Tag of the raw dataset which contains the modes for DM1.
        dm1_modes_table : str
            Name of the table containing the modes for DM1.
        dm1_modes_count : str
            Number of modes to use from DM1.
        dm1_modes_transpose : bool
            Whether to transpose the modes of DM1.
        dm2_modes_tag : str
            Tag of the raw dataset which contains the modes for DM2.
        dm2_modes_table : str
            Name of the table containing the modes for DM2.
        dm2_modes_count : str
            Number of modes to use from DM2.
        dm2_modes_transpose : bool
            Whether to transpose the modes of DM2.
        jacobian_tag : str
            Tag of the raw dataset which contains the Jacobian (DMs -> EF).
        jacobian_table : str
            Name of the table containing the Jacobian.
        lambda_scaling : float
            Scaling factor to apply to the loss.
        apply_log_scaling : bool
            Take the log of the residual EF.

        Notes
        -----
        This loss function requires there to be two DMs which use individual
        Z-score output normalization. Additionally, the outputs must be in terms
        of another basis, as they will be converted to actuator heights.
        """
        super().__init__()

        def _grab_param(arg, desired_type=float):
            if arg is not None:
                return desired_type(arg)
            return None

        # Move the output norm data to torch
        self.z_score_mean = torch.from_numpy(output_z_score_mean).to(device)
        self.z_score_std = torch.from_numpy(output_z_score_std).to(device)

        def _modes(tag, table, count, transpose, str_name):
            modes_path = f'{RAW_DATA_P}/{tag}/0_{DATA_F}'
            print(f'Loading the {str_name} modes: {modes_path}')
            modes = read_hdf(modes_path)[table][:]
            if bool(_grab_param(transpose, int)):
                print(f'Transposing {str_name} modes')
                modes = modes.T
            print(f'Modes for {str_name}: {count}')
            modes = modes[:int(count)]
            modes = modes.astype(np.float32)
            return torch.from_numpy(modes).to(device)

        self.dm1_modes = _modes(dm1_modes_tag, dm1_modes_table,
                                dm1_modes_count, dm1_modes_transpose, 'DM1')
        self.dm2_modes = _modes(dm2_modes_tag, dm2_modes_table,
                                dm2_modes_count, dm2_modes_transpose, 'DM2')

        jac_path = f'{RAW_DATA_P}/{jacobian_tag}/0_{DATA_F}'
        print(f'Loading the Jacobian: {jac_path}')
        jacobian = read_hdf(jac_path)[jacobian_table][:].astype(np.float32)
        self.jacobian = torch.from_numpy(jacobian).to(device)

        self.lambda_scaling = _grab_param(lambda_scaling)
        if self.lambda_scaling is None:
            self.lambda_scaling = 1

        self.apply_log_scaling = bool(_grab_param(apply_log_scaling, int))

    def _get_actuator_heights(self, outputs):
        # Denormalize the outputs
        outputs_denorm = z_score_denormalize(outputs, self.z_score_mean,
                                             self.z_score_std)
        # Split the coefficients associated with each DM
        dm1_coeffs, dm2_coeffs = torch.tensor_split(outputs_denorm, 2, -1)
        # Convert from coefficients to actuator heights
        dm1_heights = torch.matmul(dm1_coeffs, self.dm1_modes)
        dm2_heights = torch.matmul(dm2_coeffs, self.dm2_modes)
        # Join the actuator heights together
        return torch.cat((dm1_heights, dm2_heights), dim=-1)

    def forward(self, model_outputs, truth_outputs):
        model_output_heights = self._get_actuator_heights(model_outputs)
        truth_output_heights = self._get_actuator_heights(truth_outputs)
        # Compute the residual in terms of actuator heights
        residual_heights = model_output_heights - truth_output_heights
        # Obtain the residual EF based on the Jacobian
        residual_ef = torch.matmul(residual_heights, self.jacobian.T)
        pixel_intensity_error = residual_ef**2
        if self.apply_log_scaling:
            pixel_intensity_error = torch.log10(1 + pixel_intensity_error)
        loss = self.lambda_scaling * pixel_intensity_error
        return loss.mean()
