import numpy as np
import torch
import torch.nn as nn
from utils.create_grid_mask import create_grid_mask
from utils.hdf_read_and_write import read_hdf
from utils.load_raw_sim_data import raw_sim_data_chunk_paths
from utils.norm import min_max_norm, z_score_denormalize


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
        speckle_targeting=None,
        apply_radial_weighting=None,
        add_residual_stroke_mse=None,
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
            Take the log of the residual intensity.
        speckle_targeting : float
            Instead of targeting all pixels, only target the top N% speckles.
        apply_radial_weighting : str
            Apply a radial weighting to the EF pixels; four arguments expected
            as a single str separated by commas: mask size in pixels, inner
            radius ratio, outer radius ratio, and max weight at center
        add_residual_stroke_mse : float
            Add the MSE of the residual stroke to the total loss; passed
            value specifies the scaling factor. This loss is added after the
            existing loss is multiplied by the `lambda_scaling`.

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
            modes_path = raw_sim_data_chunk_paths(tag)[0]
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

        jac_path = raw_sim_data_chunk_paths(jacobian_tag)[0]
        print(f'Loading the Jacobian: {jac_path}')
        jacobian = read_hdf(jac_path)[jacobian_table][:].astype(np.float32)
        self.jacobian = torch.from_numpy(jacobian).to(device)

        self.lambda_scaling = _grab_param(lambda_scaling)
        if self.lambda_scaling is None:
            self.lambda_scaling = 1

        self.apply_log_scaling = bool(_grab_param(apply_log_scaling, int))
        if self.apply_log_scaling:
            print('Applying log scaling to intensity')

        self.speckle_targeting = _grab_param(speckle_targeting)
        if self.speckle_targeting is not None:
            print(f'Speckle targeting: {self.speckle_targeting*100}%')

        self.radial_weight_mask = None
        apply_radial_weighting = apply_radial_weighting
        if apply_radial_weighting is not None:
            (mask_size, inner_radius, outer_radius,
             max_weight) = apply_radial_weighting.split(',')
            print(f'Applying radial weighting (pixels: {mask_size}, '
                  f'inner radius: {inner_radius}, '
                  f'outer radius: {outer_radius}, max weight: {max_weight})')
            mask_size = int(mask_size)
            inner_radius = float(inner_radius)
            outer_radius = float(outer_radius)
            max_weight = int(max_weight)
            # Create the DH mask
            dh_mask = create_grid_mask(mask_size, inner_radius)
            dh_mask += create_grid_mask(mask_size, outer_radius)
            # Mask out the center pixels inside the inner radius
            dh_mask[dh_mask == 2] = 0
            dh_mask = dh_mask.astype(bool)
            # Create a grid of distances from the center
            distances = np.arange(mask_size) - (mask_size // 2)
            distance_grid = np.sqrt(distances[None, :]**2 +
                                    distances[:, None]**2)
            # A 1D list of all the distances inside the DH
            distances_dh = distance_grid[dh_mask]
            # Distance to the inner/outer radius of the DH from the center
            r_inner = distances_dh.min()
            r_outer = distances_dh.max()
            # Normalize all the DH distances between [0, 1]
            distances_norm = min_max_norm(distance_grid, r_inner - r_outer,
                                          r_outer)
            # Scale all the weights between [1, max_weight]
            rad_weights = 1 + (max_weight - 1) * distances_norm
            # Take just the weights from the DH and put them in a 1D array
            rad_weights = rad_weights[dh_mask]
            self.radial_weight_mask = torch.from_numpy(rad_weights).to(device)

        self.add_residual_stroke_mse = _grab_param(add_residual_stroke_mse)

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
        # Split into the real and imaginary components
        residual_efr, residual_efi = torch.tensor_split(residual_ef, 2, dim=-1)
        # Calculate the residual intensity
        residual_int = residual_efr**2 + residual_efi**2
        if self.apply_log_scaling:
            residual_int = torch.log10(1 + residual_int)
        if self.radial_weight_mask is not None:
            residual_int = residual_int * self.radial_weight_mask
        if self.speckle_targeting:
            k_pixels = int(residual_int.shape[-1] * self.speckle_targeting)
            worst_speckles, _ = torch.topk(residual_int, k=k_pixels, dim=-1)
            loss = self.lambda_scaling * worst_speckles.mean()
        else:
            loss = self.lambda_scaling * residual_int.mean()
        if self.add_residual_stroke_mse is not None:
            stroke_mse = torch.mean(residual_heights**2)
            loss = loss + (self.add_residual_stroke_mse * stroke_mse)
        return loss
