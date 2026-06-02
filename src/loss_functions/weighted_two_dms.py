import numpy as np
import torch
import torch.nn as nn
from utils.constants import DATA_F, RAW_DATA_P
from utils.hdf_read_and_write import read_hdf
from utils.norm import min_max_norm
from utils.terminate_with_message import terminate_with_message


class WeightedTwoDMs(nn.Module):

    def __init__(
        self,
        outputs_per_dm,
        device,
        linear_scaling=None,
        exp_scaling=None,
        singular_value_scaling=None,
        singular_value_scaling_square=None,
        singular_value_scaling_sqrt=None,
        singular_value_scaling_lower_bound=None,
        singular_value_scaling_no_norm_or_scaling=None,
        linear_scaling_tail=None,
        take_row_sum=None,
        apply_modified_log=None,
        apply_modified_exp=None,
        add_mae_loss=None,
        add_exp_loss_weighting=None,
        multiheaded_output=None,
        dynamic_linear_weights=None,
    ):
        """The WeightedTwoDMs class.

        Parameters
        ----------
        outputs_per_dm : int
            Number of output neurons for each DM (should be two in total).
        device : str
            The torch device to use.
        linear_scaling : float
            Lower bound, x, that the loss should be scaled to. Each DM will
            have losses ranging from [1, x].
        exp_scaling : float
            Base that the loss should be scaled with, the exponent is the
            index of the output value.
        singular_value_scaling : str
            Use singular values as the scaling for each value. The argument
            passed should be the tag of the raw HDF data which contains the
            singular values. The singular values must be stored under the
            table name of `singular_values`.
        singular_value_scaling_square : bool
            Square the singular values.
        singular_value_scaling_sqrt : bool
            Square root the singular values.
        singular_value_scaling_lower_bound : float
            The lower bound to scale the singular values between; defaults to
            a lower-bound of 0.1. Without scaling, the singular values will
            range from [~0, 1] (after min-max norm).
        singular_value_scaling_no_norm_or_scaling : bool
            Do not apply min-max normalization to the singular values and do not
            scale them between [lower bound, 1].
        linear_scaling_tail : int
            Used with the `linear_scaling` argument to change the lower mode.
        take_row_sum : bool
            Take the sum across each row instead of the mean.
        apply_modified_log : float
            Apply a modified log the outputs before the loss is calculated.
        apply_modified_exp : float
            Apply a modified exp the outputs before the loss is calculated.
        add_mae_loss : float
            Add an MAE loss, scaled by alpha, to the MSE loss.
        add_exp_loss_weighting : str
            Add an exp weighting to the loss - boosts the gradient for
            small errors. String of format 'alpha,beta' expected.
        multiheaded_output : bool
            Set to True if the architecture has a multi-headed output; this
            makes it so that the loss function only operates on a single DM's
            output values. In effect, the internal weighting is not tiled
            to account for two separate DMs. Therefore, this loss function is
            instead called twice (once for each DM).
        dynamic_linear_weights : int
            Apply dynamic linear weighting which depends on the current epoch;
            each output coefficient will go from having an equal weight on the
            first epoch to the desired weighting on the final epoch; the passed
            value should specify the total number of epochs.

        Notes
        -----
        Given delta (model_outputs - truth_outputs):
        MSE:
            - Loss: delta**2
            - Gradient: 2 * delta
        MSE with exp loss weighting:
            - Loss: delta**2 + alpha * delta**2 * e**(-delta/beta)
            - Gradient: 2 * delta + alpha * e**(-delta/beta)
                                    * (2 * delta - delta**2/beta)
        """
        super().__init__()

        def _grab_param(arg, desired_type=float):
            if arg is not None:
                return desired_type(arg)
            return None

        # The conditional block declaration of `output_weights` gives the weight
        # of each output neuron associated with the first DM
        linear_scaling = _grab_param(linear_scaling)
        exp_scaling = _grab_param(exp_scaling)
        singular_value_scaling = _grab_param(singular_value_scaling, str)
        if linear_scaling:
            end_mode = outputs_per_dm
            linear_scaling_tail = _grab_param(linear_scaling_tail, int)
            if linear_scaling_tail:
                end_mode = linear_scaling_tail
            print('Linear scaling from 1 (mode 0) to '
                  f'{linear_scaling} (mode {end_mode - 1})')
            output_weights = np.linspace(1, linear_scaling, end_mode)
            if linear_scaling_tail:
                print(f'Constant scaling of {linear_scaling} from modes '
                      f'{end_mode} to {outputs_per_dm - 1}')
                output_weights_full = np.full(outputs_per_dm, linear_scaling)
                output_weights_full[:end_mode] = output_weights
                output_weights = output_weights_full
        elif exp_scaling:
            print(f'Exponential scaling with base {exp_scaling}')
            output_weights = exp_scaling**np.arange(outputs_per_dm)
        elif singular_value_scaling:
            print(f'Singular value scaling ({singular_value_scaling})')
            path = f'{RAW_DATA_P}/{singular_value_scaling}/0_{DATA_F}'
            print(f'Path: {path}')
            # Load in the singular values
            singular_values = read_hdf(path)['singular_values'][:]
            # Use the required number of modes
            singular_values = singular_values[:outputs_per_dm]
            if _grab_param(singular_value_scaling_square, bool):
                print('Squaring the singular values')
                singular_values = singular_values**2
            if _grab_param(singular_value_scaling_sqrt, bool):
                print('Square rooting the singular values')
                singular_values = singular_values**0.5
            if _grab_param(singular_value_scaling_no_norm_or_scaling, bool):
                print('Not doing min-max norm or scaling')
                output_weights = singular_values
            else:
                # Normalize the singular values to have a min-max of [1,0]
                singular_values = min_max_norm(
                    singular_values,
                    singular_values[0] - singular_values[-1],
                    singular_values[-1],
                )
                # Grab the lower bound to scale the values between
                low_bound = _grab_param(singular_value_scaling_lower_bound)
                if low_bound is None:
                    low_bound = 0.1
                print(f'Setting lower bound to {low_bound}')
                # The singular values scaled between [1, low_bound]
                output_weights = low_bound + (1 - low_bound) * singular_values
        else:
            terminate_with_message('Unknown loss scaling')

        if _grab_param(multiheaded_output, bool):
            print('Loss function set for multi-headed output')
        else:
            # Create a copy of the output weights for the second DM
            output_weights = np.tile(output_weights, 2)

        # Keep track of the current epoch incase it is needed
        self.current_epoch = None

        self.dynamic_linear_weights = _grab_param(dynamic_linear_weights, int)
        if self.dynamic_linear_weights:
            self.total_epochs = self.dynamic_linear_weights
            self.dynamic_linear_weights = True
            self.equal_output_weights = torch.from_numpy(
                np.ones_like(output_weights)).to(device)

        # Mean division is done for each epoch when using dynamic weighting
        if not self.dynamic_linear_weights:
            # Normalize the weights to have a mean of 1
            output_weights = output_weights / output_weights.mean()
        # Move the output weights to torch
        self.output_weights = torch.from_numpy(output_weights).to(device)

        # Whether the sum or mean should be taken across each row
        self.take_row_sum = _grab_param(take_row_sum, bool)
        if self.take_row_sum:
            print('Will take the sum across each row')
        else:
            print('Will take the mean across each row')

        # Whether a modified log should be applied before determining the loss
        self.apply_modified_log = _grab_param(apply_modified_log)
        if self.apply_modified_log:
            print('Will apply a modified log before calculating the '
                  f'loss ({self.apply_modified_log} scaling)')

        # Whether a modified exp should be applied before determining the loss
        self.apply_modified_exp = _grab_param(apply_modified_exp)
        if self.apply_modified_exp:
            print('Will apply a modified exp before calculating the '
                  f'loss ({self.apply_modified_exp} base)')

        # Whether a combination of MAE and MSE loss should be used
        self.add_mae_loss = _grab_param(add_mae_loss)
        if self.add_mae_loss:
            print(f'Will add {self.add_mae_loss} MAE to the loss')

        # Whether to add exponential weighting to the loss
        self.add_exp_loss_weighting = _grab_param(add_exp_loss_weighting, str)
        if self.add_exp_loss_weighting:
            alpha, beta = self.add_exp_loss_weighting.split(',')
            self.add_exp_loss_weighting_alpha = float(alpha)
            self.add_exp_loss_weighting_beta = float(beta)
            print('Will add exp weighting to the loss '
                  f'(alpha = {self.add_exp_loss_weighting_alpha}, '
                  f'beta = {self.add_exp_loss_weighting_beta})')

    def _apply_log_transformation(self, values):
        alpha = self.apply_modified_log
        return (torch.sign(values) *
                torch.log10(1 + torch.abs(values) / alpha))

    def _apply_exp_transformation(self, values):
        base = self.apply_modified_exp
        return torch.sign(values) * (1 - base**torch.abs(values))

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, model_outputs, truth_outputs):
        if self.apply_modified_log:
            model_outputs = self._apply_log_transformation(model_outputs)
            truth_outputs = self._apply_log_transformation(truth_outputs)
        if self.apply_modified_exp:
            model_outputs = self._apply_exp_transformation(model_outputs)
            truth_outputs = self._apply_exp_transformation(truth_outputs)
        # While the absolute value is not needed for just MSE, it may be needed
        # for other things, so it is computed now incase it's needed after
        abs_delta = torch.abs(model_outputs - truth_outputs)
        loss = abs_delta**2
        if self.add_mae_loss:
            loss = loss + self.add_mae_loss * abs_delta
        # Apply the exp loss weighting - makes small errors more important
        if self.add_exp_loss_weighting:
            loss_weights = 1.0 + (
                self.add_exp_loss_weighting_alpha *
                torch.exp(-abs_delta / self.add_exp_loss_weighting_beta))
            loss = loss * loss_weights
        if self.dynamic_linear_weights:
            current_ratio = self.current_epoch / self.total_epochs
            # Cap the ratio so that the first term does not go above 1 scaling
            if current_ratio > 1:
                current_ratio = 1
            term_one = current_ratio * self.output_weights
            term_two = (1 - current_ratio) * self.equal_output_weights
            dynamic_weights = term_one + term_two
            # Normalize the weights to have a mean of 1
            dynamic_weights = dynamic_weights / dynamic_weights.mean()
            # The loss weighted with respect to each given output neuron
            weighted_loss = dynamic_weights * loss
        else:
            # The loss weighted with respect to each given output neuron
            weighted_loss = self.output_weights * loss
        if self.take_row_sum:
            weighted_loss = weighted_loss.sum(axis=1)
        return weighted_loss.mean()
