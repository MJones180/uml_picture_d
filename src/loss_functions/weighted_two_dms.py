import numpy as np
import torch
import torch.nn as nn
from utils.terminate_with_message import terminate_with_message


class WeightedTwoDMs(nn.Module):

    def __init__(
        self,
        outputs_per_dm,
        device,
        linear_scaling=None,
        exp_scaling=None,
        linear_scaling_tail=None,
        take_row_sum=None,
        apply_modified_log=None,
        apply_modified_exp=None,
        add_mae_loss=None,
        add_exp_loss_weighting=None,
        multiheaded_output=None,
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
        else:
            terminate_with_message('Unknown loss scaling')

        if _grab_param(multiheaded_output, bool):
            print('Loss function set for multi-headed output')
        else:
            # Create a copy of the output weights for the second DM
            output_weights = np.tile(output_weights, 2)

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
        # The loss weighted with respect to each given output neuron
        weighted_loss = self.output_weights * loss
        if self.take_row_sum:
            weighted_loss = weighted_loss.sum(axis=1)
        return weighted_loss.mean()
