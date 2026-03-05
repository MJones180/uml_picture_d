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
        """
        super().__init__()

        def _grab_param(arg, desired_type):
            if arg is not None:
                return desired_type(arg)
            return None

        # The conditional block declaration of `output_weights` gives the weight
        # of each output neuron associated with the first DM
        linear_scaling = _grab_param(take_row_sum, bool)
        exp_scaling = _grab_param(take_row_sum, bool)
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
        self.apply_modified_log = _grab_param(apply_modified_log, float)
        if self.apply_modified_log:
            print('Will apply a modified log before calculating the '
                  f'loss ({self.apply_modified_log} scaling)')

        # Whether a modified exp should be applied before determining the loss
        self.apply_modified_exp = _grab_param(apply_modified_exp, float)
        if self.apply_modified_exp:
            print('Will apply a modified exp before calculating the '
                  f'loss ({self.apply_modified_exp} base)')

        # Whether a combination of MAE and MSE loss should be used
        self.add_mae_loss = _grab_param(add_mae_loss, float)
        if self.add_mae_loss:
            print(f'Will add {self.add_mae_loss} MAE to the loss')

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
        difference = model_outputs - truth_outputs
        loss_diff = difference**2
        if self.add_mae_loss:
            loss_diff = loss_diff + self.add_mae_loss * torch.abs(difference)
        loss = self.output_weights * loss_diff
        if self.take_row_sum:
            loss = loss.sum(axis=1)
        return loss.mean()
