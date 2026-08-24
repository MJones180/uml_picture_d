import torch


class LandmanRMS(torch.nn.Module):

    def __init__(self, epsilon=None):
        """The LandmanRMS class.

        Parameters
        ----------
        epsilon : float
            The epsilon to add to the denominator.

        Notes
        -----
        RMS loss function based on 2024 Landman paper (Making the unmodulated
        Pyramid wavefront sensor smart).
        """
        super().__init__()

        self.epsilon = epsilon
        if self.epsilon is None:
            self.epsilon = 1
        else:
            self.epsilon = float(epsilon)
        print(f'Epsilon: {self.epsilon}')

    def forward(self, model_outputs, truth_outputs):
        # RSS of difference along features
        numer = torch.sum((truth_outputs - model_outputs)**2, dim=-1).sqrt()
        # RSS of truth along features
        denom = torch.sum(truth_outputs**2, dim=-1).sqrt() + self.epsilon
        # Calculate the loss for each row
        loss_per_row = numer / denom
        # Calculate average loss in batch
        return loss_per_row.mean()
