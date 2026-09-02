# `pol_t4_12d_dyn` network { 5312 -> 20 }.
# Trainable parameters: 13,573,652
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# CONFIG OPTIONS
# ==============================================================================
# Number of input neurons
IN_DIM = 5312
# Number of output neurons
OUT_DIM = 20
# Number of neurons expanded out to
OUTER_DIM = 768
# Number of neurons for the bottleneck
INNER_DIM = 512
# Activation slope
LEAKY_RELU = 0.2
# LayerScale starting value
GAMMA_INIT = 1.0
# Number of blocks
DEPTH = 12


# Also known as Stochastic Depth
class DropPath(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_rows = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
        keep_rows = (keep_rows + self.keep_prob).floor_()
        return x.div(self.keep_prob) * keep_rows


class BottleneckResidualBlock(nn.Module):

    def __init__(self, dropout, drop_path_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(OUTER_DIM, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(dropout),
            nn.Linear(INNER_DIM, OUTER_DIM, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((OUTER_DIM, ), GAMMA_INIT))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self, dropout=0.0, dp_max_prob=0.0):
        super().__init__()
        dropout = float(dropout)
        dp_max_prob = float(dp_max_prob)
        print(f'[Network] Dropout: {dropout}')
        print(f'[Network] DP Max Prob: {dp_max_prob}')
        # DropPath probabilities linearly increase from the first to last layer
        dp_probs = np.linspace(0, dp_max_prob, DEPTH)
        self.input = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
        )
        self.blocks = nn.Sequential(*[
            BottleneckResidualBlock(dropout, dp_probs[layer_idx])
            for layer_idx in range(DEPTH)
        ])
        self.output = nn.Linear(OUTER_DIM, OUT_DIM)

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))
