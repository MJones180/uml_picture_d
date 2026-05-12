# `dh_t70_h35do40dp5` network { 1000 -> 1000 }.
# Trainable parameters: 40,032,232
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# SHARED CONFIG OPTIONS
# ==============================================================================
# Number of input neurons
IN_DIM = 1000
# Number of output neurons per head
HEAD_OUT_DIM = 500
# Number of neurons expanded out to
OUTER_DIM = 1024
# Number of neurons for the bottleneck
INNER_DIM = 256
# Number of blocks
HEAD_DEPTH = 35
# Activation slope
LEAKY_RELU = 0.2
# LayerScale starting value
HEAD_GAMMA_INIT = 1e-4
# Dropout rate
HEAD_DROPOUT = 0.40
# DropPath probability of dropping the last layer
HEAD_DP_MAX_PROB = 0.05
# DropPath probabilities linearly increase from the first to last layer
HEAD_DP_PROBS = np.linspace(0, HEAD_DP_MAX_PROB, HEAD_DEPTH)


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

    def __init__(self, gamma_init, dropout, drop_path_prob):
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
        self.gamma = nn.Parameter(torch.full((OUTER_DIM, ), gamma_init))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self):
        super().__init__()
        self.head_1 = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_GAMMA_INIT, HEAD_DROPOUT,
                                        HEAD_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_DEPTH)
            ],
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_2 = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_GAMMA_INIT, HEAD_DROPOUT,
                                        HEAD_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_DEPTH)
            ],
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_1_out = nn.Linear(OUTER_DIM, HEAD_OUT_DIM)
        self.head_2_out = nn.Linear(OUTER_DIM, HEAD_OUT_DIM)

    def forward(self, x):
        return self.head_1_out(self.head_1(x)), self.head_2_out(self.head_2(x))
