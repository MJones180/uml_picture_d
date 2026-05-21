# `dh_t77_h25do10dp10_h45do40dp30` network { 1000 -> 1000 }.
# Trainable parameters: 64,552,168
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
# Number of neurons for the bottleneck
INNER_DIM = 512
# Activation slope
LEAKY_RELU = 0.2

# ==============================================================================
# HEAD 1 CONFIG OPTIONS
# ==============================================================================
# Number of neurons expanded out to
HEAD_1_OUTER_DIM = 1024
# Number of blocks
HEAD_1_DEPTH = 25
# LayerScale starting value
HEAD_1_GAMMA_INIT = 1e-2
# Dropout rate
HEAD_1_DROPOUT = 0.10
# DropPath probability of dropping the last layer
HEAD_1_DP_MAX_PROB = 0.10
# DropPath probabilities linearly increase from the first to last layer
HEAD_1_DP_PROBS = np.linspace(0, HEAD_1_DP_MAX_PROB, HEAD_1_DEPTH)

# ==============================================================================
# HEAD 2 CONFIG OPTIONS
# ==============================================================================
# Number of neurons expanded out to
HEAD_2_OUTER_DIM = 768
# Number of blocks
HEAD_2_DEPTH = 45
# LayerScale starting value
HEAD_2_GAMMA_INIT = 1e-2
# Dropout rate
HEAD_2_DROPOUT = 0.40
# DropPath probability of dropping the last layer
HEAD_2_DP_MAX_PROB = 0.30
# DropPath probabilities linearly increase from the first to last layer
HEAD_2_DP_PROBS = np.linspace(0, HEAD_2_DP_MAX_PROB, HEAD_2_DEPTH)


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

    def __init__(self, outer_dim, gamma_init, dropout, drop_path_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(outer_dim),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(outer_dim, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(dropout),
            nn.Linear(INNER_DIM, outer_dim, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((outer_dim, ), gamma_init))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self):
        super().__init__()
        self.head_1 = nn.Sequential(
            nn.Linear(IN_DIM, HEAD_1_OUTER_DIM, bias=False),
            nn.BatchNorm1d(HEAD_1_OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_1_OUTER_DIM, HEAD_1_GAMMA_INIT,
                                        HEAD_1_DROPOUT,
                                        HEAD_1_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_1_DEPTH)
            ],
            nn.BatchNorm1d(HEAD_1_OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_2 = nn.Sequential(
            nn.Linear(IN_DIM, HEAD_2_OUTER_DIM, bias=False),
            nn.BatchNorm1d(HEAD_2_OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_2_OUTER_DIM, HEAD_2_GAMMA_INIT,
                                        HEAD_2_DROPOUT,
                                        HEAD_2_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_2_DEPTH)
            ],
            nn.BatchNorm1d(HEAD_2_OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_1_out = nn.Linear(HEAD_1_OUTER_DIM, HEAD_OUT_DIM)
        self.head_2_out = nn.Linear(HEAD_2_OUTER_DIM, HEAD_OUT_DIM)

    def forward(self, x):
        return self.head_1_out(self.head_1(x)), self.head_2_out(self.head_2(x))
