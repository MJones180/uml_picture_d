# `dh_t73_h25do10dp10_h45do50dp10_separate_mode1` network { 1000 -> 1000 }.
# Trainable parameters: 77,795,304
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# SHARED CONFIG OPTIONS
# ==============================================================================
IN_DIM = 1000
HEAD_OUT_DIM = 499
OUTER_DIM = 1024
INNER_DIM = 512
LEAKY_RELU = 0.2

# ==============================================================================
# HEAD 1 CONFIG OPTIONS
# ==============================================================================
HEAD_1_DEPTH = 25
HEAD_1_GAMMA_INIT = 1e-2
HEAD_1_DROPOUT = 0.10
HEAD_1_DP_MAX_PROB = 0.10
HEAD_1_DP_PROBS = np.linspace(0, HEAD_1_DP_MAX_PROB, HEAD_1_DEPTH)

# ==============================================================================
# HEAD 2 CONFIG OPTIONS
# ==============================================================================
HEAD_2_DEPTH = 45
HEAD_2_GAMMA_INIT = 1e-5
HEAD_2_DROPOUT = 0.50
HEAD_2_DP_MAX_PROB = 0.10
HEAD_2_DP_PROBS = np.linspace(0, HEAD_2_DP_MAX_PROB, HEAD_2_DEPTH)


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
        self.head_1_mode_1 = nn.Sequential(
            nn.Linear(IN_DIM, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(INNER_DIM, 1),
        )
        self.head_1 = nn.Sequential(
            nn.Linear(IN_DIM + 1, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_1_GAMMA_INIT, HEAD_1_DROPOUT,
                                        HEAD_1_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_1_DEPTH)
            ],
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_1_out = nn.Linear(OUTER_DIM, HEAD_OUT_DIM)
        self.head_2_mode_1 = nn.Sequential(
            nn.Linear(IN_DIM, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(INNER_DIM, 1),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(IN_DIM + 1, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_2_GAMMA_INIT, HEAD_2_DROPOUT,
                                        HEAD_2_DP_PROBS[layer_idx])
                for layer_idx in range(HEAD_2_DEPTH)
            ],
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.head_2_out = nn.Linear(OUTER_DIM, HEAD_OUT_DIM)

    def forward(self, x):
        h1_m1 = self.head_1_mode_1(x)
        h2_m1 = self.head_2_mode_1(x)
        h1_conditioned_input = torch.cat([x, h1_m1], dim=1)
        h2_conditioned_input = torch.cat([x, h2_m1], dim=1)
        h1_rest = self.head_1_out(self.head_1(h1_conditioned_input))
        h2_rest = self.head_2_out(self.head_2(h2_conditioned_input))
        h1_final = torch.cat([h1_m1, h1_rest], dim=1)
        h2_final = torch.cat([h2_m1, h2_rest], dim=1)
        return h1_final, h2_final
