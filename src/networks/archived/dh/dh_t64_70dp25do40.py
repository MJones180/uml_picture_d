# `dh_t64_70dp25do40` network { 1000 -> 1000 }.
# Trainable parameters: 151,411,688
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# CONFIG OPTIONS
# ==============================================================================
# Number of input neurons
IN_DIM = 1000
# Number of output neurons per head
HEAD_OUT_DIM = 500
# Number of neurons expanded out to
OUTER_DIM = 2048
# Number of neurons for the bottleneck
INNER_DIM = 512
# Number of shared blocks
SHARED_DEPTH = 60
# Number of blocks per head
HEAD_DEPTH = 5
# Activation slope
LEAKY_RELU = 0.2
# Dropout rate
DROPOUT = 0.40
# LayerScale starting value
GAMMA_INIT = 1e-4
# DropPath probability of dropping the last layer
DP_MAX_PROB = 0.25
# DropPath probabilities linearly increase from the first to last layer
DP_PROBS = np.linspace(0, DP_MAX_PROB, SHARED_DEPTH + HEAD_DEPTH)


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

    def __init__(self, block_depth_idx):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(OUTER_DIM, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(INNER_DIM, OUTER_DIM, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((OUTER_DIM, ), GAMMA_INIT))
        self.drop_path = DropPath(DP_PROBS[block_depth_idx])

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
        )
        self.shared_blocks = nn.Sequential(*[
            BottleneckResidualBlock(layer_idx)
            for layer_idx in range(SHARED_DEPTH)
        ])
        self.head_1_blocks = nn.Sequential(*[
            BottleneckResidualBlock(SHARED_DEPTH + layer_idx)
            for layer_idx in range(HEAD_DEPTH)
        ])
        self.head_2_blocks = nn.Sequential(*[
            BottleneckResidualBlock(SHARED_DEPTH + layer_idx)
            for layer_idx in range(HEAD_DEPTH)
        ])
        self.head_1_out = nn.Sequential(
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(OUTER_DIM, HEAD_OUT_DIM),
        )
        self.head_2_out = nn.Sequential(
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(OUTER_DIM, HEAD_OUT_DIM),
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.shared_blocks(x)
        x_head_1 = self.head_1_blocks(x)
        x_head_2 = self.head_2_blocks(x)
        return self.head_1_out(x_head_1), self.head_2_out(x_head_2)
