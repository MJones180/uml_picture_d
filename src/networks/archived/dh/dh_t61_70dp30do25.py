# `dh_t61_70dp30do25` network { 1000 -> 1000 }.
# Trainable parameters: 151,407,592
import numpy as np
import torch
import torch.nn as nn

# ----- NEURON SIZES -----
IN_SIZE = 1000
OUT_SIZE = 1000
HIDDEN_SIZE = 2048
BOTTLENECK_SIZE = 512

# ----- LAYER PARAMS -----
NUMBER_OF_RES_BLOCKS = 70
LEAKY_RELU = 0.2
DROPOUT = 0.25
GAMMA_INIT = 1.0

# ----- DROP PATH -----
MAX_DROP_PATH = 0.3
DP_RATES = np.linspace(0, MAX_DROP_PATH, NUMBER_OF_RES_BLOCKS)


class DropPath(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        # This function drops random rows from the batch in a given layer;
        # this way, no layer is completely ignored during an update
        if self.drop_prob == 0 or not self.training:
            return x
        # Give each row in the batch a random value between [0, 1]
        keep_rows = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
        # Keep rows with a value of 1, drop rows with a value of 0
        keep_rows = (keep_rows + self.keep_prob).floor_()
        # Need to scale by the probability of keeping the row to maintain
        # the variance; mask out the rows which are not kept
        return x.div(self.keep_prob) * keep_rows


class BottleneckResidualBlock(nn.Module):

    def __init__(self, features, bottleneck_features, drop_path_prob=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(features, bottleneck_features, bias=False),
            nn.BatchNorm1d(bottleneck_features),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(bottleneck_features, features, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((features, ), GAMMA_INIT))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_SIZE))

    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(IN_SIZE, HIDDEN_SIZE, bias=False),
            nn.BatchNorm1d(HIDDEN_SIZE),
        )
        self.res_blocks = nn.Sequential(*[
            BottleneckResidualBlock(HIDDEN_SIZE, BOTTLENECK_SIZE, DP_RATES[i])
            for i in range(NUMBER_OF_RES_BLOCKS)
        ])
        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(HIDDEN_SIZE, OUT_SIZE),
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_blocks(x)
        x = self.out_layer(x)
        return x
