# `dh_t39_3` network { 1000 -> 1000 }.
# Trainable parameters: 8,309,736
import torch
import torch.nn as nn

# ----- NEURON SIZES -----
IN_SIZE = 1000
OUT_SIZE = 1000
HIDDEN_SIZE = 2048
BOTTLENECK_SIZE = 512

# ----- LAYER PARAMS -----
LEAKY_RELU = 0.2
DROPOUT = 0.1


class BottleneckResidualBlock(nn.Module):

    def __init__(self, features, bottleneck_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(features, bottleneck_features, bias=False),
            nn.BatchNorm1d(bottleneck_features),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(bottleneck_features, features),
        )

    def forward(self, x):
        return x + self.block(x)


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_SIZE))

    def __init__(self):
        super().__init__()

        self.in_layer = nn.Linear(IN_SIZE, HIDDEN_SIZE)
        self.res_block1 = BottleneckResidualBlock(HIDDEN_SIZE, BOTTLENECK_SIZE)
        self.res_block2 = BottleneckResidualBlock(HIDDEN_SIZE, BOTTLENECK_SIZE)
        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(HIDDEN_SIZE, OUT_SIZE),
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out_layer(x)
        return x
