# `dh_t44_30do10` network { 1000 -> 1000 }.
# Trainable parameters: 67,173,352
import torch
import torch.nn as nn

# ----- NEURON SIZES -----
IN_SIZE = 1000
OUT_SIZE = 1000
HIDDEN_SIZE = 2048
BOTTLENECK_SIZE = 512

# ----- LAYER PARAMS -----
NUMBER_OF_RES_BLOCKS = 30
LEAKY_RELU = 0.2
DROPOUT = 0.1


class BottleneckResidualBlock(nn.Module):

    def __init__(self, features, bottleneck_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(features, bottleneck_features, bias=False),
            nn.BatchNorm1d(bottleneck_features),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(DROPOUT),
            nn.Linear(bottleneck_features, features, bias=False),
            nn.BatchNorm1d(features),
        )
        self.final_activation = nn.LeakyReLU(LEAKY_RELU)

    def forward(self, x):
        return self.final_activation(x + self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_SIZE))

    def __init__(self):
        super().__init__()
        self.in_block = nn.Sequential(
            nn.Linear(IN_SIZE, HIDDEN_SIZE, bias=False),
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_RELU),
        )
        self.res_blocks = nn.Sequential(*[
            BottleneckResidualBlock(HIDDEN_SIZE, BOTTLENECK_SIZE)
            for _ in range(NUMBER_OF_RES_BLOCKS)
        ])
        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(HIDDEN_SIZE, OUT_SIZE),
        )

    def forward(self, x):
        x = self.in_block(x)
        x = self.res_blocks(x)
        x = self.out_layer(x)
        return x
