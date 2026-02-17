# `dh_t36_9nd` network { 1000 -> 1000 }.
# Trainable parameters: 12,501,992

import torch
import torch.nn as nn


class BottleneckResidualBlock(nn.Module):

    def __init__(self, features, bottleneck_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, bottleneck_features, bias=False),
            nn.BatchNorm1d(bottleneck_features),
            nn.LeakyReLU(0.2),
            nn.Linear(bottleneck_features, features, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class Network(nn.Module):

    def example_input():
        return torch.rand((2, 1000))

    def __init__(self):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(1000, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
        )
        self.res_block1 = BottleneckResidualBlock(2048, 1024)
        self.res_block2 = BottleneckResidualBlock(2048, 1024)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out_layer(x)
        return x
