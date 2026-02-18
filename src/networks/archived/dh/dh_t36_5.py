# `dh_t36_5` network { 1000 -> 1000 }.
# Trainable parameters: 20,894,696

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, features, dropout):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, features, bias=False),
            nn.Dropout(dropout),
            nn.BatchNorm1d(features),
            nn.LeakyReLU(0.2),
            nn.Linear(features, features, bias=False),
            nn.Dropout(dropout),
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
        self.res_block1 = ResidualBlock(2048, 0.1)
        self.res_block2 = ResidualBlock(2048, 0.1)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out_layer(x)
        return x
