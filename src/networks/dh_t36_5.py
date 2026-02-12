# `dh_t36_5` network { 1000 -> 1000 }.
# Trainable parameters: 41,772,008

import torch
import torch.nn as nn


def _make_dense_block(in_features, out_features, dropout=0):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(0.2),
        nn.Dropout(dropout),
    )


class ResidualBlock(nn.Module):

    def __init__(self, features, dropout=0.1):
        super().__init__()
        self.block = _make_dense_block(features, features, dropout)

    def forward(self, x):
        return x + self.block(x)


class Network(nn.Module):

    def example_input():
        return torch.rand((2, 1000))

    def __init__(self):
        super().__init__()
        self.in_layer = _make_dense_block(1000, 4096, 0)
        self.res_block1 = ResidualBlock(4096, 0.2)
        self.res_block2 = ResidualBlock(4096, 0.2)
        self.out_layer = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.out_layer(x)
        return x
