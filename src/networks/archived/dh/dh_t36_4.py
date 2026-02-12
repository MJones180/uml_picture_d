# `dh_t36_4` network { 1000 -> 1000 }.
# Trainable parameters: 37,688,296

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
        self.in_layer = _make_dense_block(1000, 2048, 0)
        self.res_block1 = ResidualBlock(2048, 0.2)
        self.res_block2 = ResidualBlock(2048, 0.2)
        self.res_block3 = ResidualBlock(2048, 0.2)
        self.res_block4 = ResidualBlock(2048, 0.2)
        self.res_block5 = ResidualBlock(2048, 0.2)
        self.res_block6 = ResidualBlock(2048, 0.2)
        self.res_block7 = ResidualBlock(2048, 0.2)
        self.res_block8 = ResidualBlock(2048, 0.2)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.out_layer(x)
        return x
