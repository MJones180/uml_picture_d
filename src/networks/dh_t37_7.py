# `dh_t37_7` network { 1000 -> 1000 }.
# Trainable parameters: 12,501,992

import torch
import torch.nn as nn


def create_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(0.2),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((2, 1000))

    def __init__(self):
        super().__init__()
        self.in_layer = create_block(1000, 2048)
        self.block1 = create_block(2048, 1024)
        self.block2 = create_block(1024, 2048)
        self.block3 = create_block(2048, 1024)
        self.block4 = create_block(1024, 2048)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out_layer(x)
        return x
