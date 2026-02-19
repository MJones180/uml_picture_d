# `dh_t37_8` network { 1000 -> 1000 }.
# Trainable parameters: 8,299,496

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
        self.block1 = create_block(2048, 2048)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.block1(x)
        x = self.out_layer(x)
        return x
