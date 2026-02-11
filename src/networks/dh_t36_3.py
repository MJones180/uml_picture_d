# `dh_t36_3` network { 1000 -> 1000 }.
# Trainable parameters: 58,557,416

import torch
import torch.nn as nn


def _make_dense_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((2, 1000))

    def __init__(self):
        super().__init__()
        self.layer_1 = _make_dense_block(1000, 4096)
        self.layer_2 = _make_dense_block(4096, 4096)
        self.layer_3 = _make_dense_block(4096, 4096)
        self.layer_4 = _make_dense_block(4096, 4096)
        self.out_layer = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.out_layer(x)
        return x
