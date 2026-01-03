# `dh_fcn_1` network { 5312 -> 1512 }.
# Trainable parameters: 27,956,712

import torch
import torch.nn as nn


def _make_dense_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 5312))

    def __init__(self):
        super().__init__()
        self.dense_block1 = _make_dense_block(5312, 4096, 0.3)
        self.out_layer = nn.Linear(4096, 1512)

    def forward(self, x):
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
