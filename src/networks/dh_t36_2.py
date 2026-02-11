# `dh_t36_2` network { 1000 -> 1000 }.
# Trainable parameters: 25,093,096

import torch
import torch.nn as nn


def _make_dense_block(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=False),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((2, 1000))

    def __init__(self):
        super().__init__()
        self.layer_1 = _make_dense_block(1000, 2048)
        self.layer_2 = _make_dense_block(2048, 2048)
        self.layer_3 = _make_dense_block(2048, 2048)
        self.layer_4 = _make_dense_block(2048, 2048)
        self.layer_5 = _make_dense_block(2048, 2048)
        self.layer_6 = _make_dense_block(2048, 2048)
        self.out_layer = nn.Linear(2048, 1000)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.out_layer(x)
        return x
