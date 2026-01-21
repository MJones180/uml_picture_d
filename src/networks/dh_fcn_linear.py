# `dh_fcn_linear` network { 5312 -> 1512 }.
# Trainable parameters: 27,956,712

import torch
import torch.nn as nn


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 5312))

    def __init__(self):
        super().__init__()
        self.dense_block = nn.Linear(5312, 4096)
        self.out_layer = nn.Linear(4096, 1512)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.out_layer(x)
        return x
