# `dh_fcn_linear` network { 5312 -> 1512 }.
# Trainable parameters: 8,031,744

import torch
import torch.nn as nn


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 5312))

    def __init__(self):
        super().__init__()
        self.dense_layer = nn.Linear(5312, 1512, bias=False)

    def forward(self, x):
        return self.dense_layer(x)
