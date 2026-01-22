# `dh_with_rm_3` network { 5312 -> 1512 }.
# Trainable parameters: 20,423,656

import torch
import torch.nn as nn


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 5312))

    def __init__(self):
        super().__init__()
        self.rm_dense_layer = nn.Linear(5312, 1512, bias=False)
        self.relu1 = nn.ReLU()
        self.dense_block1 = nn.Linear(1512, 4096)
        self.relu2 = nn.ReLU()
        self.out_layer = nn.Linear(4096, 1512)

    def forward(self, x):
        x = self.rm_dense_layer(x)
        x = self.relu1(x)
        x = self.dense_block1(x)
        x = self.relu2(x)
        x = self.out_layer(x)
        return x
