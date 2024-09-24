# Dense Fully Connected Single 1 (dfcs1)
# Trainable parameters: 536,599

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.flattened1 = nn.Linear(1024, 512)
        self.flattened2 = nn.Linear(512, 23)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = self.flattened2(x)
        return x
