# Dense Fully Connected Double 2 (dfcd2)
# Trainable parameters: 334,103

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.flattened1 = nn.Linear(1024, 256)
        self.flattened2 = nn.Linear(256, 256)
        self.flattened3 = nn.Linear(256, 23)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = F.relu(self.flattened2(x))
        x = self.flattened3(x)
        return x
