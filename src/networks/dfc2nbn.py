# Dense Fully Connected 2 Batch Norm (dfc2nbn)
# Trainable parameters: 3,222,295

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        # Need 2 inputs due to batch norm
        return torch.rand((2, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.flattened1 = nn.Linear(1024, 2048)
        self.batch_norm1 = nn.BatchNorm1d(2048)
        self.flattened2 = nn.Linear(2048, 512)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.flattened3 = nn.Linear(512, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.flattened4 = nn.Linear(128, 23)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.batch_norm1(self.flattened1(x)))
        x = F.relu(self.batch_norm2(self.flattened2(x)))
        x = F.relu(self.batch_norm3(self.flattened3(x)))
        x = self.flattened4(x)
        return x
