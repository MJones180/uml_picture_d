import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 31, 31))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.dropout1 = nn.Dropout(0.10)
        self.flattened1 = nn.Linear(2704, 128)
        self.flattened2 = nn.Linear(128, 23)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = self.flattened1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.flattened2(x)
        return x
