# Dense Fully Connected 1 (dfc1)
# Trainable parameters: 691,991

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.flattened1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.flattened2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.2)
        self.flattened3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.flattened4 = nn.Linear(128, 23)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = self.dropout1(x)
        x = F.relu(self.flattened2(x))
        x = self.dropout2(x)
        x = F.relu(self.flattened3(x))
        x = self.dropout3(x)
        x = self.flattened4(x)
        return x
