# Dense Fully Connected 2 (dfc2v2)
# Trainable parameters: 3,216,919

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.flattened1 = nn.Linear(1024, 2048)
        self.dropout1 = nn.Dropout(0.4)
        self.flattened2 = nn.Linear(2048, 512)
        self.dropout2 = nn.Dropout(0.4)
        self.flattened3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(0.4)
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
