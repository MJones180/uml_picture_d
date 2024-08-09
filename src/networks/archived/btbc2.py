# Back to Back Convolution 2 (btbc2)
# Trainable parameters: 981,335

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.dropout2 = nn.Dropout(0.1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.dropout4 = nn.Dropout(0.1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.flattened1 = nn.Linear(1600, 512)
        self.dropout5 = nn.Dropout(0.4)
        self.flattened2 = nn.Linear(512, 256)
        self.dropout6 = nn.Dropout(0.4)
        self.flattened3 = nn.Linear(256, 23)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = self.dropout5(x)
        x = F.relu(self.flattened2(x))
        x = self.dropout6(x)
        x = self.flattened3(x)
        return x
