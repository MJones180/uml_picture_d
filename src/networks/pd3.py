import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 31, 31))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 32, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.flattened1 = nn.Linear(1024, 512)
        self.dropout1 = nn.Dropout(0.10)
        self.flattened2 = nn.Linear(512, 256)
        self.flattened3 = nn.Linear(256, 23)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.flattened1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.flattened2(x)
        x = F.relu(x)
        x = self.flattened3(x)
        return x
