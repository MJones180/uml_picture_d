import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 31, 31))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.conv2 = nn.Conv2d(5, 10, 3)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(10, 20, 3)
        self.conv4 = nn.Conv2d(20, 40, 3)
        self.maxpool2 = nn.MaxPool2d(2)
        self.flattened1 = nn.Linear(640, 500)
        self.dropout1 = nn.Dropout(0.10)
        self.flattened2 = nn.Linear(500, 264)
        self.flattened3 = nn.Linear(264, 23)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.flattened1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.flattened2(x)
        x = F.relu(x)
        x = self.flattened3(x)
        return x
