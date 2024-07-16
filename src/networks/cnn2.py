# Convolutional Neural Network 2 (cnn2)
# Trainable parameters: 3,421,463
# Network as described by DOI: 10.1088/2040-8986/ad2256
#   Without the batch normalization

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, 5, padding='same')
        self.conv2 = nn.Conv2d(256, 256, 5, padding='same')
        # 16x16
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(256, 256, 3, padding='same')
        # 4x4
        self.maxpool2 = nn.MaxPool2d(4)
        self.conv4 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv5 = nn.Conv2d(256, 256, 3, padding='same')
        # 1x1
        self.maxpool3 = nn.MaxPool2d(4)
        self.flattened1 = nn.Linear(256, 23)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.flattened1(x)
        return x
