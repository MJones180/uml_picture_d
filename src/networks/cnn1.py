# Convolutional Neural Network 1 (cnn1)
# Trainable parameters: 285,367
# Network as described by DOI: 10.1109/CHILECON54041.2021.9703060

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5, padding='same')
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding='same')
        self.maxpool2 = nn.MaxPool2d(2)
        self.flattened1 = nn.Linear(2048, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.flattened2 = nn.Linear(128, 23)

    def forward(self, x):
        x = F.relu(self.maxpool1(self.conv1(x)))
        x = F.relu(self.maxpool2(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = self.dropout1(x)
        x = self.flattened2(x)
        return x
