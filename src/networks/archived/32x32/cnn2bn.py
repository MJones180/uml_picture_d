# Convolutional Neural Network 2 Batch Normalization (cnn2bn)
# Trainable parameters: 3,424,023
# Network as described by DOI: 10.1088/2040-8986/ad2256

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        # Need 2 inputs due to batch norm
        return torch.rand((2, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, 5, padding='same')
        self.batch_norm1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, 5, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(256)
        # 16x16
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(256, 256, 3, padding='same')
        self.batch_norm3 = nn.BatchNorm2d(256)
        # 4x4
        self.maxpool2 = nn.MaxPool2d(4)
        self.conv4 = nn.Conv2d(256, 256, 3, padding='same')
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, padding='same')
        self.batch_norm5 = nn.BatchNorm2d(256)
        # 1x1
        self.maxpool3 = nn.MaxPool2d(4)
        self.flattened1 = nn.Linear(256, 23)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.maxpool1(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.maxpool2(x)
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.flattened1(x)
        return x
