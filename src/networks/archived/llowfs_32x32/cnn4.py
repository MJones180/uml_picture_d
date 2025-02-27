# Convolutional Neural Network 4 (cnn4)
# Trainable parameters: 417,079

import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding='same')
        self.conv2 = nn.Conv2d(32, 32, 5, padding='same')
        # 32x32 -> 16x16
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        # 16x16 -> 8x8
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv5 = nn.Conv2d(64, 64, 3, padding='same')
        # 8x8 -> 4x4
        self.maxpool3 = nn.MaxPool2d(2)
        self.flattened1 = nn.Linear(1024, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.flattened2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.flattened3 = nn.Linear(128, 23)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.maxpool2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.flattened1(x))
        x = self.dropout1(x)
        x = F.relu(self.flattened2(x))
        x = self.dropout2(x)
        x = self.flattened3(x)
        return x
