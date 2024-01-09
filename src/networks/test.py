import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 31, 31))

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.flattened1 = nn.Linear(5408, 128)
        self.flattened2 = nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
        # Pass data through conv1
        x = self.conv1(x)
        # Use the rectified-linear activation function over x
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        print(x.shape)
        x = self.flattened1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.flattened2(x)

        # Apply softmax to x
        output = F.log_softmax(x, dim=1)
        return output
