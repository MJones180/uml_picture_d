# `dh_t3` network { 2x59x59 -> 1512 }.
# Trainable parameters: 7,391,976

import torch
import torch.nn as nn


def _make_conv_block(in_features, out_features, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding='same',
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
        nn.ReLU(),
    )


def _make_dense_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _make_conv_block(2, 16, 3)
        # 59x59 -> 30x30
        self.maxpool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv_block2 = _make_conv_block(16, 32, 3)
        # 30x30 -> 15x15
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv_block3 = _make_conv_block(32, 64, 3)
        # 15x15 -> 8x8
        self.maxpool3 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv_block4 = _make_conv_block(64, 128, 3)
        # 8x8 -> 4x4
        self.maxpool4 = nn.MaxPool2d(2)
        self.dense_block1 = _make_dense_block(2048, 2048, 0.3)
        self.out_layer = nn.Linear(2048, 1512)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.maxpool1(x)
        x = self.conv_block2(x)
        x = self.maxpool2(x)
        x = self.conv_block3(x)
        x = self.maxpool3(x)
        x = self.conv_block4(x)
        x = self.maxpool4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
