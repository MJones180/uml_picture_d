# Convolutional Neural Network V2 9 (cnn_v2_9)
# Trainable parameters: 2,415,287
# Based on `cnn_v2_5` with an additional dense layer at the end

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
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _make_conv_block(1, 32, 5)
        self.conv_block2 = _make_conv_block(32, 32, 5)
        # 32x32 -> 16x16
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv_block3 = _make_conv_block(32, 64, 5)
        self.conv_block4 = _make_conv_block(64, 64, 5)
        # 16x16 -> 8x8
        self.maxpool2 = nn.MaxPool2d(2)
        self.dense_block1 = _make_dense_block(4096, 512, 0.3)
        self.dense_block2 = _make_dense_block(512, 256, 0.3)
        self.out_layer = nn.Linear(256, 23)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.maxpool1(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = self.out_layer(x)
        return x
