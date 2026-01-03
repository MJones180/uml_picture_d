# `llowfs_cnn_3` network [32x32].
# Trainable parameters: 1,014,887

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
        nn.LeakyReLU(),
    )


# Performs in-block downsizing
def _make_conv_block_and_downsize(in_features, out_features, kernel_size):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding=1,
            bias=False,
            stride=2,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


def _make_dense_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 1, 32, 32))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _make_conv_block(1, 16, 3)
        self.conv_block2 = _make_conv_block(16, 16, 3)
        # 32x32 -> 16x16
        self.conv_block3 = _make_conv_block_and_downsize(16, 32, 3)
        self.conv_block4 = _make_conv_block(32, 32, 3)
        # 16x16 -> 8x8
        self.conv_block5 = _make_conv_block_and_downsize(32, 64, 3)
        self.conv_block6 = _make_conv_block(64, 64, 3)
        # 8x8 -> 4x4
        self.conv_block7 = _make_conv_block_and_downsize(64, 128, 3)
        self.conv_block8 = _make_conv_block(128, 256, 3)
        # 4x4 -> 1x1
        self.maxpool1 = nn.MaxPool2d(4)
        self.dense_block1 = _make_dense_block(256, 2048, 0.3)
        self.out_layer = nn.Linear(2048, 23)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        # Downsize
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        # Downsize
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        # Downsize
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.maxpool1(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
