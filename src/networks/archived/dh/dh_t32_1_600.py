# `dh_t32_1_600` network { 2x59x59 -> 600 }. (Based on `dh_t31_600`)
# Trainable parameters: 136,509,528

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv_block(in_features, out_features, kernel_size, return_relu=True):
    layers = [
        nn.Conv2d(
            in_features,
            out_features,
            kernel_size,
            padding='same',
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
    ]
    if return_relu:
        layers.append(nn.LeakyReLU())
    return nn.Sequential(*layers)


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


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv_block1 = _make_conv_block(channels, channels, 3)
        self.conv_block2 = _make_conv_block(channels, channels, 3, False)

    def forward(self, x):
        # Add the skip connection
        identity = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += identity
        return F.leaky_relu(out)


def _make_dense_block(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU())


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()
        self.initial_conv_block = _make_conv_block(2, 128, 3)
        self.conv_residual_1 = ResidualBlock(128)
        # 59x59 -> 30x30
        self.conv_downsize_1 = _make_conv_block_and_downsize(128, 256, 3)
        self.conv_residual_2 = ResidualBlock(256)
        # 30x30 -> 15x15
        self.conv_downsize_2 = _make_conv_block_and_downsize(256, 512, 3)
        self.conv_residual_3 = ResidualBlock(512)
        # 15x15 -> 8x8
        self.conv_downsize_3 = _make_conv_block_and_downsize(512, 1024, 3)
        self.conv_residual_4 = ResidualBlock(1024)
        # 8x8 -> 4x4
        self.conv_downsize_4 = _make_conv_block_and_downsize(1024, 2048, 3)
        self.conv_residual_5 = ResidualBlock(2048)
        # 4x4 -> 1x1
        self.avgpool = nn.AvgPool2d(4)
        self.dense_block = _make_dense_block(2048, 4096)
        self.out_layer = nn.Linear(4096, 600)
        self.out_layer_activation = nn.Tanh()

    def forward(self, x):
        x = self.initial_conv_block(x)
        x = self.conv_residual_1(x)
        x = self.conv_downsize_1(x)
        x = self.conv_residual_2(x)
        x = self.conv_downsize_2(x)
        x = self.conv_residual_3(x)
        x = self.conv_downsize_3(x)
        x = self.conv_residual_4(x)
        x = self.conv_downsize_4(x)
        x = self.conv_residual_5(x)
        x = self.avgpool(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block(x)
        x = self.out_layer(x)
        x = self.out_layer_activation(x)
        return x
