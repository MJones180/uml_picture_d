# `dh_t33_5_600` network { 2x59x59 -> 600 }. (Based on `dh_t32_8_600`)
# Trainable parameters: 86,219,096

import torch
import torch.nn as nn


def _make_conv_block(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            3,
            padding='same',
            bias=False,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


def _make_conv_block_and_downsize(in_features, out_features):
    return nn.Sequential(
        nn.Conv2d(
            in_features,
            out_features,
            3,
            padding=1,
            bias=False,
            stride=2,
        ),
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


def _make_dense_block(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU())


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()
        self.initial_conv_block = _make_conv_block(2, 128)
        self.conv_layer_1 = _make_conv_block(128, 128)
        # 59x59 -> 30x30
        self.conv_downsize_1 = _make_conv_block_and_downsize(128, 256)
        self.conv_layer_2 = _make_conv_block(256, 256)
        # 30x30 -> 15x15
        self.conv_downsize_2 = _make_conv_block_and_downsize(256, 512)
        self.conv_layer_3 = _make_conv_block(512, 512)
        # 15x15 -> 8x8
        self.conv_downsize_3 = _make_conv_block_and_downsize(512, 1024)
        self.conv_layer_4 = _make_conv_block(1024, 1024)
        # 8x8 -> 4x4
        self.conv_downsize_4 = _make_conv_block_and_downsize(1024, 2048)
        self.conv_layer_5 = _make_conv_block(2048, 2048)
        # 4x4 -> 1x1
        self.maxpool = nn.MaxPool2d(4)
        self.dense_block = _make_dense_block(2048, 4096)
        self.out_layer = nn.Linear(4096, 600)

    def forward(self, x):
        x = self.initial_conv_block(x)
        x = self.conv_layer_1(x)
        x = self.conv_downsize_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_downsize_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_downsize_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_downsize_4(x)
        x = self.conv_layer_5(x)
        x = self.maxpool(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block(x)
        x = self.out_layer(x)
        return x
