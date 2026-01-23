# `dh_t25_no_dropout` network { 2x59x59 -> 1512 }.
# Trainable parameters: 89,955,560

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


def _make_dense_block(in_features, out_features):
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU())


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _make_conv_block(2, 128, 3)
        self.conv_block2 = _make_conv_block(128, 128, 3)
        # 59x59 -> 30x30
        self.conv_block3 = _make_conv_block_and_downsize(128, 256, 3)
        self.conv_block4 = _make_conv_block(256, 256, 3)
        # 30x30 -> 15x15
        self.conv_block5 = _make_conv_block_and_downsize(256, 512, 3)
        self.conv_block6 = _make_conv_block(512, 512, 3)
        # 15x15 -> 8x8
        self.conv_block7 = _make_conv_block_and_downsize(512, 1024, 3)
        self.conv_block8 = _make_conv_block(1024, 1024, 3)
        # 8x8 -> 4x4
        self.conv_block9 = _make_conv_block_and_downsize(1024, 2048, 3)
        self.conv_block10 = _make_conv_block(2048, 2048, 3)
        # 4x4 -> 1x1
        self.avgpool1 = nn.AvgPool2d(4)
        self.dense_block1 = _make_dense_block(2048, 4096)
        self.out_layer = nn.Linear(4096, 1512)

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
        # Downsize
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        x = self.avgpool1(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
