# `dh_st_t1` network { 1x118x59 -> 1512 }.
# Trainable parameters: 13,018,792

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
        return torch.rand((1, 1, 118, 59))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _make_conv_block(1, 64, 3)
        self.conv_block2 = _make_conv_block(64, 64, 3)
        self.conv_block3 = _make_conv_block(64, 64, 3)
        # 118x59 -> 59x29
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv_block4 = _make_conv_block(64, 128, 3)
        self.conv_block5 = _make_conv_block(128, 128, 3)
        self.conv_block6 = _make_conv_block(128, 128, 3)
        # 59x29 -> 29x14
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv_block7 = _make_conv_block(128, 256, 3)
        self.conv_block8 = _make_conv_block(256, 256, 3)
        self.conv_block9 = _make_conv_block(256, 256, 3)
        # 29x14 -> 14x7
        self.maxpool3 = nn.MaxPool2d(2)
        self.conv_block10 = _make_conv_block(256, 1024, 3)
        self.conv_block11 = _make_conv_block(1024, 1024, 3)
        self.conv_block12 = _make_conv_block(1024, 1024, 3)
        # 14x7 -> 2x1
        self.maxpool4 = nn.MaxPool2d(7)
        self.dense_block1 = _make_dense_block(1024, 2048, 0.3)
        self.out_layer = nn.Linear(2048, 1512)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.maxpool1(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.maxpool2(x)
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.maxpool3(x)
        x = self.conv_block10(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        x = self.maxpool4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
