# `dh_t10_avg` network { 2x59x59 -> 1512 }.
# Trainable parameters: 53,467,112

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
        self.conv_block1 = _make_conv_block(2, 128, 3)
        self.conv_block2 = _make_conv_block(128, 128, 3)
        self.conv_block3 = _make_conv_block(128, 128, 3)
        # 59x59 -> 30x30
        self.maxpool1 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv_block4 = _make_conv_block(128, 256, 3)
        self.conv_block5 = _make_conv_block(256, 256, 3)
        self.conv_block6 = _make_conv_block(256, 256, 3)
        # 30x30 -> 15x15
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv_block7 = _make_conv_block(256, 512, 3)
        self.conv_block8 = _make_conv_block(512, 512, 3)
        self.conv_block9 = _make_conv_block(512, 512, 3)
        self.conv_block10 = _make_conv_block(512, 512, 3)
        # 15x15 -> 8x8
        self.maxpool3 = nn.MaxPool2d(2, ceil_mode=True)
        self.conv_block11 = _make_conv_block(512, 1024, 3)
        self.conv_block12 = _make_conv_block(1024, 1024, 3)
        self.conv_block13 = _make_conv_block(1024, 1024, 3)
        self.conv_block14 = _make_conv_block(1024, 1024, 3)
        # 8x8 -> 1x1
        self.avgpool1 = nn.AvgPool2d(8)
        self.dense_block1 = _make_dense_block(1024, 4096, 0.3)
        self.out_layer = nn.Linear(4096, 1512)

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
        x = self.conv_block10(x)
        x = self.maxpool3(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        x = self.conv_block13(x)
        x = self.conv_block14(x)
        x = self.avgpool1(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block1(x)
        x = self.out_layer(x)
        return x
