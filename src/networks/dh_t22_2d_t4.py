# `dh_t22_2d_t4` network { 2x59x59 -> 2x30x30 }.
# Trainable parameters: 152,165,120

import torch
import torch.nn as nn


def _bn_lr_wrapper(layer, out_features):
    return nn.Sequential(
        layer,
        nn.BatchNorm2d(out_features),
        nn.LeakyReLU(),
    )


def _conv(in_features, out_features, stride=1):
    return nn.Conv2d(
        in_channels=in_features,
        out_channels=out_features,
        kernel_size=3,
        bias=False,
        stride=stride,
        padding=1,
    )


def _conv_block(in_features, out_features):
    return _bn_lr_wrapper(
        _conv(in_features, out_features),
        out_features,
    )


def _conv_block_and_downsize(in_features, out_features):
    return _bn_lr_wrapper(
        _conv(in_features, out_features, stride=2),
        out_features,
    )


def _deconv_block_and_upsize(in_features, out_features, padding):
    return _bn_lr_wrapper(
        nn.ConvTranspose2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=4,
            bias=False,
            stride=2,
            padding=padding,
        ),
        out_features,
    )


def _dense_block(in_features, out_features, dropout):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.LeakyReLU(),
        nn.Dropout(dropout),
    )


class Network(nn.Module):

    def example_input():
        return torch.rand((1, 2, 59, 59))

    def __init__(self):
        super().__init__()
        self.conv_block1 = _conv_block(2, 128)
        self.conv_block2 = _conv_block(128, 128)
        self.conv_block3 = _conv_block(128, 128)
        # Pixels: 59x59 -> 30x30
        self.conv_block4 = _conv_block_and_downsize(128, 256)
        self.conv_block5 = _conv_block(256, 256)
        self.conv_block6 = _conv_block(256, 256)
        # Pixels: 30x30 -> 15x15
        self.conv_block7 = _conv_block_and_downsize(256, 512)
        self.conv_block8 = _conv_block(512, 512)
        self.conv_block9 = _conv_block(512, 512)
        # Pixels: 15x15 -> 8x8
        self.conv_block10 = _conv_block_and_downsize(512, 1024)
        self.conv_block11 = _conv_block(1024, 1024)
        self.conv_block12 = _conv_block(1024, 1024)
        # Pixels: 8x8 -> 4x4
        self.conv_block13 = _conv_block_and_downsize(1024, 2048)
        self.conv_block14 = _conv_block(2048, 2048)
        self.conv_block15 = _conv_block(2048, 2048)
        # Pixels: 4x4 -> 1x1
        self.avgpool1 = nn.AvgPool2d(4)
        # Dense blocks: 2048 -> 4096 -> 2048
        self.dense_block1 = _dense_block(2048, 4096, 0.3)
        self.dense_block2 = _dense_block(4096, 2048, 0.3)
        # Reshape to (batch size, 128, 4, 4)
        # Pixels: 4x4 -> 8x8
        self.deconv_block1 = _deconv_block_and_upsize(128, 512, 1)
        self.deconv_block2 = _conv_block(512, 512)
        self.deconv_block3 = _conv_block(512, 512)
        # Pixels: 8x8 -> 14x14
        self.deconv_block4 = _deconv_block_and_upsize(512, 256, 2)
        self.deconv_block5 = _conv_block(256, 256)
        self.deconv_block6 = _conv_block(256, 256)
        # Pixels: 14x14 -> 30x30
        self.deconv_block7 = _deconv_block_and_upsize(256, 128, 0)
        self.deconv_block8 = _conv_block(128, 128)
        self.deconv_block9 = _conv(128, 2)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        x = self.conv_block11(x)
        x = self.conv_block12(x)
        x = self.conv_block13(x)
        x = self.conv_block14(x)
        x = self.conv_block15(x)
        x = self.avgpool1(x)
        x = torch.squeeze(x, 2)
        x = torch.squeeze(x, 2)
        x = self.dense_block1(x)
        x = self.dense_block2(x)
        x = torch.reshape(x, (-1, 128, 4, 4))
        x = self.deconv_block1(x)
        x = self.deconv_block2(x)
        x = self.deconv_block3(x)
        x = self.deconv_block4(x)
        x = self.deconv_block5(x)
        x = self.deconv_block6(x)
        x = self.deconv_block7(x)
        x = self.deconv_block8(x)
        x = self.deconv_block9(x)
        return x
