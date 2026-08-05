# `dh_t80_50d12e_do30dp30` network { 756 -> 1000 }.
# Trainable parameters: 41,610,216
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# CONFIG OPTIONS
# ==============================================================================
# Number of input neurons
IN_DIM = 756
# Number of output neurons
OUT_DIM = 1000
# Number of neurons expanded out to
OUTER_DIM = 768
# Number of neurons for the bottleneck
INNER_DIM = 512
# Activation slope
LEAKY_RELU = 0.2
# LayerScale starting value
GAMMA_INIT = 1e-2
# Number of blocks
DEPTH = 50
# Layer to send features to the end from
FEATURE_EXTRACTION_LAYER = 12
# Dropout rate
DROPOUT = 0.30
# DropPath probability of dropping the last layer
DP_MAX_PROB = 0.30
# DropPath probabilities linearly increase from the first to last layer
DP_PROBS = np.linspace(0, DP_MAX_PROB, DEPTH)


# Also known as Stochastic Depth
class DropPath(nn.Module):

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        if self.drop_prob == 0 or not self.training:
            return x
        keep_rows = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device)
        keep_rows = (keep_rows + self.keep_prob).floor_()
        return x.div(self.keep_prob) * keep_rows


class BottleneckResidualBlock(nn.Module):

    def __init__(self, dropout, drop_path_prob):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(OUTER_DIM, INNER_DIM, bias=False),
            nn.BatchNorm1d(INNER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Dropout(dropout),
            nn.Linear(INNER_DIM, OUTER_DIM, bias=False),
        )
        self.gamma = nn.Parameter(torch.full((OUTER_DIM, ), GAMMA_INIT))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
        )
        self.blocks = nn.Sequential(*[
            BottleneckResidualBlock(DROPOUT, DP_PROBS[layer_idx])
            for layer_idx in range(DEPTH)
        ])
        self.output = nn.Sequential(
            nn.BatchNorm1d(2 * OUTER_DIM),
            nn.LeakyReLU(LEAKY_RELU),
            nn.Linear(2 * OUTER_DIM, OUT_DIM),
        )

    def forward(self, x):
        x = self.input(x)
        features_for_end = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == (FEATURE_EXTRACTION_LAYER - 1):
                features_for_end = x
        combined_features = torch.cat([features_for_end, x], dim=-1)
        return self.output(combined_features)
