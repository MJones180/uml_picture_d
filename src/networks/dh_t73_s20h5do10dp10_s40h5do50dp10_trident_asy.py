# `dh_t73_s20h5do10dp10_s40h5do50dp10_trident_asy` network { 1000 -> 1000 }.
# Trainable parameters: 97,829,864
import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# SHARED CONFIG OPTIONS
# ==============================================================================
# Number of input neurons
IN_DIM = 1000
# The output groups for each head
HEAD_OUT_GROUPS = [25, 200, 275]
# Number of neurons expanded out to
OUTER_DIM = 1024
# Number of neurons for the bottleneck
INNER_DIM = 512
# Activation slope
LEAKY_RELU = 0.2

# ==============================================================================
# HEAD 1 CONFIG OPTIONS
# ==============================================================================
HEAD_1_SHARED_DEPTH = 20
HEAD_1_BRANCH_DEPTH = 5
HEAD_1_TOTAL_DEPTH = HEAD_1_SHARED_DEPTH + HEAD_1_BRANCH_DEPTH
HEAD_1_GAMMA_INIT = 1e-2
HEAD_1_SHARED_DROPOUT = 0.10
HEAD_1_BRANCH_DROPOUT = 0.30
HEAD_1_DP_MAX_PROB = 0.10
HEAD_1_DP_PROBS = np.linspace(0, HEAD_1_DP_MAX_PROB, HEAD_1_TOTAL_DEPTH)

# ==============================================================================
# HEAD 2 CONFIG OPTIONS
# ==============================================================================
HEAD_2_SHARED_DEPTH = 40
HEAD_2_BRANCH_DEPTH = 5
HEAD_2_TOTAL_DEPTH = HEAD_2_SHARED_DEPTH + HEAD_2_BRANCH_DEPTH
HEAD_2_GAMMA_INIT = 1e-5
HEAD_2_SHARED_DROPOUT = 0.50
HEAD_2_BRANCH_DROPOUT = 0.60
HEAD_2_DP_MAX_PROB = 0.10
HEAD_2_DP_PROBS = np.linspace(0, HEAD_2_DP_MAX_PROB, HEAD_2_TOTAL_DEPTH)


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

    def __init__(self, gamma_init, dropout, drop_path_prob):
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
        self.gamma = nn.Parameter(torch.full((OUTER_DIM, ), gamma_init))
        self.drop_path = DropPath(drop_path_prob)

    def forward(self, x):
        return x + self.drop_path(self.gamma * self.block(x))


class Network(nn.Module):

    def example_input():
        return torch.rand((2, IN_DIM))

    def __init__(self):
        super().__init__()
        self.head_1_shared = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_1_GAMMA_INIT,
                                        HEAD_1_SHARED_DROPOUT,
                                        HEAD_1_DP_PROBS[i])
                for i in range(HEAD_1_SHARED_DEPTH)
            ],
        )
        self.head_1_branches = nn.ModuleList([
            nn.Sequential(
                *[
                    BottleneckResidualBlock(
                        HEAD_1_GAMMA_INIT, HEAD_1_BRANCH_DROPOUT,
                        HEAD_1_DP_PROBS[HEAD_1_SHARED_DEPTH + i])
                    for i in range(HEAD_1_BRANCH_DEPTH)
                ],
                nn.BatchNorm1d(OUTER_DIM),
                nn.LeakyReLU(LEAKY_RELU),
            ) for _ in HEAD_OUT_GROUPS
        ])
        self.head_1_outs = nn.ModuleList([
            nn.Linear(OUTER_DIM, partition_size)
            for partition_size in HEAD_OUT_GROUPS
        ])
        self.head_2_shared = nn.Sequential(
            nn.Linear(IN_DIM, OUTER_DIM, bias=False),
            nn.BatchNorm1d(OUTER_DIM),
            *[
                BottleneckResidualBlock(HEAD_2_GAMMA_INIT,
                                        HEAD_2_SHARED_DROPOUT,
                                        HEAD_2_DP_PROBS[i])
                for i in range(HEAD_2_SHARED_DEPTH)
            ],
        )
        self.head_2_branches = nn.ModuleList([
            nn.Sequential(
                *[
                    BottleneckResidualBlock(
                        HEAD_2_GAMMA_INIT, HEAD_2_BRANCH_DROPOUT,
                        HEAD_2_DP_PROBS[HEAD_2_SHARED_DEPTH + i])
                    for i in range(HEAD_2_BRANCH_DEPTH)
                ],
                nn.BatchNorm1d(OUTER_DIM),
                nn.LeakyReLU(LEAKY_RELU),
            ) for _ in HEAD_OUT_GROUPS
        ])
        self.head_2_outs = nn.ModuleList([
            nn.Linear(OUTER_DIM, partition_size)
            for partition_size in HEAD_OUT_GROUPS
        ])

    def forward(self, x):
        h1_shared_out = self.head_1_shared(x)
        h1_preds = [
            out(branch(h1_shared_out))
            for branch, out in zip(self.head_1_branches, self.head_1_outs)
        ]
        h1_final = torch.cat(h1_preds, dim=1)
        h2_shared_out = self.head_2_shared(x)
        h2_preds = [
            out(branch(h2_shared_out))
            for branch, out in zip(self.head_2_branches, self.head_2_outs)
        ]
        h2_final = torch.cat(h2_preds, dim=1)
        return h1_final, h2_final
