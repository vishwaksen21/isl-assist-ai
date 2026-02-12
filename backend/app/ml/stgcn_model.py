from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn


# MediaPipe Hands 21 landmark graph (undirected edges)
# Indices: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20
HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def build_adjacency(num_nodes: int = 21) -> torch.Tensor:
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i, j in HAND_EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    # self-loops
    for i in range(num_nodes):
        a[i, i] = 1.0
    # normalize: D^{-1/2} A D^{-1/2}
    d = torch.sum(a, dim=1)
    d_inv_sqrt = torch.pow(d, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a @ d_mat


class GraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("A", A, persistent=False)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,C)
        b, t, v, c = x.shape
        x = self.lin(x)  # (B,T,V,out)
        # aggregate neighbors: einsum over V
        x = torch.einsum("btvc,vw->btwc", x, self.A)
        return x


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        *,
        dropout: float,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                padding=(kernel_size // 2, 0),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.res = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,C)
        y = self.gcn(x)  # (B,T,V,C')
        # to conv2d: (B,C',T,V)
        y2 = y.permute(0, 3, 1, 2)
        y2 = self.tcn(y2)
        # residual: (B,C,T,V)
        x2 = x.permute(0, 3, 1, 2)
        r2 = self.res(x2)
        out = self.relu(y2 + r2)
        return out.permute(0, 2, 3, 1)  # back to (B,T,V,C)


class STGCNClassifier(nn.Module):
    """ST-GCN over hand landmarks.

    Expected input: (B,T,63) where 63 = 21 * 3.
    Internally reshapes to (B,T,V=21,C=3).
    """

    def __init__(
        self,
        num_classes: int,
        *,
        dropout: float = 0.3,
        hidden_channels: int = 64,
    ) -> None:
        super().__init__()
        A = build_adjacency(21)
        self.block1 = STGCNBlock(3, hidden_channels, A, dropout=dropout)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels, A, dropout=dropout)
        self.block3 = STGCNBlock(hidden_channels, hidden_channels, A, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,63)
        b, t, f = x.shape
        if f != 63:
            raise ValueError("STGCNClassifier expects 63 features (hand-only)")
        x = x.view(b, t, 21, 3)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # (B,T,V,C) -> (B,C,T,V)
        x2 = x.permute(0, 3, 1, 2)
        pooled = self.pool(x2).squeeze(-1).squeeze(-1)
        return self.fc(pooled)
