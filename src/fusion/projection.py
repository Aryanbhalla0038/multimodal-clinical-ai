"""Projection head: maps an encoder output into the shared d_model space."""
from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> LayerNorm."""

    def __init__(self, in_dim: int, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
