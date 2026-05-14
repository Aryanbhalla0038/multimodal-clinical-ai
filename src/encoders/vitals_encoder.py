"""Vitals/temporal encoders: BiLSTM and Transformer variants."""
from __future__ import annotations
import math

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    """Bidirectional LSTM. Input (B, T, F) -> (B, out_dim).

    We keep the full sequence on `last_hidden_states` for XAI (attention rollout).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        out_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim
        # Cached for XAI hooks
        self.last_hidden_states: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)  # (B, T, 2*hidden)
        self.last_hidden_states = out
        last = out[:, -1, :]
        return self.norm(self.proj(last))


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTransformer(nn.Module):
    """Transformer encoder over vitals time series."""

    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        out_dim: int = 512,
        dropout: float = 0.1,
        max_len: int = 96,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = _SinusoidalPE(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=4 * d_model, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim
        self.last_hidden_states: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pe(x)
        x = self.transformer(x)
        self.last_hidden_states = x
        return self.norm(self.out_proj(x[:, -1, :]))


def build_vitals_encoder(cfg) -> nn.Module:
    """Factory: cfg can be a dict-like with keys: encoder_type, input_dim, hidden_dim, num_layers."""
    enc_type = getattr(cfg, "encoder_type", "lstm") if not isinstance(cfg, dict) else cfg.get("encoder_type", "lstm")
    if enc_type == "lstm":
        return TemporalEncoder(
            input_dim=cfg["input_dim"] if isinstance(cfg, dict) else cfg.input_dim,
            hidden_dim=cfg["hidden_dim"] if isinstance(cfg, dict) else cfg.hidden_dim,
            num_layers=cfg["num_layers"] if isinstance(cfg, dict) else cfg.num_layers,
        )
    if enc_type == "transformer":
        return TemporalTransformer(
            input_dim=cfg["input_dim"] if isinstance(cfg, dict) else cfg.input_dim,
        )
    raise ValueError(f"Unknown vitals encoder_type: {enc_type}")
