"""Image encoder: ViT-Base from timm, projected to shared d_model space."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ImageEncoder(nn.Module):
    """ViT-Base patch16 224 -> 768-d CLS features -> Linear -> LayerNorm -> d_model."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        out_dim: int = 512,
        pretrained: bool = True,
        return_tokens: bool = False,
    ):
        super().__init__()
        # num_classes=0 strips the classifier; ViT forward returns the pooled CLS feature.
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        embed_dim = self.vit.num_features  # 768 for ViT-Base
        self.proj = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim
        self.return_tokens = return_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W). Returns (B, out_dim)."""
        features = self.vit(x)  # (B, 768)
        return self.norm(self.proj(features))

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Returns patch+CLS token features (B, N+1, 768) for XAI."""
        return self.vit.forward_features(x)
