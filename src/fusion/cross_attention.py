"""Cross-modal attention fusion.

Each modality (img/txt/vit) attends to the other two via MultiheadAttention,
then we concat the refined embeddings and project back to d_model.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        def mha() -> nn.MultiheadAttention:
            return nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Image attends to text, then to vitals
        self.img_txt_attn = mha()
        self.img_vit_attn = mha()
        # Text attends to image, then to vitals
        self.txt_img_attn = mha()
        self.txt_vit_attn = mha()
        # Vitals attends to image, then to text
        self.vit_img_attn = mha()
        self.vit_txt_attn = mha()

        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

        # For XAI: store the last set of attention weights
        self.last_attn_weights: dict[str, torch.Tensor] = {}

    def _attn(self, mha: nn.MultiheadAttention, q, kv, name: str) -> torch.Tensor:
        out, w = mha(q, kv, kv, need_weights=True, average_attn_weights=True)
        self.last_attn_weights[name] = w.detach()
        return out

    def forward(
        self,
        img_emb: torch.Tensor,
        txt_emb: torch.Tensor,
        vit_emb: torch.Tensor,
    ) -> torch.Tensor:
        """All inputs: (B, d_model). Returns (B, d_model)."""
        img = img_emb.unsqueeze(1)  # (B, 1, d)
        txt = txt_emb.unsqueeze(1)
        vit = vit_emb.unsqueeze(1)

        img_f = self._attn(self.img_txt_attn, img,   txt, "img_txt")
        img_f = self._attn(self.img_vit_attn, img_f, vit, "img_vit")

        txt_f = self._attn(self.txt_img_attn, txt,   img, "txt_img")
        txt_f = self._attn(self.txt_vit_attn, txt_f, vit, "txt_vit")

        vit_f = self._attn(self.vit_img_attn, vit,   img, "vit_img")
        vit_f = self._attn(self.vit_txt_attn, vit_f, txt, "vit_txt")

        combined = torch.cat([img_f.squeeze(1), txt_f.squeeze(1), vit_f.squeeze(1)], dim=-1)
        return self.norm(self.fusion_proj(combined))


class EarlyFusionBaseline(nn.Module):
    """Concatenate + 2-layer MLP. Used as the ablation baseline."""

    def __init__(self, d_model: int = 512, num_labels: int = 14, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_labels),
        )

    def forward(self, img_emb, txt_emb, vit_emb) -> torch.Tensor:
        return self.classifier(torch.cat([img_emb, txt_emb, vit_emb], dim=-1))
