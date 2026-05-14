"""Text encoder: BioBERT CLS token -> Linear -> LayerNorm -> d_model."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig, BertModel


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        out_dim: int = 512,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            # Build a BERT-base architecture without downloading weights.
            # Matches BioBERT-base-cased: hidden=768, layers=12, heads=12, vocab=28996.
            self.bert = BertModel(BertConfig(
                vocab_size=28996, hidden_size=768, num_hidden_layers=12,
                num_attention_heads=12, intermediate_size=3072,
                max_position_embeddings=512, type_vocab_size=2,
            ))
        hidden = self.bert.config.hidden_size  # 768 for BioBERT-base
        self.proj = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim
        if freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """input_ids/attention_mask: (B, L). Returns (B, out_dim)."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # (B, 768)
        return self.norm(self.proj(cls))
