"""End-to-end multimodal clinical model.

Supports modality-gated inference (pass None for any missing modality) for
ablation studies. Selectable fusion: cross_attention | early.
"""
from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn

from src.encoders.image_encoder import ImageEncoder
from src.encoders.text_encoder import TextEncoder
from src.encoders.vitals_encoder import TemporalEncoder, TemporalTransformer
from src.fusion.projection import ProjectionHead
from src.fusion.cross_attention import CrossModalAttentionFusion, EarlyFusionBaseline


FusionType = Literal["cross_attention", "early"]


class MultimodalClinicalModel(nn.Module):
    def __init__(
        self,
        num_labels: int = 14,
        d_model: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion: FusionType = "cross_attention",
        image_backbone: str = "vit_base_patch16_224",
        image_pretrained: bool = True,
        text_backbone: str = "dmis-lab/biobert-base-cased-v1.1",
        text_pretrained: bool = True,
        vitals_encoder_type: Literal["lstm", "transformer"] = "lstm",
        vitals_input_dim: int = 6,
        vitals_hidden_dim: int = 256,
        vitals_num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.fusion_type = fusion

        # Encoders (each already projects to d_model)
        self.img_encoder = ImageEncoder(
            model_name=image_backbone, out_dim=d_model, pretrained=image_pretrained,
        )
        self.txt_encoder = TextEncoder(
            model_name=text_backbone, out_dim=d_model, pretrained=text_pretrained,
        )
        if vitals_encoder_type == "lstm":
            self.vit_encoder: nn.Module = TemporalEncoder(
                input_dim=vitals_input_dim,
                hidden_dim=vitals_hidden_dim,
                out_dim=d_model,
                num_layers=vitals_num_layers,
            )
        else:
            self.vit_encoder = TemporalTransformer(
                input_dim=vitals_input_dim, out_dim=d_model,
            )

        # Projection heads keep architecture uniform; in_dim == d_model since encoders already project
        self.img_proj = ProjectionHead(d_model, d_model, dropout)
        self.txt_proj = ProjectionHead(d_model, d_model, dropout)
        self.vit_proj = ProjectionHead(d_model, d_model, dropout)

        if fusion == "cross_attention":
            self.fusion = CrossModalAttentionFusion(d_model, num_heads, dropout)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, num_labels),
            )
        elif fusion == "early":
            self.fusion = None  # type: ignore[assignment]
            self.classifier = EarlyFusionBaseline(d_model, num_labels, dropout=dropout * 2)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")

        # For ablations that drop a modality, replace its embedding with a learned token
        self.img_missing_token = nn.Parameter(torch.zeros(1, d_model))
        self.txt_missing_token = nn.Parameter(torch.zeros(1, d_model))
        self.vit_missing_token = nn.Parameter(torch.zeros(1, d_model))

    # --------------------------------------------------------------------- #
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.img_proj(self.img_encoder(image))

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.txt_proj(self.txt_encoder(input_ids, attention_mask))

    def encode_vitals(self, vitals: torch.Tensor) -> torch.Tensor:
        return self.vit_proj(self.vit_encoder(vitals))

    # --------------------------------------------------------------------- #
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        vitals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns (B, num_labels) logits. Pass None for any missing modality."""
        # Determine batch size from whichever modality is present
        ref = image if image is not None else (
            input_ids if input_ids is not None else vitals
        )
        if ref is None:
            raise ValueError("At least one modality must be provided.")
        B = ref.size(0)
        device = ref.device

        img_e = self.encode_image(image) if image is not None else self.img_missing_token.expand(B, -1).to(device)
        if input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            txt_e = self.encode_text(input_ids, attention_mask)
        else:
            txt_e = self.txt_missing_token.expand(B, -1).to(device)
        vit_e = self.encode_vitals(vitals) if vitals is not None else self.vit_missing_token.expand(B, -1).to(device)

        if self.fusion_type == "cross_attention":
            fused = self.fusion(img_e, txt_e, vit_e)  # type: ignore[misc]
            return self.classifier(fused)
        # early
        return self.classifier(img_e, txt_e, vit_e)
