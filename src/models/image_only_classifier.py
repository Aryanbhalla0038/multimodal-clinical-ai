"""Pure image-only chest X-ray classifier.

A minimal ViT-Base + linear head, with no multimodal fusion machinery.
This bypasses the failure mode of the multimodal architecture where
learnable "missing tokens" for absent modalities allow the optimizer
to ignore the image input entirely.

Call signature mirrors MultimodalClinicalModel so the FastAPI backend
can swap models with zero code changes:

    logits = model(image=img_tensor, input_ids=None,
                   attention_mask=None, vitals=None)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class ChestXRayClassifier(nn.Module):
    """ViT-Base/16 + linear head for 14-label CheXpert classification."""

    def __init__(
        self,
        num_labels: int = 14,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 768 for vit_base_patch16_224
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, num_labels)

        # Tiny init for the head so initial logits ≈ 0 (sigmoid ≈ 0.5)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor | None = None,  # ignored, kept for API parity
        attention_mask: torch.Tensor | None = None,  # ignored
        vitals: torch.Tensor | None = None,  # ignored
    ) -> torch.Tensor:
        if image is None:
            raise ValueError("ChestXRayClassifier requires an image input.")
        feat = self.backbone(image)  # (B, 768)
        return self.head(self.dropout(feat))  # (B, num_labels)
