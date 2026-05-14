"""Smoke tests for fusion modules (no heavy weights)."""
import torch

from src.fusion.projection import ProjectionHead
from src.fusion.cross_attention import CrossModalAttentionFusion, EarlyFusionBaseline


def test_projection_head_shape():
    head = ProjectionHead(in_dim=512, out_dim=512)
    x = torch.randn(4, 512)
    assert head(x).shape == (4, 512)


def test_cross_modal_attention_fusion_shape_and_caches_weights():
    fusion = CrossModalAttentionFusion(d_model=64, num_heads=4)
    img = torch.randn(2, 64)
    txt = torch.randn(2, 64)
    vit = torch.randn(2, 64)
    y = fusion(img, txt, vit)
    assert y.shape == (2, 64)
    # Six pairwise attentions cached
    assert set(fusion.last_attn_weights.keys()) == {
        "img_txt", "img_vit", "txt_img", "txt_vit", "vit_img", "vit_txt"
    }


def test_early_fusion_shape():
    fusion = EarlyFusionBaseline(d_model=64, num_labels=14)
    y = fusion(torch.randn(3, 64), torch.randn(3, 64), torch.randn(3, 64))
    assert y.shape == (3, 14)


def test_cross_attention_is_differentiable():
    fusion = CrossModalAttentionFusion(d_model=32, num_heads=4)
    img = torch.randn(2, 32, requires_grad=True)
    txt = torch.randn(2, 32, requires_grad=True)
    vit = torch.randn(2, 32, requires_grad=True)
    y = fusion(img, txt, vit).sum()
    y.backward()
    assert img.grad is not None and txt.grad is not None and vit.grad is not None
