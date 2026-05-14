"""Smoke tests for encoders. Skipped if heavy weights can't be loaded (offline)."""
import pytest
import torch

from src.encoders.vitals_encoder import TemporalEncoder, TemporalTransformer


def test_temporal_encoder_shape():
    enc = TemporalEncoder(input_dim=6, hidden_dim=64, out_dim=512, num_layers=2)
    x = torch.randn(4, 48, 6)
    y = enc(x)
    assert y.shape == (4, 512)
    assert enc.last_hidden_states is not None
    assert enc.last_hidden_states.shape == (4, 48, 128)  # 2*hidden


def test_temporal_transformer_shape():
    enc = TemporalTransformer(input_dim=6, d_model=64, nhead=4, num_layers=2, out_dim=512)
    x = torch.randn(2, 48, 6)
    y = enc(x)
    assert y.shape == (2, 512)


@pytest.mark.slow
def test_image_encoder_shape():
    # Heavy: downloads ViT weights. Mark `slow` so CI can skip.
    from src.encoders.image_encoder import ImageEncoder
    enc = ImageEncoder(pretrained=False)  # no download for unit test
    x = torch.randn(2, 3, 224, 224)
    y = enc(x)
    assert y.shape == (2, 512)


@pytest.mark.slow
def test_text_encoder_shape():
    from src.encoders.text_encoder import TextEncoder
    enc = TextEncoder()
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones(2, 32, dtype=torch.long)
    y = enc(ids, mask)
    assert y.shape == (2, 512)
