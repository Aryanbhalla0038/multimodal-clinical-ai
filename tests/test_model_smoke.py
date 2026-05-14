"""End-to-end shape/grad smoke test for MultimodalClinicalModel.

Uses pretrained=False to avoid weight downloads in CI.
NOTE: This will still attempt to load BioBERT (HF). Marked `slow` because
it requires internet on first run.
"""
import pytest
import torch


@pytest.mark.slow
def test_multimodal_model_forward_and_backward():
    from src.models.multimodal_model import MultimodalClinicalModel
    model = MultimodalClinicalModel(
        num_labels=14, d_model=64, num_heads=4,
        image_pretrained=False,
    )
    B = 2
    image = torch.randn(B, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (B, 32))
    attn_mask = torch.ones(B, 32, dtype=torch.long)
    vitals = torch.randn(B, 48, 6)

    logits = model(image=image, input_ids=input_ids,
                   attention_mask=attn_mask, vitals=vitals)
    assert logits.shape == (B, 14)
    loss = logits.sum()
    loss.backward()


@pytest.mark.slow
def test_multimodal_model_modality_dropout():
    from src.models.multimodal_model import MultimodalClinicalModel
    model = MultimodalClinicalModel(image_pretrained=False, d_model=64, num_heads=4)
    B = 2
    # Only image
    logits = model(image=torch.randn(B, 3, 224, 224))
    assert logits.shape == (B, 14)
    # Only vitals
    logits = model(vitals=torch.randn(B, 48, 6))
    assert logits.shape == (B, 14)
