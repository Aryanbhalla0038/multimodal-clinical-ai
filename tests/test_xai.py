"""XAI smoke tests (no large weight downloads)."""
import numpy as np
import torch

from src.encoders.vitals_encoder import TemporalEncoder
from src.xai.attention_rollout import get_vitals_importance, plot_vitals_attention


class _MiniModel:
    """A stand-in exposing the same .vit_encoder API."""
    def __init__(self):
        self.vit_encoder = TemporalEncoder(input_dim=6, hidden_dim=32, out_dim=64, num_layers=1)
    def eval(self): pass


def test_vitals_importance_shape_and_range():
    m = _MiniModel()
    x = torch.randn(1, 48, 6)
    imp = get_vitals_importance(m, x)
    assert imp.shape == (48,)
    assert imp.min() >= 0.0 and imp.max() <= 1.0


def test_plot_vitals_attention_returns_fig():
    imp = np.linspace(0, 1, 48)
    vitals = np.random.randn(48, 6)
    fig = plot_vitals_attention(imp, vitals)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)
