"""Per-timestep importance for the vitals encoder + plotting helper."""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def get_vitals_importance(model, vitals_tensor: torch.Tensor) -> np.ndarray:
    """
    Args:
        model: a MultimodalClinicalModel.
        vitals_tensor: (1, T, F) tensor.

    Returns: (T,) float array in [0, 1].

    For BiLSTM: per-timestep L2 norm of the hidden state.
    For Transformer encoder: average over rows of last-layer self-attention to
    obtain a per-timestep saliency.
    """
    model.eval()
    enc = model.vit_encoder
    # Force a forward pass so .last_hidden_states is populated
    _ = enc(vitals_tensor)
    states = enc.last_hidden_states  # (1, T, D)
    if states is None:
        raise RuntimeError("Vitals encoder did not cache hidden states.")
    importance = torch.linalg.norm(states.squeeze(0), dim=-1).cpu().numpy()  # (T,)
    if importance.max() - importance.min() < 1e-8:
        return np.zeros_like(importance)
    return (importance - importance.min()) / (importance.max() - importance.min())


def plot_vitals_attention(
    importance: np.ndarray,
    vitals: np.ndarray,
    vital_names: Optional[list[str]] = None,
):
    """Two-panel plot: top = importance bars, bottom = traces."""
    if vital_names is None:
        vital_names = ["HR", "SBP", "DBP", "SpO2", "RR", "Temp"]
    hours = np.arange(len(importance))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.bar(hours, importance, color="steelblue", alpha=0.75)
    ax1.set_ylabel("Attention weight")
    ax1.set_title("Timestep importance")
    ax1.set_ylim(0, 1.05)

    for i, name in enumerate(vital_names[: vitals.shape[1]]):
        ax2.plot(hours, vitals[:, i], label=name, linewidth=1.2)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Normalized value")
    ax2.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    return fig
