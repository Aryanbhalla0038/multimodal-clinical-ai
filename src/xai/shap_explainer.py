"""SHAP explainer for BioBERT clinical text.

Uses shap.Explainer with a `transformers` masker so we get
token-level attributions out of the box.
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import torch


def make_text_predict_fn(
    model,
    tokenizer,
    class_idx: int,
    device: str = "cpu",
    max_len: int = 512,
) -> Callable:
    """Returns a function: list[str] -> np.ndarray (N,) sigmoid prob for class_idx."""

    def predict(texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        tokens = tokenizer(
            list(texts), return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        ).to(device)
        with torch.no_grad():
            logits = model(input_ids=tokens["input_ids"],
                           attention_mask=tokens["attention_mask"])
            probs = torch.sigmoid(logits)[:, class_idx]
        return probs.cpu().numpy()

    return predict


def explain_text(
    model,
    tokenizer,
    text: str,
    class_idx: int,
    device: str = "cpu",
    max_len: int = 512,
):
    """Returns a shap.Explanation object. Visualize with `shap.plots.text(...)`."""
    import shap  # local import — heavy
    predict = make_text_predict_fn(model, tokenizer, class_idx, device, max_len)
    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict, masker)
    return explainer([text])
