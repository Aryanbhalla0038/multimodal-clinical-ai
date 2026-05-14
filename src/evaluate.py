"""Per-label and macro AUC evaluation."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from src.utils.labels import LABEL_NAMES


@torch.no_grad()
def evaluate(model, dataloader, device, modalities: set[str] | None = None) -> dict:
    """Returns dict with per-label AUC, macro_AUC, macro_AP, macro_F1@0.5."""
    model.eval()
    all_labels, all_preds = [], []
    if modalities is None:
        modalities = {"img", "txt", "vit"}

    for batch in dataloader:
        kwargs = {}
        if "img" in modalities:
            kwargs["image"] = batch["image"].to(device)
        if "txt" in modalities:
            kwargs["input_ids"] = batch["input_ids"].to(device)
            kwargs["attention_mask"] = batch["attention_mask"].to(device)
        if "vit" in modalities:
            kwargs["vitals"] = batch["vitals"].to(device)

        logits = model(**kwargs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(batch["labels"].numpy())

    y_true = np.concatenate(all_labels)  # (N, 14)
    y_pred = np.concatenate(all_preds)

    results: dict = {}
    aucs, aps = [], []
    for i, name in enumerate(LABEL_NAMES):
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            # Happens when a label has only one class in val set
            auc, ap = float("nan"), float("nan")
        results[f"auc/{name}"] = round(float(auc), 4)
        results[f"ap/{name}"] = round(float(ap), 4)
        aucs.append(auc)
        aps.append(ap)

    f1 = f1_score(y_true, (y_pred > 0.5).astype(np.int32), average="macro", zero_division=0)
    results["macro_AUC"] = round(float(np.nanmean(aucs)), 4)
    results["macro_AP"] = round(float(np.nanmean(aps)), 4)
    results["macro_F1@0.5"] = round(float(f1), 4)
    return results
