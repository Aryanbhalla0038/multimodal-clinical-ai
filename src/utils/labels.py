"""CheXpert 14-label taxonomy and uncertainty handling."""
from __future__ import annotations
import numpy as np
import pandas as pd

LABEL_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

NUM_LABELS = len(LABEL_NAMES)


def apply_uncertainty_policy(
    df: pd.DataFrame,
    policy: str = "u_zeros",
    u_ones_labels: list[str] | None = None,
) -> pd.DataFrame:
    """
    CheXpert labels: 1.0 = positive, 0.0 = negative, -1.0 = uncertain, NaN = not mentioned.

    policy:
      - "u_zeros": uncertain -> 0
      - "u_ones":  uncertain -> 1
      - "u_ignore": uncertain -> NaN (ignored in BCE via masking elsewhere)

    `u_ones_labels` overrides the global policy: those labels use U-Ones.
    NaN ("not mentioned") is always treated as 0 (negative) per CheXpert convention
    unless using u_ignore for that label too.
    """
    df = df.copy()
    u_ones_labels = u_ones_labels or []
    for col in LABEL_NAMES:
        if col not in df.columns:
            continue
        s = df[col]
        # not mentioned -> 0
        s = s.fillna(0.0)
        if col in u_ones_labels:
            s = s.replace(-1.0, 1.0)
        elif policy == "u_ones":
            s = s.replace(-1.0, 1.0)
        elif policy == "u_zeros":
            s = s.replace(-1.0, 0.0)
        elif policy == "u_ignore":
            s = s.replace(-1.0, np.nan)
        else:
            raise ValueError(f"Unknown uncertainty policy: {policy}")
        df[col] = s
    return df
