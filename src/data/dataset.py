"""Multimodal dataset.

Supports:
  - Real CheXpert: images + labels CSV (and optionally paired notes/vitals).
  - Synthetic-MIMIC fallback: random plausible notes & vitals so the full
    pipeline trains end-to-end while you wait for MIMIC access.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.image_transforms import load_image_rgb, build_train_transform, build_eval_transform
from src.data.note_tokenizer import tokenize_note
from src.data.vitals_preprocessor import synthetic_vitals
from src.utils.labels import LABEL_NAMES, apply_uncertainty_policy


# A small corpus of canned clinical-style sentences for synthetic notes.
# Token frequency roughly biased toward findings the labels describe.
_SYNTH_PHRASES = {
    "Atelectasis":       "left lower lobe atelectasis with volume loss",
    "Cardiomegaly":      "the cardiac silhouette is enlarged",
    "Consolidation":     "patchy airspace consolidation in the right lower lobe",
    "Edema":             "interstitial pulmonary edema and vascular congestion",
    "Enlarged Cardiomediastinum": "the cardiomediastinal silhouette is widened",
    "Fracture":          "acute right rib fracture identified",
    "Lung Lesion":       "a 1 cm nodular lung lesion is seen",
    "Lung Opacity":      "diffuse bilateral lung opacities",
    "No Finding":        "no acute cardiopulmonary process",
    "Pleural Effusion":  "moderate right-sided pleural effusion",
    "Pleural Other":     "pleural thickening laterally",
    "Pneumonia":         "findings concerning for pneumonia",
    "Pneumothorax":      "small apical pneumothorax noted",
    "Support Devices":   "endotracheal tube and right IJ central line in place",
}


def synthetic_note(label_vec: np.ndarray, seed: int | None = None) -> str:
    """Build a synthetic discharge-summary-style note conditioned on the labels."""
    rng = np.random.default_rng(seed)
    parts = ["Assessment and Plan:"]
    for i, name in enumerate(LABEL_NAMES):
        if label_vec[i] > 0.5:
            parts.append(_SYNTH_PHRASES[name] + ".")
    if len(parts) == 1:
        parts.append(_SYNTH_PHRASES["No Finding"] + ".")
    # Pad with neutral filler so length variance is realistic
    filler = [
        "Vitals stable.", "Patient afebrile.",
        "Will continue current management.",
        "Monitor overnight.", "Repeat imaging in the morning.",
    ]
    for _ in range(rng.integers(2, 6)):
        parts.append(rng.choice(filler))
    return " ".join(parts)


class MultimodalDataset(Dataset):
    """
    Args:
        labels_csv: CSV with at least an image-path column + the 14 CheXpert label columns.
        image_root: prepended to relative paths in `image_col`.
        image_col: name of the image-path column (CheXpert default: "Path").
        notes_col: optional column name with raw clinical text. If absent, synthetic notes are generated.
        vitals_dir: optional dir with per-row `.npy` vitals arrays (shape (T, 6)).
                    If None, synthetic vitals are generated deterministically per row.
        split: "train" | "val" | "test" (only affects augmentation).
    """

    def __init__(
        self,
        labels_csv: str | Path | pd.DataFrame,
        image_root: str | Path,
        image_col: str = "Path",
        notes_col: Optional[str] = None,
        vitals_dir: Optional[str | Path] = None,
        split: str = "train",
        image_size: int = 224,
        text_max_len: int = 512,
        vitals_seq_len: int = 48,
        uncertainty_policy: str = "u_zeros",
        u_ones_labels: Optional[list[str]] = None,
        text_model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        synthetic_text: bool = True,
        synthetic_vitals_flag: bool = True,
    ):
        if isinstance(labels_csv, pd.DataFrame):
            self.df = labels_csv.copy()
        else:
            self.df = pd.read_csv(labels_csv)

        # Ensure all 14 label columns exist (some splits may omit some)
        for name in LABEL_NAMES:
            if name not in self.df.columns:
                self.df[name] = 0.0
        self.df = apply_uncertainty_policy(self.df, policy=uncertainty_policy,
                                           u_ones_labels=u_ones_labels)

        self.image_root = Path(image_root)
        self.image_col = image_col
        self.notes_col = notes_col
        self.vitals_dir = Path(vitals_dir) if vitals_dir is not None else None
        self.split = split
        self.image_size = image_size
        self.text_max_len = text_max_len
        self.vitals_seq_len = vitals_seq_len
        self.text_model_name = text_model_name
        self.synthetic_text = synthetic_text
        self.synthetic_vitals_flag = synthetic_vitals_flag

        self.tf = (build_train_transform(image_size) if split == "train"
                   else build_eval_transform(image_size))

    def __len__(self) -> int:
        return len(self.df)

    def _get_labels(self, row: pd.Series) -> torch.Tensor:
        vec = row[LABEL_NAMES].to_numpy(dtype=np.float32)
        # Replace NaN (u_ignore) with 0 here; loss masking handled in train loop if needed.
        vec = np.nan_to_num(vec, nan=0.0)
        return torch.from_numpy(vec)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        # ---- Image ----
        img_path = self.image_root / str(row[self.image_col])
        image = self.tf(load_image_rgb(img_path))

        # ---- Labels ----
        labels = self._get_labels(row)

        # ---- Text ----
        if self.notes_col and self.notes_col in row and isinstance(row[self.notes_col], str):
            text = row[self.notes_col]
        elif self.synthetic_text:
            text = synthetic_note(labels.numpy(), seed=idx)
        else:
            text = ""
        tok = tokenize_note(text, max_len=self.text_max_len, model_name=self.text_model_name)
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        # ---- Vitals ----
        if self.vitals_dir is not None:
            vp = self.vitals_dir / f"{idx}.npy"
            if vp.exists():
                vitals_np = np.load(vp).astype(np.float32)
            elif self.synthetic_vitals_flag:
                vitals_np = synthetic_vitals(seed=idx, seq_len=self.vitals_seq_len)
            else:
                vitals_np = np.zeros((self.vitals_seq_len, 6), dtype=np.float32)
        elif self.synthetic_vitals_flag:
            vitals_np = synthetic_vitals(seed=idx, seq_len=self.vitals_seq_len)
        else:
            vitals_np = np.zeros((self.vitals_seq_len, 6), dtype=np.float32)
        vitals = torch.from_numpy(vitals_np)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vitals": vitals,
            "labels": labels,
            "id": str(row.get("patient_id", idx)),
        }
