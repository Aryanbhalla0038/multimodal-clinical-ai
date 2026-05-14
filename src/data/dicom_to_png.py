"""Convert DICOM chest X-rays to normalized PNGs.

Used for MIMIC-CXR (DICOM). CheXpert ships JPGs already; for those use
`load_image` directly.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

try:
    import pydicom  # optional at import time
except ImportError:  # pragma: no cover
    pydicom = None


def dicom_to_array(dicom_path: str | Path) -> np.ndarray:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM conversion")
    dcm = pydicom.dcmread(str(dicom_path))
    arr = dcm.pixel_array.astype(np.float32)
    # Window leveling — clip to ±2 stds of the image
    mu, sd = float(arr.mean()), float(arr.std())
    arr = np.clip(arr, mu - 2 * sd, mu + 2 * sd)
    return arr


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    return arr.astype(np.uint8)


def dicom_to_png(dicom_path: str | Path, out_path: str | Path, size: int = 224) -> None:
    arr = dicom_to_array(dicom_path)
    arr = normalize_to_uint8(arr)
    arr = cv2.resize(arr, (size, size))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), arr)


def batch_convert(dicom_dir: str | Path, out_dir: str | Path, size: int = 224) -> int:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in Path(dicom_dir).rglob("*.dcm"):
        dicom_to_png(p, out_dir / f"{p.stem}.png", size=size)
        n += 1
    return n
