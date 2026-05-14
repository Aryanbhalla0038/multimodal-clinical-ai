"""FastAPI backend for the dashboard.

Run:
    uvicorn dashboard.api.main:app --reload --port 8000

Env vars:
    CKPT_PATH   path to a trained .pt checkpoint (optional; demo mode if absent)
    DEVICE      "cuda" or "cpu"  (default: auto)
"""
from __future__ import annotations
import base64
import io
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.data.image_transforms import build_eval_transform
from src.data.note_tokenizer import tokenize_note, get_tokenizer
from src.data.vitals_preprocessor import preprocess_vitals
from src.models.multimodal_model import MultimodalClinicalModel
from src.utils.labels import LABEL_NAMES
from src.xai.gradcam import ViTGradCAM, overlay_heatmap
from src.xai.attention_rollout import get_vitals_importance

app = FastAPI(title="Multimodal Clinical AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = os.environ.get("CKPT_PATH", "checkpoints/all__cross_attention/best_model.pt")

_model: Optional[MultimodalClinicalModel] = None
_tokenizer = None
_eval_tf = build_eval_transform(224)


def _load_model() -> MultimodalClinicalModel:
    global _model, _tokenizer
    if _model is not None:
        return _model
    model = MultimodalClinicalModel(image_pretrained=False, text_pretrained=False)
    ckpt_path = Path(CKPT_PATH)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sd = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[api] Loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"[api] WARNING: no checkpoint at {ckpt_path}; DEMO mode (random weights).")
    model.to(DEVICE).eval()
    _model = model
    _tokenizer = get_tokenizer()
    return model


def _img_bytes_to_tensor(b: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return _eval_tf(img).unsqueeze(0).to(DEVICE)


def _img_bytes_to_rgb_uint8(b: bytes, size: int = 224) -> np.ndarray:
    img = Image.open(io.BytesIO(b)).convert("RGB").resize((size, size))
    return np.array(img, dtype=np.uint8)


def _vitals_csv_to_tensor(b: bytes) -> torch.Tensor:
    df = pd.read_csv(io.BytesIO(b))
    arr, _ = preprocess_vitals(df)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


def _encode_b64_png(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


@app.on_event("startup")
def _startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": DEVICE, "labels": LABEL_NAMES,
            "ckpt": str(CKPT_PATH), "loaded": _model is not None}


@app.get("/labels")
def labels() -> dict:
    return {"labels": LABEL_NAMES}


@app.post("/predict")
async def predict(
    image: UploadFile = File(default=None),
    note: str = Form(default=""),
    vitals_csv: UploadFile = File(default=None),
    gradcam_label: Optional[str] = Form(default=None),
):
    model = _load_model()
    missing: list[str] = []

    img_t = None
    img_rgb = None
    if image is not None:
        raw = await image.read()
        if raw:
            img_t = _img_bytes_to_tensor(raw)
            img_rgb = _img_bytes_to_rgb_uint8(raw)
    if img_t is None:
        missing.append("image")

    input_ids = attn_mask = None
    if note and note.strip():
        tok = tokenize_note(note)
        input_ids = tok["input_ids"].to(DEVICE)
        attn_mask = tok["attention_mask"].to(DEVICE)
    else:
        missing.append("notes")

    vit_t = None
    if vitals_csv is not None:
        raw_v = await vitals_csv.read()
        if raw_v:
            try:
                vit_t = _vitals_csv_to_tensor(raw_v)
            except Exception as exc:  # noqa: BLE001
                print(f"[api] vitals parse failed: {exc}")
    if vit_t is None:
        missing.append("vitals")

    if img_t is None and input_ids is None and vit_t is None:
        return {"error": "Provide at least one modality."}

    with torch.no_grad():
        logits = model(image=img_t, input_ids=input_ids,
                       attention_mask=attn_mask, vitals=vit_t)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()
    predictions = dict(zip(LABEL_NAMES, [round(float(p), 4) for p in probs]))

    # Grad-CAM
    gradcam_b64 = None
    if img_t is not None:
        try:
            target_label = gradcam_label if gradcam_label in LABEL_NAMES else max(
                predictions, key=predictions.get)
            class_idx = LABEL_NAMES.index(target_label)
            cam_obj = ViTGradCAM(model)
            try:
                heatmap = cam_obj.generate(img_t.clone().requires_grad_(False), class_idx)
            finally:
                cam_obj.close()
            overlay = overlay_heatmap(img_rgb, heatmap, alpha=0.45)
            gradcam_b64 = _encode_b64_png(overlay)
        except Exception as exc:  # noqa: BLE001
            print(f"[api] gradcam failed: {exc}")

    # Vitals importance
    vitals_importance = None
    if vit_t is not None:
        try:
            vitals_importance = get_vitals_importance(model, vit_t).tolist()
        except Exception as exc:  # noqa: BLE001
            print(f"[api] vitals importance failed: {exc}")

    return {
        "predictions": predictions,
        "gradcam_image": gradcam_b64,
        "gradcam_label": target_label if img_t is not None else None,
        "vitals_importance": vitals_importance,
        "missing_modalities": missing,
        "disclaimer": "For research purposes only. Not for clinical decision-making.",
    }
