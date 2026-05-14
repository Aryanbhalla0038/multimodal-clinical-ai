# Multimodal Clinical AI — Implementation Guide
### ViT + BioBERT + Temporal Encoder with Cross-Modal Attention Fusion
**24-Week Roadmap | CheXpert 14-Label Diagnosis | Grad-CAM · SHAP · Attention Rollout**

---

## Architecture Overview

```
Input Modalities
       │                    │                      │
Radiology Images      Clinical Notes         Patient Vitals
CheXpert/MIMIC-CXR  Discharge summaries     HR, BP, SpO₂ series
       │                    │                      │
  ViT Encoder          BioBERT Encoder       Temporal Encoder
 Patch embeddings      CLS token emb          LSTM / TST
       │                    │                      │
       └──────────── Fusion Layer ─────────────────┘
              Cross-modal attention fusion
         Projects all → shared d=512, cross-attends
                          │
               Multi-label diagnosis head
               Softmax over 14 CheXpert labels
                          │
              Explainability Layer (XAI)
        ┌──────────────┬──────────────────┐
     Grad-CAM      SHAP values      Attention rollout
  Image attention  Token importance  Vital importance
```

---

## Project Folder Structure

```
multimodal-clinical-ai/
├── data/
│   ├── raw/                    # MIMIC-III, CheXpert raw downloads
│   ├── processed/              # preprocessed tensors & CSVs
│   └── splits/                 # train/val/test JSON splits
├── src/
│   ├── encoders/
│   │   ├── image_encoder.py    # ViT wrapper
│   │   ├── text_encoder.py     # BioBERT wrapper
│   │   └── vitals_encoder.py   # LSTM / TST
│   ├── fusion/
│   │   ├── projection.py       # Linear projection heads
│   │   └── cross_attention.py  # nn.MultiheadAttention
│   ├── models/
│   │   └── multimodal_model.py # Full pipeline
│   ├── xai/
│   │   ├── gradcam.py
│   │   ├── shap_explainer.py
│   │   └── attention_rollout.py
│   ├── data/
│   │   ├── dicom_to_png.py
│   │   ├── note_tokenizer.py
│   │   └── vitals_preprocessor.py
│   └── train.py / evaluate.py
├── dashboard/                  # React UI
│   ├── src/
│   └── public/
├── notebooks/                  # Exploratory analysis
├── configs/                    # YAML experiment configs
├── tests/
├── requirements.txt
└── README.md
```

---

## Python Dependencies

```bash
conda create -n clinicalai python=3.10
conda activate clinicalai

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install timm transformers datasets
pip install shap captum einops
pip install pandas numpy scikit-learn matplotlib
pip install monai
pip install wandb
pip install fastapi uvicorn python-multipart  # dashboard backend
```

**`configs/base.yaml`** — never hardcode hyperparameters:

```yaml
model:
  d_model: 512
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  lr: 1e-4
  weight_decay: 1e-2
  epochs: 20
  scheduler: cosine

data:
  image_size: 224
  vitals_seq_len: 48
  text_max_len: 512
  num_labels: 14
```

---

## Data Access — Do This Before Week 1

| Dataset | Access URL | Wait Time | Size |
|---------|-----------|-----------|------|
| MIMIC-III | physionet.org (CITI training required) | 1–2 weeks | ~6 GB compressed |
| CheXpert | stanfordmlgroup.github.io/competitions/chexpert | 24 hours | ~439 GB (use small version first) |

> **Critical:** Add `data/raw/` to `.gitignore` immediately. Never commit patient data. Use DVC for data versioning.

---

---

# Phase 1 — Foundation (Weeks 1–4)

## Weeks 1–2: Literature Review

Read in this order. Write one paragraph per paper as you go — these become your Related Work section.

| Paper | Authors | Key Takeaway for Your Project |
|-------|---------|-------------------------------|
| CheXpert | Irvin et al. (2019) | Your 14-label taxonomy and label uncertainty policy (U-Ones / U-Zeros / U-Ignore) |
| BioViL-T | Bannur et al. (2023) | Closest prior work — understand how cross-modal differs from their approach |
| MIMIC-Extract | Wang et al. (2020) | Follow their vitals extraction pipeline exactly |
| ViT | Dosovitskiy et al. (2020) | Patch embeddings, CLS token, how ViT differs from CNN |
| BioBERT | Lee et al. (2019) | Domain-adapted BERT; use HuggingFace pretrained weights |
| CLIP | Radford et al. (2021) | Intuition for aligning visual and textual modalities |
| Grad-CAM | Selvaraju et al. (2017) | You'll implement this in Phase 4 |
| SHAP | Lundberg & Lee (2017) | Understand DeepSHAP vs TreeSHAP — use DeepSHAP for BioBERT |

> **Tip:** For MIMIC notes, truncate to the **last** 512 tokens — the Assessment & Plan at the end is more clinically relevant than the History at the beginning.

---

## Weeks 3–4: Data Pipeline

### Image Pipeline — `src/data/dicom_to_png.py`

```python
import pydicom, cv2, numpy as np
from pathlib import Path

def dicom_to_png(dicom_path: str, out_path: str, size: int = 224):
    dcm = pydicom.dcmread(dicom_path)
    arr = dcm.pixel_array.astype(np.float32)

    # Window leveling — clip to 2 standard deviations
    arr = np.clip(arr, arr.mean() - 2*arr.std(),
                       arr.mean() + 2*arr.std())

    # Normalize to [0, 255]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr = arr.astype(np.uint8)

    img = cv2.resize(arr, (size, size))
    cv2.imwrite(out_path, img)


def batch_convert(dicom_dir: str, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for p in Path(dicom_dir).rglob("*.dcm"):
        dicom_to_png(str(p), f"{out_dir}/{p.stem}.png")
```

### Text Pipeline — `src/data/note_tokenizer.py`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

def tokenize_note(text: str, max_len: int = 512) -> dict:
    # Truncate from the END (Assessment & Plan is at the end of MIMIC notes)
    return tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
```

### Vitals Pipeline — `src/data/vitals_preprocessor.py`

```python
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

VITALS = ["heart_rate", "sbp", "dbp", "spo2", "resp_rate", "temp"]
SEQ_LEN = 48  # 48 hours of hourly measurements

def preprocess_vitals(df: pd.DataFrame) -> np.ndarray:
    df = df.set_index("charttime").resample("1H").mean()
    df = df.ffill().bfill()
    df = df[VITALS].iloc[:SEQ_LEN]

    # Pad if shorter than SEQ_LEN
    if len(df) < SEQ_LEN:
        pad = pd.DataFrame(0, index=range(SEQ_LEN - len(df)), columns=VITALS)
        df = pd.concat([df, pad])

    scaler = StandardScaler()
    return scaler.fit_transform(df.values)   # shape: (48, 6)
```

> **Warning:** Write pytest unit tests for every preprocessing function. A silent normalization bug corrupts your entire training run. 30 minutes of testing saves 3 days of debugging.

```python
# tests/test_preprocessing.py
def test_vitals_output_shape():
    dummy = pd.DataFrame({col: np.random.rand(30) for col in VITALS},
                          index=pd.date_range("2023-01-01", periods=30, freq="1H"))
    dummy["charttime"] = dummy.index
    result = preprocess_vitals(dummy)
    assert result.shape == (48, 6), f"Expected (48,6), got {result.shape}"
```

---

---

# Phase 2 — Unimodal Baselines (Weeks 5–10)

These three baselines are critical — they prove your fusion model actually adds value.

## Weeks 5–6: Image-Only Baseline

### Encoder — `src/encoders/image_encoder.py`

```python
import timm, torch, torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "vit_base_patch16_224", out_dim: int = 512):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = self.vit.num_features       # 768 for ViT-Base
        self.proj = nn.Linear(embed_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 224, 224)
        features = self.vit(x)           # (B, 768)
        return self.norm(self.proj(features))   # (B, 512)
```

### Training Setup

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

model = ImageEncoder()
head  = nn.Linear(512, 14)  # 14 CheXpert labels

optimizer = optim.AdamW(
    list(model.parameters()) + list(head.parameters()),
    lr=1e-4, weight_decay=1e-2
)
criterion = nn.BCEWithLogitsLoss()  # multi-label — not CrossEntropy
scheduler = CosineAnnealingLR(optimizer, T_max=20)
```

**Target:** AUC > 0.85 on validation set | Batch: 32 | Epochs: 20

> **Tip:** Log with `wandb.log()` every epoch — AUC per label, loss curves, gradient norms. You'll overlay these against the fusion model curves later.

---

## Weeks 7–8: Text-Only Baseline

### Encoder — `src/encoders/text_encoder.py`

```python
from transformers import AutoModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.1", out_dim: int = 512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size   # 768
        self.proj = nn.Linear(hidden, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]    # CLS token: (B, 768)
        return self.norm(self.proj(cls))         # (B, 512)
```

Fine-tune on MIMIC discharge summaries mapped to the same 14 CheXpert labels.

---

## Weeks 9–10: Vitals-Only Baseline

### Encoder — `src/encoders/vitals_encoder.py`

```python
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 256,
                 out_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,       # 6 vital signs
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, out_dim)  # *2 for bidirectional
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 48, 6)
        out, _ = self.lstm(x)
        last = out[:, -1, :]    # (B, 512) — last timestep
        return self.norm(self.proj(last))
```

> **Optional contribution:** Also implement a Temporal Transformer using `nn.TransformerEncoder` with sinusoidal positional encoding. Compare LSTM vs TST in your ablation study — it's a free research comparison.

```python
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=3, out_dim=512):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.input_proj(x)          # (B, 48, d_model)
        x = self.transformer(x)
        return self.norm(self.out_proj(x[:, -1, :]))
```

---

---

# Phase 3 — Fusion Model (Weeks 11–17)

## Weeks 11–13: Projection Layer

### `src/fusion/projection.py`

```python
import torch.nn as nn

class ProjectionHead(nn.Module):
    """Maps any encoder output to shared d=512 space."""
    def __init__(self, in_dim: int, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)

# One projection head per modality — keeps architecture uniform
img_proj = ProjectionHead(in_dim=512)
txt_proj = ProjectionHead(in_dim=512)
vit_proj = ProjectionHead(in_dim=512)
```

---

## Weeks 14–16: Cross-Modal Attention Fusion

### `src/fusion/cross_attention.py`

```python
import torch, torch.nn as nn

class CrossModalAttentionFusion(nn.Module):
    """
    Each modality attends to the other two via nn.MultiheadAttention.
    Outputs a single fused (B, 512) representation.
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        mha = lambda: nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.img_txt_attn = mha()
        self.img_vit_attn = mha()
        self.txt_img_attn = mha()
        self.txt_vit_attn = mha()
        self.vit_img_attn = mha()
        self.vit_txt_attn = mha()

        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, img_emb, txt_emb, vit_emb):
        # Add sequence dim: (B, D) → (B, 1, D)
        img = img_emb.unsqueeze(1)
        txt = txt_emb.unsqueeze(1)
        vit = vit_emb.unsqueeze(1)

        # Each modality attends to the other two (sequentially)
        img_f, _ = self.img_txt_attn(img, txt, txt)
        img_f, _ = self.img_vit_attn(img_f, vit, vit)

        txt_f, _ = self.txt_img_attn(txt, img, img)
        txt_f, _ = self.txt_vit_attn(txt_f, vit, vit)

        vit_f, _ = self.vit_img_attn(vit, img, img)
        vit_f, _ = self.vit_txt_attn(vit_f, txt, txt)

        # Concatenate and project back to d=512
        combined = torch.cat([img_f.squeeze(1), txt_f.squeeze(1), vit_f.squeeze(1)], dim=-1)
        return self.norm(self.fusion_proj(combined))   # (B, 512)
```

### Early Fusion Baseline (implement for comparison)

```python
class EarlyFusionBaseline(nn.Module):
    """Simple concatenate → linear. Compare against cross-attention."""
    def __init__(self, d_model=512, num_labels=14):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_labels)
        )

    def forward(self, img_emb, txt_emb, vit_emb):
        return self.classifier(torch.cat([img_emb, txt_emb, vit_emb], dim=-1))
```

> **Key insight:** Implementing both early fusion and cross-attention, then comparing them, is itself a research contribution. The comparison table goes in your paper.

---

## Full Model — `src/models/multimodal_model.py`

```python
class MultimodalClinicalModel(nn.Module):
    def __init__(self, num_labels: int = 14):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.txt_encoder = TextEncoder()
        self.vit_encoder = TemporalEncoder()
        self.img_proj    = ProjectionHead(512)
        self.txt_proj    = ProjectionHead(512)
        self.vit_proj    = ProjectionHead(512)
        self.fusion      = CrossModalAttentionFusion()
        self.classifier  = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_labels)
        )

    def forward(self, image=None, input_ids=None,
                attention_mask=None, vitals=None):
        """
        Modality-gated: pass None for any missing modality.
        Used for ablation studies — same model, different inputs.
        """
        embs = []
        if image is not None:
            embs.append(self.img_proj(self.img_encoder(image)))
        if input_ids is not None:
            embs.append(self.txt_proj(self.txt_encoder(input_ids, attention_mask)))
        if vitals is not None:
            embs.append(self.vit_proj(self.vit_encoder(vitals)))

        if len(embs) == 3:
            fused = self.fusion(*embs)
        else:
            fused = torch.cat(embs, dim=-1)   # fallback for ablations

        return self.classifier(fused)   # (B, 14) logits
```

---

## Week 17: Evaluation

### `src/evaluate.py`

```python
from sklearn.metrics import roc_auc_score, f1_score
import torch, numpy as np

LABEL_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_CM", "Fracture", "Lung_Lesion", "Lung_Opacity",
    "No_Finding", "Pleural_Effusion", "Pleural_Other",
    "Pneumonia", "Pneumothorax", "Support_Devices"
]

def evaluate(model, dataloader, device) -> dict:
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(
                image=batch["image"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                vitals=batch["vitals"].to(device)
            )
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(batch["labels"].numpy())

    y_true = np.concatenate(all_labels)   # (N, 14)
    y_pred = np.concatenate(all_preds)    # (N, 14)

    results = {}
    for i, name in enumerate(LABEL_NAMES):
        results[name] = round(roc_auc_score(y_true[:, i], y_pred[:, i]), 4)

    results["macro_AUC"] = round(np.mean(list(results.values())), 4)
    return results
```

---

---

# Phase 4 — Explainability (Weeks 18–21)

## Weeks 18–19: Grad-CAM on ViT

### `src/xai/gradcam.py`

```python
import torch, cv2, numpy as np
from captum.attr import LayerGradCam   # easier than manual hooks for ViT

class GradCAMViT:
    def __init__(self, model: ImageEncoder):
        self.model = model
        # Target the last transformer block's norm layer
        target_layer = model.vit.blocks[-1].norm1
        self.gradcam = LayerGradCam(model, target_layer)

    def generate(self, image: torch.Tensor, class_idx: int) -> np.ndarray:
        """Returns a (224, 224) heatmap normalized to [0, 1]."""
        attributions = self.gradcam.attribute(image, target=class_idx)
        cam = attributions.squeeze().mean(dim=0).numpy()  # (14, 14) patches
        cam = np.maximum(cam, 0)   # ReLU
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (224, 224))


def overlay_heatmap(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original X-ray image."""
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
```

---

## Week 20: SHAP on BioBERT

### `src/xai/shap_explainer.py`

```python
import shap, torch

def explain_text_prediction(text_encoder, classifier_head,
                             tokenizer, text: str, class_idx: int):
    """
    Returns SHAP values showing which tokens drove the prediction.
    Visualize with shap.plots.text(shap_values) in a notebook.
    """
    def predict_fn(texts: list) -> np.ndarray:
        tokens = tokenizer(texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            emb = text_encoder(**tokens)
            logits = classifier_head(emb)
        return torch.sigmoid(logits)[:, class_idx].numpy()

    explainer = shap.Explainer(predict_fn, tokenizer)
    shap_values = explainer([text])

    # In a Jupyter notebook:
    # shap.plots.text(shap_values)   # color-coded token attribution HTML

    return shap_values
```

---

## Week 21: Attention Rollout for Vitals

### `src/xai/attention_rollout.py`

```python
import torch, numpy as np
import matplotlib.pyplot as plt

def get_vitals_importance(model, vitals_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract per-timestep importance from LSTM hidden states.
    Returns a (48,) array normalized to [0, 1].
    """
    model.eval()
    with torch.no_grad():
        out, _ = model.vit_encoder.lstm(vitals_tensor)  # (B, 48, 512)
        importance = torch.norm(out, dim=-1).squeeze().numpy()  # (48,)
        return (importance - importance.min()) / (importance.max() - importance.min())


def plot_vitals_attention(importance: np.ndarray, vitals: np.ndarray,
                          vital_names: list = None) -> plt.Figure:
    """
    Two-panel plot: top = timestep importance bars,
    bottom = raw vital sign traces.
    """
    if vital_names is None:
        vital_names = ["HR", "SBP", "DBP", "SpO2", "RR", "Temp"]

    hours = np.arange(len(importance))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.bar(hours, importance, color="steelblue", alpha=0.75)
    ax1.set_ylabel("Attention weight")
    ax1.set_title("Timestep importance (attention rollout)")

    for i, name in enumerate(vital_names):
        ax2.plot(hours, vitals[:, i], label=name, linewidth=1.2)
    ax2.set_xlabel("Hour (0–47)")
    ax2.set_ylabel("Normalized value")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    return fig
```

---

---

# Phase 5 — Analysis & Writing (Weeks 22–24)

## Week 22: Ablation Studies

Run all 8 configurations. Use a CLI flag to control modalities — same model, same hyperparameters, only inputs change.

```bash
# Training CLI
python src/train.py --modalities img           # image only
python src/train.py --modalities txt           # text only
python src/train.py --modalities vit           # vitals only
python src/train.py --modalities img,txt       # image + text
python src/train.py --modalities img,vit       # image + vitals
python src/train.py --modalities txt,vit       # text + vitals
python src/train.py --modalities img,txt,vit --fusion early
python src/train.py --modalities img,txt,vit --fusion cross_attention
```

### Ablation Results Table (template for your paper)

| Configuration | Fusion | Macro AUC | Pneumonia AUC | Edema AUC | Pleural Eff. AUC |
|---------------|--------|-----------|---------------|-----------|-----------------|
| Image only | — | | | | |
| Text only | — | | | | |
| Vitals only | — | | | | |
| Image + Text | — | | | | |
| Image + Vitals | — | | | | |
| Text + Vitals | — | | | | |
| All modalities | Early | | | | |
| **All modalities** | **Cross-attn** | | | | |

---

## Week 23: Error Analysis

```python
def find_hard_failures(model, dataloader, label_idx: int, top_k: int = 50):
    """Find highest-confidence wrong predictions for a given label."""
    failures = []
    for batch in dataloader:
        with torch.no_grad():
            probs = torch.sigmoid(model(**batch))[:, label_idx]
        true_labels = batch["labels"][:, label_idx]
        for i, (p, t) in enumerate(zip(probs, true_labels)):
            if round(p.item()) != t.item():
                failures.append({"prob": p.item(), "true": t.item(),
                                  "patient_id": batch["id"][i]})
    return sorted(failures, key=lambda x: abs(x["prob"] - 0.5), reverse=True)[:top_k]
```

**Stratify AUC by:**
- Note length (short < 200 tokens vs long > 400 tokens)
- Vitals missingness rate (< 10% missing vs > 30% missing)
- Image quality (CheXpert quality label)

Document systematic failure modes in your Limitations section.

---

## Week 24: Paper Structure

1. **Abstract** (150 words): problem → method → key result (e.g. "+4.2 macro-AUC over best unimodal baseline")
2. **Introduction**: clinical motivation, multimodal gap, 3 bullet-point contributions
3. **Related Work**: use your week 1–2 paragraph notes
4. **Methods**: architecture diagram, each encoder, fusion, training details
5. **Experiments**: datasets, metrics, baselines, ablation table, XAI figures
6. **Discussion**: what works, failure modes, clinical implications
7. **Conclusion + Future Work**

---

---

# Dashboard — UI/UX Design

## Architecture

```
Browser (React + TypeScript)
        │
        │  REST (JSON)
        ▼
FastAPI backend (Python)
        │
        ├── ImageEncoder  ──── ViT weights
        ├── TextEncoder   ──── BioBERT weights
        ├── VitalsEncoder ──── LSTM weights
        └── XAI layer     ──── Grad-CAM · SHAP · Rollout
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| Input panel | Upload DICOM/PNG, paste clinical notes, enter vitals CSV or manual values |
| Predictions | 14-label probability bars, confidence scores, color-coded severity |
| Grad-CAM viewer | X-ray with heatmap overlay, toggle by diagnosis label, opacity slider |
| SHAP text view | Color-highlighted note tokens — red = pushes prediction up, blue = down |
| Vitals timeline | 48-hour vitals chart with attention rollout bars overlaid as background |
| Audit log | All predictions with timestamps, downloadable as PDF |

---

## FastAPI Backend — `dashboard/api/main.py`

```python
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch, io, base64, cv2

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = MultimodalClinicalModel()
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location="cpu"))
model.eval()

@app.post("/predict")
async def predict(
    image: UploadFile = File(None),
    note: str = Form(""),
    vitals_csv: UploadFile = File(None)
):
    # 1. Preprocess each provided modality
    img_tensor = preprocess_image(await image.read()) if image else None
    txt_tokens = tokenize_note(note) if note else (None, None)
    vit_tensor = preprocess_vitals_csv(await vitals_csv.read()) if vitals_csv else None

    # 2. Forward pass
    with torch.no_grad():
        logits = model(image=img_tensor,
                       input_ids=txt_tokens[0],
                       attention_mask=txt_tokens[1],
                       vitals=vit_tensor)
        probs = torch.sigmoid(logits).squeeze().tolist()

    # 3. XAI
    gradcam_b64 = generate_gradcam_b64(img_tensor) if img_tensor is not None else None
    shap_vals   = compute_shap(note) if note else None

    return {
        "predictions": dict(zip(LABEL_NAMES, probs)),
        "gradcam_image": gradcam_b64,
        "shap_values": shap_vals,
        "missing_modalities": [m for m, v in
            [("image", img_tensor), ("notes", note), ("vitals", vit_tensor)] if not v]
    }

# Run: uvicorn main:app --reload --port 8000
```

---

## React Scaffold

```bash
npx create-react-app dashboard --template typescript
cd dashboard
npm install recharts axios @tanstack/react-query
npm install tailwindcss @tailwindcss/forms
npm install lucide-react
```

### Color Design Tokens — `dashboard/src/styles/tokens.css`

```css
:root {
  --brand-primary:   #185FA5;  /* CheXpert blue */
  --severity-high:   #D85A30;  /* Positive finding — red-orange */
  --severity-medium: #BA7517;  /* Uncertain — amber */
  --severity-low:    #1D9E75;  /* Negative / normal — teal */
  --surface-1:       #FFFFFF;
  --surface-2:       #F8F9FA;
  --border:          #E2E8F0;
  --text-primary:    #1A202C;
  --text-muted:      #718096;
}
```

### Prediction Bar Component — `dashboard/src/components/PredictionBar.tsx`

```tsx
interface PredictionBarProps {
  label: string;
  probability: number;   // 0–1
}

export const PredictionBar = ({ label, probability }: PredictionBarProps) => {
  const color = probability > 0.7
    ? "var(--severity-high)"
    : probability > 0.4
    ? "var(--severity-medium)"
    : "var(--severity-low)";

  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontSize: 13, marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ fontWeight: 500, color }}>{(probability * 100).toFixed(1)}%</span>
      </div>
      <div style={{ height: 6, background: "#E2E8F0", borderRadius: 3 }}>
        <div style={{ height: "100%", width: `${probability * 100}%`,
                      background: color, borderRadius: 3,
                      transition: "width 0.4s ease" }} />
      </div>
    </div>
  );
};
```

---

## Clinical UX Rules

1. **Never auto-submit.** All predictions require an explicit "Run Analysis" button. Clinical tools must not silently process patient data.
2. **Uncertainty is first-class.** Always show probability bars — never a binary yes/no label.
3. **Explainability always visible.** Grad-CAM, SHAP, and attention rollout tabs are one click from the prediction — not buried in settings.
4. **Missing modality warning.** If vitals are not provided, show a yellow banner: "Prediction accuracy may be reduced — vitals not provided."
5. **Disclaimer always visible.** Fixed footer: *"For research purposes only. Not for clinical decision-making."*
6. **Loading states for all async ops.** Inference takes 2–5 seconds — show a spinner with "Analyzing..." messaging.

---

---

# Training Pipeline — `src/train.py`

```python
import argparse, torch, wandb
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modalities", type=str, default="img,txt,vit")
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["early", "cross_attention"])
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()

def train():
    args = parse_args()
    cfg = load_yaml(args.config)
    modalities = set(args.modalities.split(","))

    wandb.init(project="multimodal-clinical-ai",
               name=f"{args.modalities}__{args.fusion}",
               config={**cfg, "modalities": args.modalities, "fusion": args.fusion})

    model = MultimodalClinicalModel(num_labels=14).to(cfg["device"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            # Zero-out modalities not in this run (for ablations)
            image = batch["image"].to(cfg["device"]) if "img" in modalities else None
            input_ids = batch["input_ids"].to(cfg["device"]) if "txt" in modalities else None
            attn_mask = batch["attention_mask"].to(cfg["device"]) if "txt" in modalities else None
            vitals = batch["vitals"].to(cfg["device"]) if "vit" in modalities else None

            logits = model(image, input_ids, attn_mask, vitals)
            loss = criterion(logits, batch["labels"].to(cfg["device"]))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluate and log
        metrics = evaluate(model, val_loader, cfg["device"])
        wandb.log({"epoch": epoch, "loss": loss.item(), **metrics})
        print(f"Epoch {epoch+1}: Macro AUC = {metrics['macro_AUC']}")

if __name__ == "__main__":
    train()
```

---

---

# Checkpoints & Experiment Tracking

```
checkpoints/
├── img_only/                    # Phase 2 baselines
│   └── best_model.pt
├── txt_only/
│   └── best_model.pt
├── vit_only/
│   └── best_model.pt
├── img_txt__early/              # Phase 3 ablations
├── img_txt__cross_attention/
├── all__early/
└── all__cross_attention/        # Your headline model
    ├── best_model.pt
    └── config.yaml
```

Save checkpoints with:

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "macro_auc": metrics["macro_AUC"],
    "config": cfg
}, f"checkpoints/{run_name}/best_model.pt")
```

---

---

# 14 CheXpert Labels Reference

| Label | Prevalence (CheXpert) | Clinical Notes |
|-------|-----------------------|----------------|
| Atelectasis | 29.0% | Lung collapse — common post-op |
| Cardiomegaly | 23.3% | Enlarged heart shadow |
| Consolidation | 4.8% | Airspace filled (pneumonia, hemorrhage) |
| Edema | 48.9% | Fluid in lungs — common in CHF |
| Enlarged Cardiomediastinum | 26.4% | Widened mediastinum |
| Fracture | 4.0% | Rib/clavicle fractures |
| Lung Lesion | 4.3% | Mass or nodule |
| Lung Opacity | 26.0% | Non-specific opacity |
| No Finding | 22.0% | Normal study |
| Pleural Effusion | 74.6% | Fluid in pleural space |
| Pleural Other | 2.9% | Thickening, calcification |
| Pneumonia | 2.2% | Infectious consolidation |
| Pneumothorax | 17.9% | Air in pleural space |
| Support Devices | 66.0% | ET tubes, pacemakers, etc. |

> **Note on label uncertainty:** CheXpert uses U-Ones policy for Atelectasis and Edema (treat uncertain as positive) and U-Zeros for the remaining labels as a simple starting policy. This is a hyperparameter — ablate it.

---

*Guide covers 24 weeks of implementation. See interactive dashboard in VS Code for phase navigation.*
