# Multimodal Clinical AI

ViT + BioBERT + Temporal Encoder with Cross-Modal Attention Fusion for 14-label CheXpert chest X-ray diagnosis. Implements Grad-CAM, SHAP, and attention rollout for explainability, plus a FastAPI + React dashboard.

See `multimodal_clinical_ai_guide.md` for the full 24-week roadmap.

## Run the whole project (single command)

### Prerequisites
- **Python 3.10+** on PATH
- **Node.js 18+** on PATH (for the React dashboard) — install from https://nodejs.org/
- **Trained checkpoint** at `checkpoints/all__cross_attention/best_model.pt`. Download from the [v0.1.0 Release](https://github.com/Aryanbhalla0038/multimodal-clinical-ai/releases/tag/v0.1.0) (786 MB) or train your own via `notebooks/kaggle_train.ipynb`.
- **Optional image-only checkpoint** at `checkpoints/image_only_pure/best_model.pt`. Download from the [v0.2.0 Release](https://github.com/Aryanbhalla0038/multimodal-clinical-ai/releases/tag/v0.2.0) (327 MB).

### One-time setup
```powershell
.\setup.ps1
```
This creates `.venv`, installs Python + Node deps, and runs the pytest smoke suite. Takes ~3 min.

### Launch backend + frontend with one command
```powershell
.\run.ps1
```
Behavior:
- Spawns a **new PowerShell window** running the FastAPI backend on http://127.0.0.1:9001
- Waits up to 60 s for `/health` to come up, then prints `Backend is healthy.`
- Starts the Vite dev server on http://localhost:5173 in the current window
- Open http://localhost:5173 in your browser → upload an X-ray, paste a clinical note, click **Run Analysis**

To run the pure image-only checkpoint instead:
```powershell
$env:CKPT_PATH = "checkpoints/image_only_pure/best_model.pt"
.\run.ps1
```

### Stopping everything
- Press **Ctrl+C** in the current window to stop the frontend
- Close the backend's separate PowerShell window

### Manual two-terminal alternative

If you'd rather see both logs separately:

**Terminal 1 (backend):**
```powershell
$env:CKPT_PATH = "checkpoints/all__cross_attention/best_model.pt"
python -m uvicorn dashboard.api.main:app --port 9001 --host 127.0.0.1
```

**Terminal 2 (frontend):**
```powershell
cd dashboard
npm run dev
```

## Free deployment (frontend + backend together)

Deploy as a **Hugging Face Docker Space** (single container):

1. Create a new Space on Hugging Face with SDK = **Docker**.
2. Push this repo (contains `Dockerfile`) to the Space repository.
3. In Space settings, set hardware = **CPU Basic** (free).
4. Set these environment variables (optional overrides):
   - `MODEL_KIND=image_only`
   - `CKPT_PATH=checkpoints/image_only_pure/best_model.pt`
   - `CKPT_URL=https://github.com/Aryanbhalla0038/multimodal-clinical-ai/releases/download/v0.2.0/best_model_image_only_pure.pt`

Notes:
- The backend serves the built React app from `dashboard/dist`, so one URL hosts both UI and API.
- API is available at `/api/*` (`/api/health`, `/api/predict`, `/api/labels`).
- First boot may take time while the checkpoint downloads.

## Train on Kaggle GPU (real CheXpert)

Open `notebooks/kaggle_train.ipynb` on Kaggle, attach the **CheXpert** dataset (`ashery/chexpert` or the official one), enable GPU T4 x2, and **Save Version → Save & Run All (Commit)**. Wait ~3 hours, then download `best_model.pt` from the Output tab into `checkpoints/all__cross_attention/`.

## Run the unit tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/ -v
```
60 tests, runs on CPU in ~30 s.

## Structure

```
src/
  data/        # preprocessing (image, text, vitals) + dataset
  encoders/    # ViT, BioBERT, LSTM/Transformer
  fusion/      # projection + cross-modal attention
  models/      # full multimodal model
  xai/         # Grad-CAM, SHAP, attention rollout
  utils/       # config, labels
  train.py
  evaluate.py
dashboard/     # FastAPI backend + React frontend
configs/       # YAML experiment configs
tests/         # pytest suite
notebooks/     # Kaggle training notebook
```

## Modalities

| Modality | Source | Encoder |
|----------|--------|---------|
| Chest X-ray | CheXpert / MIMIC-CXR | ViT-Base (timm, pretrained) |
| Clinical note | MIMIC-III discharge summaries | BioBERT (`dmis-lab/biobert-base-cased-v1.1`) |
| Vitals (HR/SBP/DBP/SpO2/RR/Temp × 48h) | MIMIC-III | BiLSTM (or Transformer) |

Fusion: each modality is projected to `d=512`, pairwise cross-attention, concatenated, then linear → 14-label sigmoid.

## Preliminary results (synthetic notes)

3-config ablation on 20 k CheXpert subset, 5 epochs, BS 16, lr 1e-4, AMP. Full report: [`results/ablation/RESULTS.md`](results/ablation/RESULTS.md).

| Config | Best Macro AUC | Best Epoch | Time |
|---|---|---|---|
| `img_only`  | 0.5069 (≈ chance) | 4 | 17.2 min |
| `txt_only`  | 0.9983 | 1 | 22.5 min |
| `all_xattn` | 0.9986 | 1 | 40.6 min |

Two findings worth noting:
1. **Synthetic notes leak labels** — text-using configs saturate at ~1.0 AUC at epoch 1.
2. **Training instability** — both text configs collapse from ~1.0 → ~0.5 by epoch 5 (AMP + high-LR + no warmup). The deployed checkpoint is from epoch 1, near the peak.
3. **Image branch under-trained** — 5 × 20 k samples is too few to fine-tune ViT-Base; needs lr 3e-5 + warmup + more epochs.

These limitations will be addressed when real MIMIC-III notes/vitals replace the synthetic stand-in (pending PhysioNet DUA).

**Disclaimer:** For research only. Not for clinical decision-making.
