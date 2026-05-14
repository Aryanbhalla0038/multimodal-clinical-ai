# Dashboard

## Run locally

Terminal 1 — FastAPI backend (from repo root):

```bash
# Optional: point to a trained checkpoint
$env:CKPT_PATH = "checkpoints/all__cross_attention/best_model.pt"  # PowerShell

uvicorn dashboard.api.main:app --reload --port 8000
```

Terminal 2 — React frontend:

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:5173
```

Vite proxies `/api/*` → `http://localhost:8000/*`, so no CORS headache.

## Demo mode (no checkpoint)

If `CKPT_PATH` doesn't exist, the API still serves random-weight predictions so
you can validate the UI end-to-end before training finishes.
