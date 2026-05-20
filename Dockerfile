FROM node:20-bookworm-slim AS frontend-build
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    MODEL_KIND=image_only \
    CKPT_PATH=checkpoints/image_only_pure/best_model.pt \
    CKPT_URL=https://github.com/Aryanbhalla0038/multimodal-clinical-ai/releases/download/v0.2.0/best_model_image_only_pure.pt

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-build /app/dashboard/dist ./dashboard/dist

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "dashboard.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
