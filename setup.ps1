<#
.SYNOPSIS
  One-time setup: creates venv, installs Python + Node deps.

.EXAMPLE
  .\setup.ps1
#>
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

# ---- Python venv -----------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "[setup] Creating virtual environment .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}
& .\.venv\Scripts\Activate.ps1
Write-Host "[setup] Installing Python dependencies ..." -ForegroundColor Cyan
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# ---- Node deps -------------------------------------------------------------
Write-Host "[setup] Installing frontend dependencies ..." -ForegroundColor Cyan
Push-Location dashboard
npm install
Pop-Location

# ---- Smoke test ------------------------------------------------------------
Write-Host "[setup] Running pytest smoke tests ..." -ForegroundColor Cyan
pytest tests/ -q --tb=short

Write-Host ""
Write-Host "[setup] DONE." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. Place best_model.pt at:  checkpoints/all__cross_attention/best_model.pt"
Write-Host "  2. Run:                     .\run.ps1"
