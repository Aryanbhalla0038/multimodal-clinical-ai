<#
.SYNOPSIS
  Single-command launcher: starts FastAPI backend + Vite frontend.

.DESCRIPTION
  - Backend (port 9001) runs in a new PowerShell window so it stays alive after this script returns.
  - Frontend (port 5173) runs in the current window with live tail.
  - Press Ctrl+C in this window to stop the frontend; close the backend window to stop the backend.

.EXAMPLE
  .\run.ps1
#>
param(
    [string]$BackendPort = "9001",
    # Precedence: explicit -Ckpt > $env:CKPT_PATH > default
    [string]$Ckpt = $(if ($env:CKPT_PATH) { $env:CKPT_PATH } else { "checkpoints/all__cross_attention/best_model.pt" })
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo

# ---- Sanity checks ---------------------------------------------------------
if (-not (Test-Path $Ckpt)) {
    Write-Error "Checkpoint not found at: $Ckpt`nDownload best_model.pt from Kaggle and place it there."
}
if (-not (Test-Path "dashboard\node_modules")) {
    Write-Host "[run] First-time frontend deps missing. Running 'npm install' in dashboard/..." -ForegroundColor Yellow
    Push-Location dashboard
    npm install
    Pop-Location
}

# ---- Start backend in a NEW PowerShell window ------------------------------
$backendCmd = @"
Set-Location '$repo'
`$env:CKPT_PATH = '$Ckpt'
Write-Host '[backend] starting on port $BackendPort ...' -ForegroundColor Cyan
python -m uvicorn dashboard.api.main:app --port $BackendPort --host 127.0.0.1
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WindowStyle Normal

Write-Host "[run] Backend launched in separate window. Waiting for /health to come up ..." -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $r = Invoke-WebRequest "http://127.0.0.1:$BackendPort/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch {}
}
if (-not $ok) {
    Write-Warning "[run] Backend did not respond within 60 s. Frontend will still start, but /predict may fail."
} else {
    Write-Host "[run] Backend is healthy." -ForegroundColor Green
}

# ---- Start frontend in the CURRENT window ---------------------------------
Write-Host "[run] Starting Vite dev server (http://localhost:5173) ..." -ForegroundColor Cyan
Push-Location dashboard
try {
    npm run dev
} finally {
    Pop-Location
}
