param(
    [string]$Port = "8501"
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectDir

Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   ResearchMind AI - Launch Script" -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Check / create .env ────────────────────────────────────────

if (-not (Test-Path ".env")) {
    Write-Host "[1/4] No .env file found. Let's set one up." -ForegroundColor Yellow
    $apiKey = Read-Host "Paste your Gemini API key (get one free at https://aistudio.google.com/app/apikey)"
    if ([string]::IsNullOrWhiteSpace($apiKey)) {
        Write-Host "ERROR: API key cannot be empty." -ForegroundColor Red
        exit 1
    }
    Set-Content -Path ".env" -Value "GEMINI_API_KEY=$apiKey"
    Write-Host "   .env file created." -ForegroundColor Green
} else {
    Write-Host "[1/4] .env file found." -ForegroundColor Green
}

# ── Step 2: Create virtual environment ─────────────────────────────────

if (-not (Test-Path ".venv")) {
    Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    if (-not $?) { Write-Host "ERROR: Failed to create venv." -ForegroundColor Red; exit 1 }
    Write-Host "   Done." -ForegroundColor Green
} else {
    Write-Host "[2/4] Virtual environment found." -ForegroundColor Green
}

# ── Step 3: Install dependencies ───────────────────────────────────────

Write-Host "[3/4] Checking dependencies..." -ForegroundColor Yellow
$pip = Join-Path ".venv" "Scripts\pip.exe"
& $pip install -q -r requirements.txt
if (-not $?) { Write-Host "ERROR: Failed to install dependencies." -ForegroundColor Red; exit 1 }
Write-Host "   Dependencies ready." -ForegroundColor Green

# ── Step 4: Launch the app ─────────────────────────────────────────────

Write-Host "[4/4] Launching ResearchMind AI..." -ForegroundColor Yellow

# Kill any lingering streamlit processes on our port
$existing = netstat -ano | Select-String ":$Port "
if ($existing) {
    Write-Host "   Port $Port is in use. Cleaning up..." -ForegroundColor Yellow
    Get-Process -Name "streamlit" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "   Opening http://localhost:$Port in your browser..." -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to stop the app." -ForegroundColor Cyan
Write-Host ""

$streamlit = Join-Path ".venv" "Scripts\streamlit.exe"
$env:STREAMLIT_CONSOLE_EMAIL = ""
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
Start-Process "http://localhost:$Port"
& $streamlit run app.py --server.port $Port --server.headless false
