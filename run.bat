@echo off
cd /d "%~dp0"

echo =======================================
echo    ResearchMind AI - Launcher
echo =======================================
echo.

:: Step 1: Check .env
if not exist .env (
    echo [1/4] No .env file found. Creating one...
    set /p key="Paste your Gemini API key (get one free at https://aistudio.google.com/app/apikey): "
    echo GEMINI_API_KEY=%key%> .env
    echo    .env created.
) else (
    echo [1/4] .env file found.
)

:: Step 2: Create venv if needed
if not exist .venv (
    echo [2/4] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo    Done.
) else (
    echo [2/4] Virtual environment found.
)

:: Step 3: Install dependencies
echo [3/4] Installing dependencies (first time only)...
.venv\Scripts\pip.exe install -q -r requirements.txt
echo    Dependencies ready.

:: Step 4: Launch
echo [4/4] Launching ResearchMind AI...
echo.
echo    Opening http://localhost:8501 in your browser...
echo    Press Ctrl+C to stop the app.
echo.

set STREAMLIT_CONSOLE_EMAIL=
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
start http://localhost:8501
.venv\Scripts\streamlit run app.py --server.port 8501 --server.headless false

pause
