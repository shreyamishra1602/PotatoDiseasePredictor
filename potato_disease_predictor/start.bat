@echo off
title Potato Disease Predictor
echo ================================================
echo   POTATO DISEASE PREDICTOR
echo ================================================
echo.

:: Auto-setup if first run
if not exist ".venv" (
    echo First run detected! Setting up...
    echo.
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Python not found!
        echo Download from: https://www.python.org/downloads/
        echo Check "Add Python to PATH" during install!
        pause
        exit /b 1
    )
    echo Creating virtual environment...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    echo Installing dependencies (first time takes a few minutes)...
    pip install --upgrade pip -q
    pip install -r requirements_windows.txt
    echo.
    echo Setup complete!
    echo.
) else (
    call .venv\Scripts\activate.bat
)

echo Starting dashboard... (browser will open automatically)
echo Press Ctrl+C to stop.
echo.
streamlit run app_combined.py --server.headless false
