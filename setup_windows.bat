@echo off
title Potato Disease Predictor - Setup
echo ================================================
echo   POTATO DISEASE PREDICTOR - WINDOWS SETUP
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found!
    echo.
    echo Download Python from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during install!
    echo.
    pause
    exit /b 1
)

echo Python found!
echo.

:: Create venv
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    echo Done!
) else (
    echo Virtual environment already exists.
)

:: Activate and install
echo.
echo Installing dependencies (this takes a few minutes first time)...
echo.
call .venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r requirements_windows.txt -q

echo.
echo ================================================
echo   SETUP COMPLETE!
echo ================================================
echo.
echo Now double-click 'start.bat' to launch the dashboard.
echo.
pause
