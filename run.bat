@echo off
title IoT Smart Irrigation Simulator

echo =========================================
echo   IoT Smart Irrigation Simulator
echo =========================================
echo.

:: Check for Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    set PY=python
    goto :found
)
where python3 >nul 2>nul
if %errorlevel% equ 0 (
    set PY=python3
    goto :found
)

echo [ERROR] Python not found!
echo Install it from https://www.python.org/downloads/
echo Make sure to check "Add Python to PATH" during installation.
echo.
pause
exit /b 1

:found
echo [OK] Found Python:
%PY% --version

:: Install dependencies
echo.
echo Installing dependencies...
%PY% -m pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo Retrying with --user flag...
    %PY% -m pip install -r requirements.txt --quiet --user
)

echo.
echo Starting the simulator...
echo    Open http://localhost:8501 in your browser
echo    Press Ctrl+C to stop
echo.

%PY% -m streamlit run app.py

pause
