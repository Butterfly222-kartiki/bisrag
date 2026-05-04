@echo off
:: -------------------------------------------------------
:: start.bat -- BIS Standards Finder: Setup & Inference
:: -------------------------------------------------------

set VENV_DIR=venv
set REQUIREMENTS=requirements.txt
set INPUT=hidden_private_dataset.json
set OUTPUT=team_results.json

:: -------------------------------------------------------
:: 1. Check Python 3.11
:: -------------------------------------------------------
echo [1/4] Checking Python version...

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11 and ensure it is on your PATH.
    exit /b 1
)

for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYTHON_VERSION=%%V
echo   Found Python %PYTHON_VERSION%

:: -------------------------------------------------------
:: 2. Create virtual environment
:: -------------------------------------------------------
echo [2/4] Creating virtual environment...

if not exist "%VENV_DIR%\" (
    python -m venv %VENV_DIR%
    echo   Virtual environment created at .\%VENV_DIR%
) else (
    echo   Virtual environment already exists, skipping creation.
)

:: -------------------------------------------------------
:: 3. Activate venv and install dependencies
:: -------------------------------------------------------
echo [3/4] Installing dependencies...

call %VENV_DIR%\Scripts\activate.bat

python -m pip install --upgrade pip --quiet
python -m pip install -r %REQUIREMENTS% --quiet

echo   Dependencies installed.

:: -------------------------------------------------------
:: 4. Run inference
:: -------------------------------------------------------
echo [4/4] Running inference...

python inference.py --input %INPUT% --output %OUTPUT%

echo.
echo Done. Results saved to %OUTPUT%
