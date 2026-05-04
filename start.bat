@echo off
:: -------------------------------------------------------
:: start.bat -- BIS Standards Finder: Setup & Inference
:: -------------------------------------------------------

set VENV_DIR=venv
set REQUIREMENTS=requirements.txt
set INPUT=hidden_private_dataset.json
set OUTPUT=team_results.json

:: -------------------------------------------------------
:: 1. Find Python 3.11 specifically
:: -------------------------------------------------------
echo [1/4] Looking for Python 3.11...

set PYTHON311=

:: Try the Python launcher first (py -3.11)
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON311=py -3.11
    goto :found
)

:: Try python3.11 directly
python3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON311=python3.11
    goto :found
)

:: Try python and check if it is 3.11
python --version >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=2" %%V in ('python --version 2^>^&1') do set RAW_VER=%%V
    for /f "tokens=1,2 delims=." %%A in ("%RAW_VER%") do set MAJ=%%A& set MIN=%%B
    if "%MAJ%"=="3" if "%MIN%"=="11" (
        set PYTHON311=python
        goto :found
    )
)

echo.
echo ERROR: Python 3.11 not found.
echo.
echo Your system has a different Python version installed.
echo Please install Python 3.11 from:
echo   https://www.python.org/downloads/release/python-3119/
echo During install check "Add Python to PATH" and "Use py launcher".
echo Then re-run this script.
echo.
exit /b 1

:found
for /f "tokens=2" %%V in ('%PYTHON311% --version 2^>^&1') do set FOUND_VER=%%V
echo   Found Python %FOUND_VER% via: %PYTHON311%

:: -------------------------------------------------------
:: 2. Create virtual environment with Python 3.11
:: -------------------------------------------------------
echo [2/4] Creating virtual environment...

if exist "%VENV_DIR%\" (
    echo   Removing old virtual environment...
    rmdir /s /q %VENV_DIR%
)

%PYTHON311% -m venv %VENV_DIR%
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    exit /b 1
)
echo   Virtual environment created at .\%VENV_DIR%

:: -------------------------------------------------------
:: 3. Activate venv and install dependencies
:: -------------------------------------------------------
echo [3/4] Installing dependencies...

call %VENV_DIR%\Scripts\activate.bat

python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    exit /b 1
)

python -m pip install -r %REQUIREMENTS%
if errorlevel 1 (
    echo ERROR: Failed to install dependencies. Check requirements.txt and your internet connection.
    exit /b 1
)

echo   Dependencies installed successfully.

:: -------------------------------------------------------
:: 4. Run inference
:: -------------------------------------------------------
echo [4/4] Running inference...

python inference.py --input %INPUT% --output %OUTPUT%
if errorlevel 1 (
    echo ERROR: Inference failed. Check the error above.
    exit /b 1
)

echo.
echo Done. Results saved to %OUTPUT%