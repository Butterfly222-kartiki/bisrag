#!/usr/bin/env bash
# -------------------------------------------------------
# start.sh — BIS Standards Finder: Setup & Inference
# -------------------------------------------------------

set -euo pipefail

PYTHON=python3.11
VENV_DIR=venv
REQUIREMENTS=requirements.txt
INPUT=hidden_private_dataset.json
OUTPUT=team_results.json

# -------------------------------------------------------
# 1. Check Python 3.11
# -------------------------------------------------------
echo "[1/4] Checking Python version..."

if ! command -v "$PYTHON" &> /dev/null; then
  echo "ERROR: $PYTHON not found. Install Python 3.11 and ensure it is on your PATH."
  exit 1
fi

PYTHON_VERSION=$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
  echo "ERROR: Expected Python 3.11, found Python $PYTHON_VERSION."
  exit 1
fi

echo "  Python $PYTHON_VERSION found."

# -------------------------------------------------------
# 2. Create virtual environment
# -------------------------------------------------------
echo "[2/4] Creating virtual environment..."

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON" -m venv "$VENV_DIR"
  echo "  Virtual environment created at ./$VENV_DIR"
else
  echo "  Virtual environment already exists, skipping creation."
fi

# -------------------------------------------------------
# 3. Activate venv and install dependencies
# -------------------------------------------------------
echo "[3/4] Installing dependencies..."

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS" --quiet

echo "  Dependencies installed."

# -------------------------------------------------------
# 4. Run inference
# -------------------------------------------------------
echo "[4/4] Running inference..."

python inference.py --input "$INPUT" --output "$OUTPUT"

echo ""
echo "Done. Results saved to $OUTPUT"
