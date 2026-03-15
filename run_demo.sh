#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  Evidence-to-Journal Matching — One-Click Demo
# ═══════════════════════════════════════════════════════════════
#
#  After cloning the repo, just run:
#    bash run_demo.sh
#
# ═══════════════════════════════════════════════════════════════

set -e

ENV_NAME="osama-env"
PYTHON_VERSION="3.11"

echo ""
echo "  Evidence-to-Journal Matching — Setting Up"
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed."
    echo "Install from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create env if needed
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda env '${ENV_NAME}' exists."
else
    echo "  Creating conda env '${ENV_NAME}'..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install deps
echo "  Installing dependencies..."
pip install -r requirements.txt -q

# Setup .env
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  IMPORTANT: Edit .env and add your OpenAI API key."
    echo "  Then run this script again."
    echo ""
    exit 0
fi

# Clean vector store
rm -rf vectordb/*
mkdir -p vectordb data logs

# Verify
python -c "
from app.config import get_config
from app.models import Invoice, JournalGroup
print('  All modules OK.')
" || { echo "ERROR: Module check failed."; exit 1; }

# Launch
echo ""
echo "  Setup complete. Launching dashboard..."
echo ""
echo "  App: http://localhost:8501"
echo ""
echo "  Usage:"
echo "    1. Upload XLSX -> click 'Add to Vector Store'"
echo "    2. Upload PDF invoices -> click 'Run Matching'"
echo "    3. View results"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

streamlit run app/dashboard/streamlit_app.py \
    --server.port=8501 \
    --server.address=localhost \
    --server.headless=true