#!/bin/bash
# NEET Predictor — Setup for Training on Any PC
# Run this on a fresh machine with Python 3.10+

set -e

echo "============================================"
echo "  NEET College Predictor — Training Setup"
echo "============================================"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python: $PYTHON_VERSION"

# 2. Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# 3. Install dependencies
echo "Installing dependencies..."
pip install -r requirements_train.txt -q

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Run these commands in order:"
echo ""
echo "  1. Prepare training data:"
echo "     python scripts/prepare_training_data.py"
echo ""
echo "  2. Train the model:"
echo "     python scripts/train_model.py"
echo ""
echo "  3. Predict (example):"
echo "     python scripts/predict.py --rank 15000 --category General --quota \"All India\" --state Maharashtra"
echo ""
