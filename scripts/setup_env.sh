#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "Setting up NEET Pipeline environment..."

# Check Python 3.12
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN="/opt/homebrew/bin/python3.12"
fi

if ! command -v "$PYTHON_BIN" &>/dev/null; then
    echo "ERROR: Python 3.12 not found. Install via: brew install python@3.12"
    exit 1
fi

echo "Using Python: $($PYTHON_BIN --version)"

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv .venv
fi

source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Setup Crawl4AI (installs Chromium)
echo "Setting up Crawl4AI browser..."
crawl4ai-setup

# Start PostgreSQL
echo "Starting PostgreSQL..."
if command -v docker &>/dev/null; then
    docker compose up -d postgres
    echo "Waiting for PostgreSQL to be healthy..."
    sleep 5
else
    echo "WARNING: Docker not found. PostgreSQL setup skipped."
fi

echo ""
echo "Setup complete!"
echo "Activate venv: source .venv/bin/activate"
echo "Run pipeline:  ./scripts/run_pipeline.sh full"
