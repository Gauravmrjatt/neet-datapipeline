#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run setup_env.sh first."
    exit 1
fi

# Parse arguments
PHASE="${1:-full}"
CONFIG="${2:-config/settings.yaml}"

echo "NEET Counselling Dataset Pipeline"
echo "=================================="
echo "Phase: $PHASE"
echo "Config: $CONFIG"
echo ""

case "$PHASE" in
    full)
        python -m src.coordinator --config "$CONFIG"
        ;;
    1|2|3|4|5|6|7)
        python -m src.coordinator --config "$CONFIG" --phase "$PHASE"
        ;;
    resume*)
        RESUME_PHASE="${PHASE#resume}"
        python -m src.coordinator --config "$CONFIG" --resume "$RESUME_PHASE"
        ;;
    status)
        python -m src.coordinator --config "$CONFIG" --status
        ;;
    *)
        echo "Usage: $0 {full|1-7|resume<N>|status} [config_path]"
        exit 1
        ;;
esac
