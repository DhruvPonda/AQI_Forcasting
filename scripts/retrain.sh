#!/usr/bin/env bash
set -euo pipefail

# Weekly retraining and champion/challenger model promotion
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================================="
echo "Starting Weekly Retraining Pipeline at $(date)"
echo "=========================================================="

echo "[1/3] Retraining candidate models across all horizons..."
python -m src.train

echo "[2/3] Updating 7-day multi-horizon forecasts with active champion..."
python -m src.predict

echo "[3/3] Evaluating accuracy metrics and drift..."
python -m src.evaluate

echo "=========================================================="
echo "Weekly Retraining Completed Successfully at $(date)"
echo "=========================================================="
