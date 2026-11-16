#!/usr/bin/env bash
set -euo pipefail

# Daily ingestion, validation, forecasting, and evaluation pipeline
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "=========================================================="
echo "Starting Daily AQI Forecasting Pipeline at $(date)"
echo "Working directory: $PROJECT_DIR"
echo "=========================================================="

echo "[1/5] Ingesting latest Air Quality and Weather data..."
python -m src.collect

echo "[2/5] Cleaning, deduplicating and aggregating to daily Parquet..."
python -m src.clean

echo "[3/5] Running automated data quality validation..."
python -m src.validate

echo "[4/5] Generating 7-day multi-horizon forecasts..."
python -m src.predict

echo "[5/5] Evaluating past predictions and checking data drift..."
python -m src.evaluate

echo "=========================================================="
echo "Daily Pipeline Completed Successfully at $(date)"
echo "=========================================================="
