"""Data quality validation module for checking schema, ranges, freshness, and completeness."""

from datetime import datetime, timedelta
from typing import Any, Dict, List
import pandas as pd

from src.config import CITIES, FORECAST_CONFIG
from src.storage import storage
from src.utils import get_current_ist_time, get_ist_today_str, logger


def run_data_validation(df: pd.DataFrame = None) -> Dict[str, Any]:
    """Runs data quality checks on processed daily dataset and updates metrics/pipeline_status.json."""
    logger.info("Running automated data quality validation...")
    validation_time = get_current_ist_time().isoformat()
    today_str = get_ist_today_str()

    if df is None:
        df = storage.load_parquet("processed/daily_dataset.parquet")

    checks = []
    overall_status = "PASS"

    if df is None or df.empty:
        status_payload = {
            "last_run_timestamp": validation_time,
            "overall_status": "FAIL",
            "error": "Dataset is missing or empty",
            "checks": [],
            "storage_stats": storage.get_storage_stats(),
        }
        storage.save_json("metrics/pipeline_status.json", status_payload)
        return status_payload

    # 1. Schema Check
    required_cols = ["city", "date", FORECAST_CONFIG.get("target", "us_aqi_mean")]
    missing_cols = [c for c in required_cols if c not in df.columns]
    passed_schema = len(missing_cols) == 0
    checks.append({
        "check": "Schema Check",
        "description": "Ensure required columns exist (city, date, target)",
        "status": "PASS" if passed_schema else "FAIL",
        "details": f"Missing: {missing_cols}" if missing_cols else "All required columns present",
    })
    if not passed_schema:
        overall_status = "FAIL"

    # 2. Volume Check
    expected_min_rows = len(CITIES) * 7
    total_rows = len(df)
    passed_volume = total_rows >= expected_min_rows
    checks.append({
        "check": "Volume Check",
        "description": f"Dataset should contain at least {expected_min_rows} records",
        "status": "PASS" if passed_volume else "WARNING",
        "details": f"Total rows: {total_rows}",
    })

    # 3. Range Check (US AQI: 0 to 500)
    target_col = FORECAST_CONFIG.get("target", "us_aqi_mean")
    if target_col in df.columns:
        invalid_aqi = df[(df[target_col] < 0) | (df[target_col] > 500)]
        passed_range = len(invalid_aqi) == 0
        checks.append({
            "check": "AQI Range Check",
            "description": "US AQI values must lie between 0 and 500",
            "status": "PASS" if passed_range else "FAIL",
            "details": f"Out-of-range rows: {len(invalid_aqi)} (min: {df[target_col].min()}, max: {df[target_col].max()})",
        })
        if not passed_range:
            overall_status = "FAIL"

    # 4. Freshness Check
    max_date = df["date"].max()
    days_lag = (pd.to_datetime(today_str) - pd.to_datetime(max_date)).days
    passed_freshness = days_lag <= 2
    checks.append({
        "check": "Freshness Check",
        "description": "Data should be updated within the last 2 days",
        "status": "PASS" if passed_freshness else "WARNING",
        "details": f"Latest date in dataset: {max_date} (lag: {days_lag} days)",
    })

    # 5. Completeness / Missingness Check
    target_missing = df[target_col].isna().sum() if target_col in df.columns else 0
    pct_missing = (target_missing / total_rows) * 100 if total_rows > 0 else 100
    passed_completeness = pct_missing < 5.0
    checks.append({
        "check": "Completeness Check",
        "description": "Missing values for target should be below 5%",
        "status": "PASS" if passed_completeness else "WARNING",
        "details": f"Missing target count: {target_missing} ({pct_missing:.2f}%)",
    })

    # Storage stats
    storage_stats = storage.get_storage_stats()

    # Model metadata if available
    current_model = storage.load_json("models/current.json")

    status_payload = {
        "last_run_timestamp": validation_time,
        "overall_status": overall_status,
        "total_records": total_rows,
        "cities_covered": df["city"].nunique(),
        "date_range": [df["date"].min(), df["date"].max()],
        "checks": checks,
        "storage_stats": storage_stats,
        "current_model": current_model,
    }

    storage.save_json("metrics/pipeline_status.json", status_payload)
    logger.info(f"Data quality validation complete: status = {overall_status}")
    return status_payload


if __name__ == "__main__":
    run_data_validation()
