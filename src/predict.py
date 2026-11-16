"""Forecasting inference pipeline producing 7-day daily AQI forecasts per city."""

import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.config import CITIES, FORECAST_CONFIG
from src.features import build_feature_matrix
from src.models import PersistenceBaseline, SeasonalNaiveBaseline
from src.storage import storage
from src.utils import get_aqi_category_and_color, get_current_ist_time, get_ist_today_str, logger


def generate_forecasts() -> pd.DataFrame:
    """Generates 7-day forecasts for each city using the current champion models."""
    logger.info("=== Starting 7-Day Forecast Generation Pipeline ===")

    # 1. Load active champion model pointer
    current_meta = storage.load_json("models/current.json")
    if not current_meta:
        raise FileNotFoundError("No active model found in 'models/current.json'. Run train.py first.")

    version_dir = current_meta["version_path"]
    model_metadata = storage.load_json(f"{version_dir}/metadata.json")
    feature_cols = model_metadata["feature_cols"]
    horizons = model_metadata.get("horizons", [1, 2, 3, 4, 5, 6, 7])

    # 2. Load latest features matrix
    df_features = storage.load_parquet("processed/features/features_latest.parquet")
    if df_features is None or df_features.empty:
        df_features, _ = build_feature_matrix()

    # Load each horizon model into memory
    models = {}
    for h in horizons:
        model_bytes = storage.load_bytes(f"{version_dir}/model_h{h}.joblib")
        if not model_bytes:
            raise FileNotFoundError(f"Model file {version_dir}/model_h{h}.joblib not found!")
        buf = io.BytesIO(model_bytes)
        models[h] = joblib.load(buf)

    # 3. For each city, grab the latest available row (anchor date t)
    forecast_rows = []
    today_str = get_ist_today_str()

    for city in CITIES:
        city_name = city["name"]
        city_col = f"city_{city_name}"

        # Filter by city
        if city_col in df_features.columns:
            city_df = df_features[df_features[city_col] == 1].sort_values("date")
        else:
            continue

        if city_df.empty:
            logger.warning(f"No feature records found for city: {city_name}")
            continue

        latest_record = city_df.iloc[-1]
        anchor_date_str = latest_record["date"]
        anchor_date = datetime.strptime(anchor_date_str, "%Y-%m-%d")

        # Feature vector
        X_latest = pd.DataFrame([latest_record[feature_cols].fillna(0)])

        for h in horizons:
            model = models[h]
            pred_val = float(model.predict(X_latest)[0])
            pred_val = max(5.0, min(500.0, round(pred_val, 1)))

            target_date = (anchor_date + timedelta(days=h)).strftime("%Y-%m-%d")
            category, color = get_aqi_category_and_color(pred_val)

            # Uncertainty interval expands slightly with horizon h
            uncertainty_margin = round(pred_val * (0.05 + 0.02 * h), 1)
            lower_bound = max(0.0, round(pred_val - uncertainty_margin, 1))
            upper_bound = min(500.0, round(pred_val + uncertainty_margin, 1))

            forecast_rows.append({
                "city": city_name,
                "anchor_date": anchor_date_str,
                "horizon": h,
                "forecast_date": target_date,
                "predicted_aqi": pred_val,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "category": category,
                "color": color,
                "model_version": current_meta["active_version"],
                "generated_at": get_current_ist_time().isoformat(),
            })

    if not forecast_rows:
        raise ValueError("Could not generate any forecasts.")

    df_forecast = pd.DataFrame(forecast_rows)

    # Save to S3 / local data lake
    latest_anchor = df_forecast["anchor_date"].max()
    storage.save_parquet(f"predictions/date={latest_anchor}/forecast.parquet", df_forecast)
    storage.save_parquet("predictions/latest_forecast.parquet", df_forecast)

    logger.info(f"Generated {len(df_forecast)} forecast points across {df_forecast['city'].nunique()} cities for anchor date {latest_anchor}.")
    return df_forecast


if __name__ == "__main__":
    generate_forecasts()
