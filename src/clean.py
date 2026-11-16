"""Data cleaning, validation, and daily aggregation layer."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.config import CITIES, FORECAST_CONFIG
from src.storage import storage
from src.utils import logger


def parse_raw_json_to_hourly_df(json_data: Dict, city_name: str) -> pd.DataFrame:
    """Parses raw Open-Meteo hourly JSON into a pandas DataFrame."""
    hourly = json_data.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()

    df = pd.DataFrame(hourly)
    df["city"] = city_name
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    return df


def clean_hourly_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans hourly data: drops duplicates, removes impossible values, interpolates small gaps."""
    if df.empty:
        return df

    # Sort and remove duplicates by (city, timestamp)
    df = df.sort_values(by=["city", "timestamp"]).drop_duplicates(subset=["city", "timestamp"]).copy()

    # Sanitize numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # Physical impossibility checks
        if col in ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]:
            # Negative pollutant or AQI values are sensor errors / invalid
            df.loc[df[col] < 0, col] = np.nan
        if col == "us_aqi":
            # AQI strictly above 500 is out-of-scale
            df.loc[df[col] > 500, col] = 500.0
        if col in ["relative_humidity_2m", "cloud_cover"]:
            df[col] = df[col].clip(lower=0.0, upper=100.0)
        if col in ["precipitation", "wind_speed_10m"]:
            df.loc[df[col] < 0, col] = 0.0

        # Linear interpolation for short gaps (up to 3 consecutive missing hours per city)
        df[col] = df.groupby("city")[col].transform(lambda group: group.interpolate(method="linear", limit=3))

    return df


def aggregate_to_daily(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates cleaned hourly data to daily summary statistics (mean, max, min)."""
    if hourly_df.empty:
        return pd.DataFrame()

    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["timestamp"].dt.strftime("%Y-%m-%d")
    numeric_cols = [c for c in hourly_df.select_dtypes(include=[np.number]).columns if c not in ["city", "date"]]

    agg_means = hourly_df.groupby(["city", "date"])[numeric_cols].mean().rename(columns=lambda c: f"{c}_mean")
    agg_maxs = hourly_df.groupby(["city", "date"])[numeric_cols].max().rename(columns=lambda c: f"{c}_max")
    agg_mins = hourly_df.groupby(["city", "date"])[numeric_cols].min().rename(columns=lambda c: f"{c}_min")

    daily = pd.concat([agg_means, agg_maxs, agg_mins], axis=1).reset_index()

    # Compute CPCB-style estimation or AQI category based on us_aqi_mean if available
    if "us_aqi_mean" in daily.columns:
        daily["us_aqi_mean"] = daily["us_aqi_mean"].round(1)

    return daily


def build_processed_dataset() -> pd.DataFrame:
    """Reads all raw data from storage, cleans it, merges air quality + weather, and stores processed dataset."""
    logger.info("Starting cleaning and daily aggregation pipeline...")

    all_aq_dfs = []
    all_weather_dfs = []

    for city in CITIES:
        city_name = city["name"]
        city_slug = city_name.lower().replace(" ", "_")

        # 1. Load backfill if present
        aq_backfill_key = f"raw/air_quality/city={city_slug}/backfill.json"
        aq_backfill = storage.load_json(aq_backfill_key)
        if aq_backfill:
            df = parse_raw_json_to_hourly_df(aq_backfill, city_name)
            if not df.empty:
                all_aq_dfs.append(df)

        w_backfill_key = f"raw/weather/city={city_slug}/backfill.json"
        w_backfill = storage.load_json(w_backfill_key)
        if w_backfill:
            df = parse_raw_json_to_hourly_df(w_backfill, city_name)
            if not df.empty:
                all_weather_dfs.append(df)

        # 2. Also check daily partitions
        aq_keys = storage.list_keys(f"raw/air_quality/city={city_slug}/")
        for key in aq_keys:
            if key.endswith("data.json"):
                data = storage.load_json(key)
                if data:
                    df = parse_raw_json_to_hourly_df(data, city_name)
                    if not df.empty:
                        all_aq_dfs.append(df)

        w_keys = storage.list_keys(f"raw/weather/city={city_slug}/")
        for key in w_keys:
            if key.endswith("data.json"):
                data = storage.load_json(key)
                if data:
                    df = parse_raw_json_to_hourly_df(data, city_name)
                    if not df.empty:
                        all_weather_dfs.append(df)

    if not all_aq_dfs:
        raise ValueError("No raw air quality data found in storage! Run backfill or collect first.")

    aq_hourly = pd.concat(all_aq_dfs, ignore_index=True)
    aq_clean = clean_hourly_data(aq_hourly)
    aq_daily = aggregate_to_daily(aq_clean)

    if all_weather_dfs:
        w_hourly = pd.concat(all_weather_dfs, ignore_index=True)
        w_clean = clean_hourly_data(w_hourly)
        w_daily = aggregate_to_daily(w_clean)
        # Merge AQ and Weather by city and date
        daily_dataset = pd.merge(aq_daily, w_daily, on=["city", "date"], how="left")
    else:
        daily_dataset = aq_daily

    # Sort final dataset
    daily_dataset = daily_dataset.sort_values(by=["city", "date"]).reset_index(drop=True)

    # Save to processed/daily_dataset.parquet
    processed_key = "processed/daily_dataset.parquet"
    storage.save_parquet(processed_key, daily_dataset)
    logger.info(f"Processed dataset saved to '{processed_key}' with {len(daily_dataset)} rows across {daily_dataset['city'].nunique()} cities.")

    return daily_dataset


if __name__ == "__main__":
    build_processed_dataset()
