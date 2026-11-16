"""Unit tests for data cleaning and daily aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.clean import clean_hourly_data, aggregate_to_daily


def test_clean_hourly_data_removes_negatives_and_clips():
    df = pd.DataFrame({
        "city": ["Delhi", "Delhi", "Delhi"],
        "timestamp": pd.to_datetime(["2026-11-01 00:00", "2026-11-01 01:00", "2026-11-01 02:00"]),
        "us_aqi": [-10.0, 550.0, 120.0],
        "relative_humidity_2m": [110.0, -5.0, 60.0],
    })

    cleaned = clean_hourly_data(df)

    # AQI > 500 clipped to 500
    assert cleaned.loc[cleaned["timestamp"] == "2026-11-01 01:00", "us_aqi"].iloc[0] == 500.0
    # Humidity clamped between 0 and 100
    assert cleaned["relative_humidity_2m"].max() <= 100.0
    assert cleaned["relative_humidity_2m"].min() >= 0.0


def test_clean_hourly_data_drops_duplicates():
    df = pd.DataFrame({
        "city": ["Mumbai", "Mumbai"],
        "timestamp": pd.to_datetime(["2026-11-01 00:00", "2026-11-01 00:00"]),
        "us_aqi": [80.0, 80.0],
    })

    cleaned = clean_hourly_data(df)
    assert len(cleaned) == 1


def test_aggregate_to_daily():
    df = pd.DataFrame({
        "city": ["Delhi"] * 24,
        "timestamp": pd.date_range("2026-11-01", periods=24, freq="h"),
        "us_aqi": [100.0] * 24,
    })

    daily = aggregate_to_daily(df)
    assert len(daily) == 1
    assert daily["city"].iloc[0] == "Delhi"
    assert daily["date"].iloc[0] == "2026-11-01"
    assert daily["us_aqi_mean"].iloc[0] == 100.0
