"""Unit tests for feature engineering and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from src.features import create_calendar_and_seasonal_features, create_lag_and_rolling_features


def test_calendar_and_seasonal_features():
    df = pd.DataFrame({
        "city": ["Delhi", "Mumbai"],
        "date": ["2026-11-01", "2026-07-15"],
        "us_aqi_mean": [250.0, 70.0],
    })

    feat_df = create_calendar_and_seasonal_features(df)
    assert "day_of_week" in feat_df.columns
    assert "is_weekend" in feat_df.columns
    assert "is_crop_burning_season" in feat_df.columns
    assert feat_df.loc[feat_df["date"] == "2026-11-01", "is_crop_burning_season"].iloc[0] == 1
    assert feat_df.loc[feat_df["date"] == "2026-07-15", "is_monsoon"].iloc[0] == 1


def test_lag_features_no_future_leakage():
    dates = pd.date_range("2026-10-01", periods=10, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "city": ["Delhi"] * 10,
        "date": dates,
        "us_aqi_mean": [10.0 * (i + 1) for i in range(10)],
    })

    feat_df = create_lag_and_rolling_features(df, target_col="us_aqi_mean")

    # Day 0 lag 1 must be NaN
    assert pd.isna(feat_df.loc[feat_df["date"] == dates[0], "us_aqi_mean_lag_1"].iloc[0])
    # Day 1 lag 1 must equal Day 0 value
    assert feat_df.loc[feat_df["date"] == dates[1], "us_aqi_mean_lag_1"].iloc[0] == 10.0
    # Day 2 lag 1 must equal Day 1 value
    assert feat_df.loc[feat_df["date"] == dates[2], "us_aqi_mean_lag_1"].iloc[0] == 20.0
