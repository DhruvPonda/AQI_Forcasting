"""Feature engineering for time-series direct multi-horizon forecasting."""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.config import FORECAST_CONFIG
from src.storage import storage
from src.utils import logger


def create_calendar_and_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates calendar, weekend, seasonal, and pollution-period features."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])

    df["day_of_week"] = dt.dt.dayofweek
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear

    # Cyclical encoding for day of year
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # Indian seasonal indicators:
    # Winter: Dec-Feb (12, 1, 2)
    # Summer/Pre-monsoon: Mar-May (3, 4, 5)
    # Monsoon: Jun-Sep (6, 7, 8, 9)
    # Post-monsoon / Crop burning: Oct-Nov (10, 11) - high pollution season in North India
    df["is_winter"] = dt.dt.month.isin([12, 1, 2]).astype(int)
    df["is_monsoon"] = dt.dt.month.isin([6, 7, 8, 9]).astype(int)
    df["is_crop_burning_season"] = dt.dt.month.isin([10, 11]).astype(int)

    return df


def create_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "us_aqi_mean") -> pd.DataFrame:
    """Computes time-series lag and rolling window features per city without data leakage."""
    df = df.sort_values(by=["city", "date"]).copy()

    # AQI Lags: t-1, t-2, t-3, t-7, t-14
    for lag in [1, 2, 3, 7, 14]:
        df[f"{target_col}_lag_{lag}"] = df.groupby("city")[target_col].shift(lag)

    # Rolling window statistics (computed on t-1 to prevent target leakage)
    shifted_target = df.groupby("city")[target_col].shift(1)
    for window in [3, 7, 14]:
        df[f"{target_col}_roll_mean_{window}"] = shifted_target.groupby(df["city"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = shifted_target.groupby(df["city"]).transform(
            lambda s: s.rolling(window, min_periods=1).std()
        ).fillna(0.0)
        df[f"{target_col}_roll_max_{window}"] = shifted_target.groupby(df["city"]).transform(
            lambda s: s.rolling(window, min_periods=1).max()
        )

    # 7-day trend
    df[f"{target_col}_diff_7"] = df[f"{target_col}_lag_1"] - df[f"{target_col}_lag_7"]

    # Weather lags and rolling statistics
    weather_cols = [
        "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean",
        "precipitation_mean", "surface_pressure_mean"
    ]
    for w_col in weather_cols:
        if w_col in df.columns:
            # t-1 weather
            df[f"{w_col}_lag_1"] = df.groupby("city")[w_col].shift(1)
            # 3-day rolling mean
            shifted_w = df.groupby("city")[w_col].shift(1)
            df[f"{w_col}_roll_3"] = shifted_w.groupby(df["city"]).transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )

    # Pollutant lags (PM2.5, PM10, NO2, O3)
    pollutant_cols = ["pm2_5_mean", "pm10_mean", "nitrogen_dioxide_mean", "ozone_mean"]
    for p_col in pollutant_cols:
        if p_col in df.columns:
            df[f"{p_col}_lag_1"] = df.groupby("city")[p_col].shift(1)

    return df


def prepare_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: List[int] = None,
    target_col: str = "us_aqi_mean",
) -> pd.DataFrame:
    """Prepares direct multi-horizon targets y_h = AQI(t + h) per city."""
    if horizons is None:
        horizons = FORECAST_CONFIG.get("horizons", [1, 2, 3, 4, 5, 6, 7])

    df = df.sort_values(by=["city", "date"]).copy()
    for h in horizons:
        # Shift negatively to align future actual target at horizon h with anchor date t
        df[f"target_h{h}"] = df.groupby("city")[target_col].shift(-h)

    return df


def build_feature_matrix(df: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[str]]:
    """Generates the full feature matrix from the daily processed dataset."""
    logger.info("Building feature matrix with calendar, lag, rolling, and weather features...")

    if df is None:
        df = storage.load_parquet("processed/daily_dataset.parquet")

    target_col = FORECAST_CONFIG.get("target", "us_aqi_mean")

    df_feats = create_calendar_and_seasonal_features(df)
    df_feats = create_lag_and_rolling_features(df_feats, target_col=target_col)
    df_feats = prepare_multi_horizon_targets(df_feats, target_col=target_col)

    # One-hot encode cities
    df_encoded = pd.get_dummies(df_feats, columns=["city"], drop_first=False, dtype=int)

    # Identify feature columns
    non_feature_cols = ["date"] + [c for c in df_encoded.columns if c.startswith("target_h")] + [
        c for c in df.columns if c not in ["city", "date"] and not c.endswith(("_lag_1", "_roll_3", "_diff_7"))
        and not c.startswith("city_")
    ]
    feature_cols = [c for c in df_encoded.columns if c not in non_feature_cols]

    # Save to storage
    storage.save_parquet("processed/features/features_latest.parquet", df_encoded)
    logger.info(f"Feature matrix generated with {len(feature_cols)} features and {len(df_encoded)} rows.")

    return df_encoded, feature_cols


if __name__ == "__main__":
    build_feature_matrix()
