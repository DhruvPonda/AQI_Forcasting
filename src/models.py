"""Baseline and custom model classes for AQI forecasting."""

import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts future AQI as the latest observed lag 1 value."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "us_aqi_mean_lag_1" in X.columns:
            return X["us_aqi_mean_lag_1"].fillna(0).to_numpy()
        return np.zeros(len(X))


class SeasonalNaiveBaseline:
    """Predicts future AQI as the value observed 7 days ago (lag 7)."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "us_aqi_mean_lag_7" in X.columns:
            return X["us_aqi_mean_lag_7"].fillna(X.get("us_aqi_mean_lag_1", 0)).to_numpy()
        return np.zeros(len(X))
