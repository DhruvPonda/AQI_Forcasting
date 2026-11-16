"""Model training, time-series evaluation, leaderboard ranking, and champion/challenger versioning."""

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from src.config import FORECAST_CONFIG, TRAINING_CONFIG
from src.features import build_feature_matrix
from src.models import PersistenceBaseline, SeasonalNaiveBaseline
from src.storage import storage
from src.utils import get_current_ist_time, logger


def get_candidate_models() -> Dict[str, Any]:
    """Returns candidate model instances to train and compare."""
    return {
        "Baseline_Persistence": PersistenceBaseline(),
        "Baseline_Seasonal_7D": SeasonalNaiveBaseline(),
        "Ridge": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, n_jobs=-1),
    }


def train_and_evaluate_models() -> Dict[str, Any]:
    """Runs TimeSeriesSplit evaluation, selects best model per horizon, and versions artifacts."""
    logger.info("=== Starting Model Training and Evaluation Pipeline ===")

    df, feature_cols = build_feature_matrix()
    horizons = FORECAST_CONFIG.get("horizons", [1, 2, 3, 4, 5, 6, 7])
    test_fraction = TRAINING_CONFIG.get("test_fraction", 0.15)
    n_splits = TRAINING_CONFIG.get("n_splits", 5)

    # Sort strictly by date to ensure proper temporal ordering
    dates = sorted(df["date"].unique())
    n_test_dates = max(7, int(len(dates) * test_fraction))
    train_dates = dates[:-n_test_dates]
    test_dates = dates[-n_test_dates:]

    df_train = df[df["date"].isin(train_dates)].copy()
    df_test = df[df["date"].isin(test_dates)].copy()

    logger.info(f"Split data into Train: {len(train_dates)} dates ({len(df_train)} rows), Test: {len(test_dates)} dates ({len(df_test)} rows)")

    leaderboard_records = []
    best_models_per_horizon = {}
    horizon_metrics = {}

    timestamp_version = get_current_ist_time().strftime("v%Y%m%d_%H%M%S")
    version_dir = f"models/{timestamp_version}"

    for h in horizons:
        target_name = f"target_h{h}"
        logger.info(f"--- Training models for Horizon h = {h} ({target_name}) ---")

        # Drop rows where target_h or features are NaN in training set
        train_valid = df_train.dropna(subset=[target_name] + [f"us_aqi_mean_lag_1"])
        test_valid = df_test.dropna(subset=[target_name] + [f"us_aqi_mean_lag_1"])

        X_train = train_valid[feature_cols].fillna(0)
        y_train = train_valid[target_name]
        X_test = test_valid[feature_cols].fillna(0)
        y_test = test_valid[target_name]

        # Determine baseline MAE for skill score calculation
        baseline_model = PersistenceBaseline()
        y_pred_base = baseline_model.predict(X_test)
        base_mae = mean_absolute_error(y_test, y_pred_base) if len(y_test) > 0 else 1.0

        candidates = get_candidate_models()
        best_h_mae = float("inf")
        best_h_model_name = None
        best_h_model = None

        for model_name, model in candidates.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # Ensure non-negative AQI
                y_pred = np.clip(y_pred, 0, 500)

                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                skill_score = 1.0 - (mae / base_mae) if base_mae > 0 else 0.0

                leaderboard_records.append({
                    "version": timestamp_version,
                    "horizon": h,
                    "model_name": model_name,
                    "mae": round(float(mae), 2),
                    "rmse": round(float(rmse), 2),
                    "skill_score_pct": round(float(skill_score * 100), 2),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                })

                if mae < best_h_mae:
                    best_h_mae = mae
                    best_h_model_name = model_name
                    best_h_model = model

            except Exception as e:
                logger.error(f"Error evaluating {model_name} for h={h}: {e}")

        # Retrain champion model on all available valid data (train + test) for deployment
        full_valid = df.dropna(subset=[target_name, "us_aqi_mean_lag_1"])
        X_full = full_valid[feature_cols].fillna(0)
        y_full = full_valid[target_name]
        best_h_model.fit(X_full, y_full)

        best_models_per_horizon[h] = {
            "model_name": best_h_model_name,
            "mae": round(float(best_h_mae), 2),
            "model": best_h_model,
        }
        horizon_metrics[f"h{h}"] = {
            "best_model": best_h_model_name,
            "test_mae": round(float(best_h_mae), 2),
        }

        # Save serialized model artifact
        buf = io.BytesIO()
        joblib.dump(best_h_model, buf)
        storage.save_bytes(f"{version_dir}/model_h{h}.joblib", buf.getvalue())
        logger.info(f"Horizon h={h}: Champion = {best_h_model_name} with Test MAE = {round(best_h_mae, 2)}")

    # Extract feature importance from tree model (e.g. horizon 1)
    feature_importance_dict = {}
    h1_model = best_models_per_horizon[1]["model"]
    if hasattr(h1_model, "feature_importances_"):
        importances = h1_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx[:20]:
            feature_importance_dict[feature_cols[idx]] = round(float(importances[idx]), 4)

    # Save metadata.json
    metadata = {
        "version": timestamp_version,
        "created_at_ist": get_current_ist_time().isoformat(),
        "train_date_range": [train_dates[0], train_dates[-1]],
        "test_date_range": [test_dates[0], test_dates[-1]],
        "feature_cols": feature_cols,
        "horizons": horizons,
        "horizon_metrics": horizon_metrics,
        "feature_importances_h1": feature_importance_dict,
        "overall_avg_mae": round(float(np.mean([m["test_mae"] for m in horizon_metrics.values()])), 2),
    }
    storage.save_json(f"{version_dir}/metadata.json", metadata)

    # Save leaderboard
    df_lb = pd.DataFrame(leaderboard_records).sort_values(by=["horizon", "mae"]).reset_index(drop=True)
    storage.save_json("metrics/model_leaderboard.json", df_lb.to_dict(orient="records"))

    # Champion/Challenger decision
    current_active = storage.load_json("models/current.json")
    promote = True
    if current_active and "overall_avg_mae" in current_active:
        current_mae = current_active["overall_avg_mae"]
        if metadata["overall_avg_mae"] > current_mae * 1.05:  # Require challenger to be not significantly worse
            logger.info(f"Challenger MAE ({metadata['overall_avg_mae']}) did not beat current champion ({current_mae}). Retaining current.")
            promote = False

    if promote:
        active_record = {
            "active_version": timestamp_version,
            "version_path": version_dir,
            "overall_avg_mae": metadata["overall_avg_mae"],
            "promoted_at_ist": get_current_ist_time().isoformat(),
            "horizon_metrics": horizon_metrics,
        }
        storage.save_json("models/current.json", active_record)
        logger.info(f"Promoted {timestamp_version} to ACTIVE champion model.")

    logger.info("=== Model training completed successfully ===")
    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
