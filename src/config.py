"""Configuration loader and management for AQI Forecasting Pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Loads configuration from YAML file and applies environment overrides."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Environment variable overrides
    if "STORAGE_MODE" in os.environ:
        config.setdefault("storage", {})["mode"] = os.environ["STORAGE_MODE"].lower()
    if "AWS_S3_BUCKET" in os.environ:
        config.setdefault("aws", {})["bucket"] = os.environ["AWS_S3_BUCKET"]
    if "AWS_REGION" in os.environ:
        config.setdefault("aws", {})["region"] = os.environ["AWS_REGION"]

    return config


# Global config instance
CONFIG = load_config()

CITIES: List[Dict[str, Any]] = CONFIG.get("cities", [])
TIMEZONE: str = CONFIG.get("timezone", "Asia/Kolkata")
AWS_CONFIG: Dict[str, str] = CONFIG.get("aws", {})
STORAGE_CONFIG: Dict[str, Any] = CONFIG.get("storage", {"mode": "local", "local_dir": "data"})
FORECAST_CONFIG: Dict[str, Any] = CONFIG.get("forecast", {"horizons": [1, 2, 3, 4, 5, 6, 7], "target": "us_aqi_mean"})
TRAINING_CONFIG: Dict[str, Any] = CONFIG.get("training", {})
MONITORING_CONFIG: Dict[str, Any] = CONFIG.get("monitoring", {})
APIS_CONFIG: Dict[str, Any] = CONFIG.get("apis", {})
