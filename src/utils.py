"""Utility helpers: logging, date/time formatting, and AQI categorization."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pytz

from src.config import BASE_DIR, TIMEZONE

# Setup logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("aqi_pipeline")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(LOG_DIR / "pipeline.log", encoding="utf-8")
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)


def get_current_ist_time() -> datetime:
    """Returns the current datetime in the configured timezone (Asia/Kolkata)."""
    tz = pytz.timezone(TIMEZONE)
    return datetime.now(tz)


def get_ist_today_str() -> str:
    """Returns current date in YYYY-MM-DD format in IST."""
    return get_current_ist_time().strftime("%Y-%m-%d")


def get_aqi_category_and_color(aqi: float) -> Tuple[str, str]:
    """Returns the EPA US AQI category label and hex color for a given AQI value."""
    if aqi is None or aqi < 0:
        return "Unknown", "#808080"
    if aqi <= 50:
        return "Good", "#00E400"
    elif aqi <= 100:
        return "Moderate", "#FFFF00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00"
    elif aqi <= 200:
        return "Unhealthy", "#FF0000"
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97"
    else:
        return "Hazardous", "#7E0023"
