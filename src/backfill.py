"""Historical bootstrap/backfill script for Open-Meteo Air Quality and Weather data."""

import argparse
from datetime import datetime, timedelta
from typing import Any, Dict, List
import requests

from src.config import APIS_CONFIG, CITIES, TIMEZONE
from src.storage import storage
from src.utils import get_current_ist_time, logger


def backfill_city_data(city: Dict[str, Any], days: int = 90) -> None:
    """Fetches past historical air quality and weather data for a city and saves to raw storage."""
    city_name = city["name"]
    city_slug = city_name.lower().replace(" ", "_")
    logger.info(f"Backfilling {days} days of historical data for {city_name} (lat: {city['lat']}, lon: {city['lon']})...")

    # 1. Fetch historical Air Quality
    aq_url = APIS_CONFIG.get("air_quality_url", "https://air-quality-api.open-meteo.com/v1/air-quality")
    aq_vars = APIS_CONFIG.get("hourly_aq_vars", [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"
    ])
    aq_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(aq_vars),
        "past_days": days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    resp_aq = requests.get(aq_url, params=aq_params, timeout=30)
    resp_aq.raise_for_status()
    aq_json = resp_aq.json()
    aq_json["_metadata"] = {
        "city": city_name,
        "type": "air_quality_backfill",
        "days": days,
        "fetch_time_ist": get_current_ist_time().isoformat(),
    }

    # Save comprehensive backfill file
    storage.save_json(f"raw/air_quality/city={city_slug}/backfill.json", aq_json)

    # 2. Fetch historical Weather
    weather_url = APIS_CONFIG.get("weather_url", "https://api.open-meteo.com/v1/forecast")
    weather_vars = APIS_CONFIG.get("hourly_weather_vars", [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "surface_pressure", "cloud_cover"
    ])
    weather_params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(weather_vars),
        "past_days": days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }
    resp_w = requests.get(weather_url, params=weather_params, timeout=30)
    resp_w.raise_for_status()
    w_json = resp_w.json()
    w_json["_metadata"] = {
        "city": city_name,
        "type": "weather_backfill",
        "days": days,
        "fetch_time_ist": get_current_ist_time().isoformat(),
    }

    # Save comprehensive backfill file
    storage.save_json(f"raw/weather/city={city_slug}/backfill.json", w_json)

    # Partition into daily slices to populate the raw data lake partition layout
    hourly_times = aq_json.get("hourly", {}).get("time", [])
    if hourly_times:
        dates_seen = sorted(list(set(t.split("T")[0] for t in hourly_times)))
        for dt_str in dates_seen:
            # We can also store the daily partition if not existing
            daily_aq_key = f"raw/air_quality/city={city_slug}/date={dt_str}/data.json"
            daily_w_key = f"raw/weather/city={city_slug}/date={dt_str}/data.json"

            # Filter indices for this date
            indices = [i for i, t in enumerate(hourly_times) if t.startswith(dt_str)]
            if indices:
                sub_aq = {
                    "hourly": {k: [aq_json["hourly"][k][i] for i in indices] for k in aq_json["hourly"]},
                    "_metadata": {"city": city_name, "date": dt_str, "type": "air_quality"}
                }
                storage.save_json(daily_aq_key, sub_aq)

            w_times = w_json.get("hourly", {}).get("time", [])
            w_indices = [i for i, t in enumerate(w_times) if t.startswith(dt_str)]
            if w_indices:
                sub_w = {
                    "hourly": {k: [w_json["hourly"][k][i] for i in w_indices] for k in w_json["hourly"]},
                    "_metadata": {"city": city_name, "date": dt_str, "type": "weather"}
                }
                storage.save_json(daily_w_key, sub_w)

    logger.info(f"Finished backfilling {city_name}.")


def run_backfill(days: int = 90) -> None:
    """Backfills historical data for all cities."""
    logger.info(f"=== Starting historical backfill for {len(CITIES)} cities ({days} days) ===")
    for city in CITIES:
        try:
            backfill_city_data(city, days=days)
        except Exception as e:
            logger.error(f"Error backfilling {city['name']}: {e}")
    logger.info("=== Historical backfill completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap historical air quality & weather data.")
    parser.add_argument("--days", type=int, default=90, help="Number of historical days to fetch (default: 90)")
    args = parser.parse_args()
    run_backfill(days=args.days)
