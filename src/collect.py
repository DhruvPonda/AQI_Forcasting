"""Daily ingestion script for fetching Air Quality and Weather data from Open-Meteo."""

import json
import time
from typing import Any, Dict, Optional
import requests

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # Graceful fallback if tenacity is not yet installed
    def retry(*args, **kwargs):
        def decorator(f):
            def wrapper(*a, **kw):
                for attempt in range(3):
                    try:
                        return f(*a, **kw)
                    except Exception as e:
                        if attempt == 2:
                            raise
                        time.sleep(2 ** attempt)
            return wrapper
        return decorator
    stop_after_attempt = None
    wait_exponential = None

from src.config import APIS_CONFIG, CITIES, TIMEZONE
from src.storage import storage
from src.utils import get_current_ist_time, get_ist_today_str, logger


@retry(stop=stop_after_attempt(3) if stop_after_attempt else None, 
       wait=wait_exponential(multiplier=1, min=2, max=10) if wait_exponential else None)
def fetch_api_data(url: str, params: Dict[str, Any], timeout: int = 15) -> Dict[str, Any]:
    """Fetches JSON data from an API with retry and timeout."""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_air_quality_for_city(city: Dict[str, Any], past_days: int = 7, forecast_days: int = 1) -> Dict[str, Any]:
    """Fetches hourly air quality data from Open-Meteo Air Quality API."""
    url = APIS_CONFIG.get("air_quality_url", "https://air-quality-api.open-meteo.com/v1/air-quality")
    hourly_vars = APIS_CONFIG.get("hourly_aq_vars", [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"
    ])
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(hourly_vars),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    data = fetch_api_data(url, params)
    data["_metadata"] = {
        "city": city["name"],
        "latitude": city["lat"],
        "longitude": city["lon"],
        "fetch_time_ist": get_current_ist_time().isoformat(),
        "type": "air_quality",
    }
    return data


def fetch_weather_for_city(city: Dict[str, Any], past_days: int = 7, forecast_days: int = 7) -> Dict[str, Any]:
    """Fetches hourly weather (historical past + future forecast) from Open-Meteo Forecast API."""
    url = APIS_CONFIG.get("weather_url", "https://api.open-meteo.com/v1/forecast")
    hourly_vars = APIS_CONFIG.get("hourly_weather_vars", [
        "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
        "wind_direction_10m", "precipitation", "surface_pressure", "cloud_cover"
    ])
    params = {
        "latitude": city["lat"],
        "longitude": city["lon"],
        "hourly": ",".join(hourly_vars),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": TIMEZONE,
    }
    data = fetch_api_data(url, params)
    data["_metadata"] = {
        "city": city["name"],
        "latitude": city["lat"],
        "longitude": city["lon"],
        "fetch_time_ist": get_current_ist_time().isoformat(),
        "type": "weather",
    }
    return data


def run_collection(past_days: int = 7, forecast_days_weather: int = 7) -> Dict[str, Any]:
    """Runs daily data collection for all configured cities and saves raw JSON to data lake."""
    today_str = get_ist_today_str()
    logger.info(f"Starting daily data collection for {len(CITIES)} cities on date {today_str}...")

    results = {
        "date": today_str,
        "timestamp": get_current_ist_time().isoformat(),
        "cities_success": [],
        "cities_failed": [],
    }

    for city in CITIES:
        city_name = city["name"]
        city_slug = city_name.lower().replace(" ", "_")

        try:
            # 1. Fetch Air Quality
            aq_data = fetch_air_quality_for_city(city, past_days=past_days, forecast_days=1)
            aq_key = f"raw/air_quality/city={city_slug}/date={today_str}/data.json"
            storage.save_json(aq_key, aq_data)

            # 2. Fetch Weather
            weather_data = fetch_weather_for_city(city, past_days=past_days, forecast_days=forecast_days_weather)
            weather_key = f"raw/weather/city={city_slug}/date={today_str}/data.json"
            storage.save_json(weather_key, weather_data)

            logger.info(f"Successfully collected and stored raw data for {city_name}.")
            results["cities_success"].append(city_name)
        except Exception as e:
            logger.error(f"Failed data collection for city {city_name}: {e}")
            results["cities_failed"].append({"city": city_name, "error": str(e)})

    # Log summary
    summary_key = f"metrics/daily_ingestion/{today_str}.json"
    storage.save_json(summary_key, results)
    logger.info(f"Collection completed. Success: {len(results['cities_success'])}, Failed: {len(results['cities_failed'])}")
    return results


if __name__ == "__main__":
    run_collection()
