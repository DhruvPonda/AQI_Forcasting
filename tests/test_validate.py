"""Unit tests for data validation checks."""

import pandas as pd
import pytest

from src.validate import run_data_validation


def test_validation_schema_and_range():
    df = pd.DataFrame({
        "city": ["Delhi"] * 50,
        "date": pd.date_range("2026-10-01", periods=50, freq="D").strftime("%Y-%m-%d"),
        "us_aqi_mean": [120.0] * 50,
    })

    result = run_data_validation(df)
    assert result["overall_status"] == "PASS"

    checks_dict = {c["check"]: c["status"] for c in result["checks"]}
    assert checks_dict["Schema Check"] == "PASS"
    assert checks_dict["AQI Range Check"] == "PASS"


def test_validation_detects_out_of_range():
    df = pd.DataFrame({
        "city": ["Delhi"] * 50,
        "date": pd.date_range("2026-10-01", periods=50, freq="D").strftime("%Y-%m-%d"),
        "us_aqi_mean": [-5.0] + [120.0] * 49,
    })

    result = run_data_validation(df)
    assert result["overall_status"] == "FAIL"
    checks_dict = {c["check"]: c["status"] for c in result["checks"]}
    assert checks_dict["AQI Range Check"] == "FAIL"
