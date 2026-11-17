"""Trends and Exploratory Data Analysis page."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import CITIES
from src.storage import storage

st.set_page_config(page_title="Trends & EDA | AQI Pipeline", page_icon="📈", layout="wide")

st.title("📈 Historical AQI Trends & Meteorological Analysis")
st.markdown("Exploratory Data Analysis over the cleaned multi-city time-series dataset.")

daily_df = storage.load_parquet("processed/daily_dataset.parquet")

if daily_df is None or daily_df.empty:
    st.warning("No processed dataset found. Run cleaning first.")
    st.stop()

# City filter
cities = sorted(daily_df["city"].unique())
selected_cities = st.multiselect("Select cities to compare", cities, default=cities[:3])

if not selected_cities:
    st.info("Select at least one city.")
    st.stop()

filtered_df = daily_df[daily_df["city"].isin(selected_cities)].copy()
filtered_df["date"] = pd.to_datetime(filtered_df["date"])

# 1. Historical AQI Timeline
st.subheader("🗓️ Daily Average AQI Over Time")
fig_time = px.line(
    filtered_df,
    x="date",
    y="us_aqi_mean",
    color="city",
    title="Daily Mean US AQI Timeline",
    labels={"us_aqi_mean": "Mean US AQI", "date": "Date", "city": "City"},
    template="plotly_dark",
)
fig_time.add_hrect(y0=0, y1=50, line_width=0, fillcolor="#00E400", opacity=0.1, annotation_text="Good (0-50)")
fig_time.add_hrect(y0=51, y1=100, line_width=0, fillcolor="#FFFF00", opacity=0.1, annotation_text="Moderate (51-100)")
fig_time.add_hrect(y0=101, y1=150, line_width=0, fillcolor="#FF7E00", opacity=0.1, annotation_text="Sensitive (101-150)")
fig_time.add_hrect(y0=151, y1=200, line_width=0, fillcolor="#FF0000", opacity=0.1, annotation_text="Unhealthy (151-200)")
fig_time.add_hrect(y0=201, y1=500, line_width=0, fillcolor="#8F3F97", opacity=0.1, annotation_text="Hazardous (>200)")

st.plotly_chart(fig_time, use_container_width=True)

# 2. Weather vs AQI Correlation
st.subheader("🌦️ Meteorological Factors vs Air Quality")
col1, col2 = st.columns(2)

with col1:
    if "wind_speed_10m_mean" in filtered_df.columns:
        fig_wind = px.scatter(
            filtered_df,
            x="wind_speed_10m_mean",
            y="us_aqi_mean",
            color="city",
            trendline="ols",
            title="Wind Speed vs AQI (Ventilation Effect)",
            labels={"wind_speed_10m_mean": "Wind Speed (km/h)", "us_aqi_mean": "US AQI"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_wind, use_container_width=True)

with col2:
    if "temperature_2m_mean" in filtered_df.columns:
        fig_temp = px.scatter(
            filtered_df,
            x="temperature_2m_mean",
            y="us_aqi_mean",
            color="city",
            trendline="ols",
            title="Temperature vs AQI (Inversion Effect)",
            labels={"temperature_2m_mean": "Temperature (°C)", "us_aqi_mean": "US AQI"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

# 3. Monthly Heatmap
st.subheader("🗓️ Monthly AQI Heatmap by City")
heatmap_df = daily_df.copy()
heatmap_df["date"] = pd.to_datetime(heatmap_df["date"])
heatmap_df["year_month"] = heatmap_df["date"].dt.strftime("%Y-%m")
pivot_month = heatmap_df.groupby(["city", "year_month"])["us_aqi_mean"].mean().unstack()

fig_heat = px.imshow(
    pivot_month,
    labels=dict(x="Month", y="City", color="Mean AQI"),
    color_continuous_scale="YlOrRd",
    template="plotly_dark",
    aspect="auto",
    title="Average AQI Heatmap by City and Month",
)
st.plotly_chart(fig_heat, use_container_width=True)
