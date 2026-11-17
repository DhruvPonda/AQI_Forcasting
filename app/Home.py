"""Streamlit Main Entrypoint - Overview Page."""

import sys
from pathlib import Path

# Ensure project root is in sys.path so 'src' can always be imported
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config import CITIES
from src.storage import storage
from src.utils import get_aqi_category_and_color

st.set_page_config(
    page_title="AQI Forecast & Cloud Pipeline",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E232A;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #00E400;
        margin-bottom: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_dashboard_data():
    daily_df = storage.load_parquet("processed/daily_dataset.parquet")
    forecast_df = storage.load_parquet("predictions/latest_forecast.parquet")
    status = storage.load_json("metrics/pipeline_status.json")
    model_active = storage.load_json("models/current.json")
    return daily_df, forecast_df, status, model_active


daily_df, forecast_df, status, model_active = load_dashboard_data()

# Sidebar
st.sidebar.title("🌫️ AQI Pipeline")
st.sidebar.markdown("**Cloud Architecture & ML**")

if model_active:
    st.sidebar.success(f"**Model:** `{model_active.get('active_version', 'Unknown')}`")
    st.sidebar.caption(f"Champion MAE: **{model_active.get('overall_avg_mae', 'N/A')}**")
else:
    st.sidebar.info("Model: Not trained yet")

if status:
    overall_st = status.get("overall_status", "UNKNOWN")
    color = "green" if overall_st == "PASS" else "orange"
    st.sidebar.markdown(f"**Pipeline Health:** :{color}[● {overall_st}]")
    st.sidebar.caption(f"Last sync: {status.get('last_run_timestamp', 'N/A')[:19]}")
    stats = status.get("storage_stats", {})
    st.sidebar.caption(f"Storage Mode: **{stats.get('mode', 'local').upper()}** ({stats.get('total_size_mb', 0)} MB)")

st.sidebar.divider()
st.sidebar.markdown("### Navigation")
st.sidebar.caption("Use the sidebar pages to explore Trends, Leaderboard, Explainability, Forecast vs Actual, and Pipeline Status.")

# Main Page
st.title("🇮🇳 Indian Cities Air Quality Forecasting")
st.markdown("""
An automated end-to-end data pipeline running on **AWS (EC2 + S3)**. 
Ingesting hourly atmospheric & weather data from Open-Meteo, validating quality, training multi-horizon ML models, and serving 7-day AQI forecasts.
""")

if forecast_df is None or forecast_df.empty:
    st.warning("No forecasts available yet. Please run the pipeline (backfill, clean, train, predict).")
    st.stop()

# City selector
city_names = [c["name"] for c in CITIES]
selected_city = st.selectbox("Select City for 7-Day Forecast", city_names, index=0)

# Filter for selected city
city_forecast = forecast_df[forecast_df["city"] == selected_city].sort_values("horizon")

# Current observed AQI
col1, col2, col3, col4 = st.columns([1.5, 1.2, 1.2, 1.5])

curr_aqi = None
if daily_df is not None and not daily_df.empty:
    city_hist = daily_df[daily_df["city"] == selected_city].sort_values("date")
    if not city_hist.empty:
        curr_aqi = city_hist.iloc[-1].get("us_aqi_mean", 100)
        curr_date = city_hist.iloc[-1].get("date", "Today")

if curr_aqi is not None:
    cat, col_hex = get_aqi_category_and_color(curr_aqi)
    with col1:
        st.markdown(f"""
        <div style="background-color: #262730; padding: 20px; border-radius: 10px; border-left: 8px solid {col_hex};">
            <span style="font-size: 14px; color: #BBB;">Latest Observed AQI ({curr_date})</span>
            <h1 style="margin: 0; font-size: 44px; color: white;">{curr_aqi:.0f}</h1>
            <span style="background-color: {col_hex}; color: #111; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 13px;">{cat}</span>
        </div>
        """, unsafe_allow_html=True)

if not city_forecast.empty:
    t_plus_1 = city_forecast[city_forecast["horizon"] == 1].iloc[0]
    t_plus_3 = city_forecast[city_forecast["horizon"] == 3].iloc[0]
    t_plus_7 = city_forecast[city_forecast["horizon"] == 7].iloc[0]

    with col2:
        st.metric("1-Day Ahead (Tomorrow)", f"{t_plus_1['predicted_aqi']:.0f}", f"{t_plus_1['category']}")
    with col3:
        st.metric("3-Days Ahead", f"{t_plus_3['predicted_aqi']:.0f}", f"{t_plus_3['category']}")
    with col4:
        st.metric("7-Days Ahead", f"{t_plus_7['predicted_aqi']:.0f}", f"{t_plus_7['category']}")

st.divider()

# Interactive Plotly Forecast Chart with Confidence Band
st.subheader(f"📅 7-Day AQI Forecast for {selected_city}")

fig = go.Figure()

# Add uncertainty band
fig.add_trace(go.Scatter(
    x=city_forecast["forecast_date"],
    y=city_forecast["upper_bound"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    name="Upper Bound",
))
fig.add_trace(go.Scatter(
    x=city_forecast["forecast_date"],
    y=city_forecast["lower_bound"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(255, 126, 0, 0.2)",
    name="Uncertainty Range (±σ)",
))

# Add main forecast line
fig.add_trace(go.Scatter(
    x=city_forecast["forecast_date"],
    y=city_forecast["predicted_aqi"],
    mode="lines+markers+text",
    text=city_forecast["predicted_aqi"].astype(int),
    textposition="top center",
    line=dict(color="#FF7E00", width=3),
    marker=dict(size=9, color=city_forecast["color"]),
    name="Predicted AQI",
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Forecast Date",
    yaxis_title="US AQI",
    yaxis=dict(range=[0, max(350, city_forecast["upper_bound"].max() + 20)]),
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    height=400,
)

st.plotly_chart(fig, use_container_width=True)

# Comparison across all cities
st.subheader("🏙️ Cross-City 7-Day Comparison")
pivot_forecast = forecast_df.pivot(index="city", columns="horizon", values="predicted_aqi")
pivot_forecast.columns = [f"Day +{c}" for c in pivot_forecast.columns]
st.dataframe(pivot_forecast, use_container_width=True)
