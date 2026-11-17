"""Forecast vs Actual evaluation page."""

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

st.set_page_config(page_title="Forecast vs Actual | AQI Pipeline", page_icon="🎯", layout="wide")

st.title("🎯 Forecast vs Actuals Tracking")
st.markdown("Ground truth comparison: overlaying previously forecasted values with actual observed AQI.")

accuracy_df = storage.load_parquet("metrics/accuracy_over_time.parquet")
daily_df = storage.load_parquet("processed/daily_dataset.parquet")

if accuracy_df is None or accuracy_df.empty:
    # If live accuracy pairs are still accumulating, build a historical validation view from daily dataset
    st.info("Direct forecast-vs-actual matching is tracking recent daily runs. Displaying recent observed trajectories vs model horizon predictions.")
    
    if daily_df is not None and not daily_df.empty:
        cities = sorted(daily_df["city"].unique())
        selected_city = st.selectbox("Select City", cities, index=0)
        c_df = daily_df[daily_df["city"] == selected_city].tail(30).copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=c_df["date"],
            y=c_df["us_aqi_mean"],
            mode="lines+markers",
            name="Actual AQI",
            line=dict(color="#00E400", width=3)
        ))
        
        # Load forecast
        fc = storage.load_parquet("predictions/latest_forecast.parquet")
        if fc is not None and not fc.empty:
            fc_city = fc[fc["city"] == selected_city]
            fig.add_trace(go.Scatter(
                x=fc_city["forecast_date"],
                y=fc_city["predicted_aqi"],
                mode="lines+markers",
                name="7-Day Forecast",
                line=dict(color="#FF7E00", width=3, dash="dash")
            ))

        fig.update_layout(
            template="plotly_dark",
            title=f"Recent Actuals & Active Forecast for {selected_city}",
            xaxis_title="Date",
            yaxis_title="US AQI",
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    cities = sorted(accuracy_df["city"].unique())
    selected_city = st.selectbox("Select City", cities, index=0)

    city_acc = accuracy_df[accuracy_df["city"] == selected_city].sort_values("forecast_date")

    col1, col2, col3 = st.columns(3)
    avg_mae = city_acc["abs_error"].mean()
    avg_rmse = (city_acc["sq_error"].mean()) ** 0.5
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{avg_mae:.2f} AQI pts")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{avg_rmse:.2f}")
    with col3:
        st.metric("Evaluated Days", f"{len(city_acc)}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_acc["forecast_date"],
        y=city_acc["actual_aqi"],
        mode="lines+markers",
        name="Actual AQI",
        line=dict(color="#00E400", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=city_acc["forecast_date"],
        y=city_acc["predicted_aqi"],
        mode="lines+markers",
        name="Predicted AQI",
        line=dict(color="#FF7E00", width=2.5, dash="dot"),
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"Forecast vs Actual Comparison for {selected_city}",
        xaxis_title="Date",
        yaxis_title="US AQI",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Error by Horizon (h = 1 to 7)")
    h_error = accuracy_df.groupby("horizon")["abs_error"].mean().reset_index()
    fig_h = px.bar(
        h_error,
        x="horizon",
        y="abs_error",
        title="MAE by Forecast Horizon (Uncertainty Growth)",
        labels={"horizon": "Forecast Horizon (Days Ahead)", "abs_error": "Mean Absolute Error"},
        template="plotly_dark",
        color="abs_error",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(fig_h, use_container_width=True)
