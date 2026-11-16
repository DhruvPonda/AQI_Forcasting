"""Accuracy Over Time and Data Drift Monitoring page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import MONITORING_CONFIG
from src.storage import storage

st.set_page_config(page_title="Accuracy Over Time & Drift | AQI Pipeline", page_icon="📉", layout="wide")

st.title("📉 Accuracy Over Time & Data Drift Monitoring")
st.markdown("MLOps monitoring: tracking rolling forecasting accuracy and statistical distribution shift.")

accuracy_df = storage.load_parquet("metrics/accuracy_over_time.parquet")
drift_report = storage.load_json("metrics/drift_report.json")
threshold_mae = MONITORING_CONFIG.get("retrain_if_mae_above", 25.0)

col_m1, col_m2, col_m3 = st.columns(3)

if accuracy_df is not None and not accuracy_df.empty:
    overall_mae = accuracy_df["abs_error"].mean()
    h1_mae = accuracy_df[accuracy_df["horizon"] == 1]["abs_error"].mean()
    with col_m1:
        st.metric("Overall 7-Horizon MAE", f"{overall_mae:.2f} AQI pts")
    with col_m2:
        st.metric("1-Day Horizon MAE", f"{h1_mae:.2f} AQI pts")
    with col_m3:
        status_color = "normal" if overall_mae <= threshold_mae else "inverse"
        retrain_flag = "HEALTHY" if overall_mae <= threshold_mae else "RETRAIN RECOMMENDED"
        st.metric("Retraining Health", retrain_flag, f"Threshold: {threshold_mae} pts")

st.divider()

# 1. Rolling MAE Over Time
st.subheader("📅 Rolling Forecast Error Over Time")
if accuracy_df is not None and not accuracy_df.empty:
    daily_mae = accuracy_df.groupby(["forecast_date", "horizon"])["abs_error"].mean().reset_index()
    daily_mae["rolling_7d_mae"] = daily_mae.groupby("horizon")["abs_error"].transform(lambda s: s.rolling(7, min_periods=1).mean())

    fig_rolling = px.line(
        daily_mae[daily_mae["horizon"].isin([1, 3, 7])],
        x="forecast_date",
        y="rolling_7d_mae",
        color="horizon",
        markers=True,
        title="7-Day Rolling MAE by Forecast Horizon",
        labels={"forecast_date": "Forecast Date", "rolling_7d_mae": "7-Day Rolling MAE", "horizon": "Horizon"},
        template="plotly_dark",
    )
    fig_rolling.add_hline(y=threshold_mae, line_dash="dash", line_color="red", annotation_text=f"Retrain Threshold ({threshold_mae})")
    st.plotly_chart(fig_rolling, use_container_width=True)

# 2. Data Drift Monitoring
st.subheader("🧪 Statistical Data Drift & Distribution Shifts")
st.markdown("""
Using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) tests** to detect feature drift between baseline historical data and current incoming records.
- **PSI < 0.1**: No significant shift
- **0.1 ≤ PSI < 0.25**: Moderate shift (monitor)
- **PSI ≥ 0.25**: Significant drift (triggers automated retraining)
""")

if drift_report:
    drift_rows = []
    for var, metrics in drift_report.items():
        if isinstance(metrics, dict):
            drift_rows.append({
                "Variable": var,
                "PSI (Population Stability)": metrics.get("psi"),
                "KS-Statistic": metrics.get("ks_statistic"),
                "KS p-value": metrics.get("ks_pvalue"),
                "Drift Alert": "⚠️ DRIFT DETECTED" if metrics.get("drift_detected") else "✅ STABLE",
            })
    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)
else:
    st.info("No drift report generated yet.")
