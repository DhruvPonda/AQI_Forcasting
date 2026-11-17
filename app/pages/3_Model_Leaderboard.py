"""Model Leaderboard and Multi-Horizon Comparison page."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from src.storage import storage

st.set_page_config(page_title="Model Leaderboard | AQI Pipeline", page_icon="🏆", layout="wide")

st.title("🏆 Machine Learning Model Leaderboard")
st.markdown("""
Comparison of candidate time-series forecasting models across all 7 forecast horizons.
Evaluated with **expanding-window time-series validation** on held-out test data.
""")

leaderboard_data = storage.load_json("metrics/model_leaderboard.json")
current_model = storage.load_json("models/current.json")

if not leaderboard_data:
    st.warning("No model leaderboard metrics available. Run model training first.")
    st.stop()

df_lb = pd.DataFrame(leaderboard_data)

if current_model:
    st.success(f"**Current Champion Version:** `{current_model.get('active_version')}` | **Overall Mean MAE:** {current_model.get('overall_avg_mae')} AQI points")

# Horizon Filter
col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    horizons = sorted(df_lb["horizon"].unique())
    selected_h = st.selectbox("Filter by Forecast Horizon", ["All Horizons"] + [f"Horizon {h} (t+{h}d)" for h in horizons])

if selected_h != "All Horizons":
    h_num = int(selected_h.split()[1])
    view_df = df_lb[df_lb["horizon"] == h_num].sort_values("mae")
else:
    view_df = df_lb.sort_values(by=["horizon", "mae"])

# Leaderboard table
st.subheader("Leaderboard Table")
st.dataframe(
    view_df[["horizon", "model_name", "mae", "rmse", "skill_score_pct", "train_samples", "test_samples"]].rename(
        columns={
            "horizon": "Horizon (Days)",
            "model_name": "Model",
            "mae": "MAE (AQI pts)",
            "rmse": "RMSE",
            "skill_score_pct": "Skill Score vs Baseline (%)",
            "train_samples": "Train Rows",
            "test_samples": "Test Rows",
        }
    ),
    use_container_width=True,
)

# Comparison Bar Chart
st.subheader("📊 Model MAE Comparison Across Horizons")
fig_comp = px.bar(
    df_lb,
    x="horizon",
    y="mae",
    color="model_name",
    barmode="group",
    title="Model MAE by Horizon (Lower is Better)",
    labels={"horizon": "Forecast Horizon (h = days ahead)", "mae": "MAE (AQI points)", "model_name": "Model"},
    template="plotly_dark",
)
st.plotly_chart(fig_comp, use_container_width=True)

# Skill Score vs Baseline
st.subheader("🎯 Skill Score vs Naive Persistence Baseline")
fig_skill = px.line(
    df_lb[df_lb["model_name"] != "Baseline_Persistence"],
    x="horizon",
    y="skill_score_pct",
    color="model_name",
    markers=True,
    title="Skill Score (% Error Reduction vs Naive Baseline)",
    labels={"horizon": "Forecast Horizon (h = days ahead)", "skill_score_pct": "Skill Score (%)"},
    template="plotly_dark",
)
st.plotly_chart(fig_skill, use_container_width=True)
