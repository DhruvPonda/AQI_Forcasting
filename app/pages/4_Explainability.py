"""Model Explainability and Feature Importance page."""

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

st.set_page_config(page_title="Explainability | AQI Pipeline", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability & Driver Analysis")
st.markdown("""
Understanding what drives the model's predictions: feature importance derived from tree models (Random Forest & XGBoost) and environmental insights.
""")

current_model = storage.load_json("models/current.json")
if not current_model:
    st.warning("No trained model found. Run training first.")
    st.stop()

version_dir = current_model.get("version_path")
meta = storage.load_json(f"{version_dir}/metadata.json")

if not meta or "feature_importances_h1" not in meta:
    st.warning("Feature importance metadata not found for the current model.")
    st.stop()

feat_imp = meta["feature_importances_h1"]
df_imp = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)

# 1. Top Feature Importance Chart
st.subheader("🌲 Top Drivers of 1-Day Ahead AQI (Horizon 1)")
fig_imp = px.bar(
    df_imp.tail(15),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Relative Feature Importance (Tree Champion Model)",
    labels={"Importance": "Feature Importance Score", "Feature": "Input Feature"},
    template="plotly_dark",
    color="Importance",
    color_continuous_scale="Viridis",
)
st.plotly_chart(fig_imp, use_container_width=True)

# 2. Domain & Scientific Interpretation
st.subheader("💡 Environmental & Meteorological Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### 1. Autoregressive Inertia (`us_aqi_mean_lag_1`, `roll_mean_7`)
    - Atmospheric pollutant concentrations exhibit strong persistence. 
    - Yesterday's AQI and the 7-day rolling baseline carry the highest predictive power for tomorrow's air quality.
    
    #### 2. Wind Speed & Dispersion (`wind_speed_10m`)
    - Low wind speeds (< 5-8 km/h) trap emissions near the ground (stagnant boundary layer).
    - Higher wind speeds significantly dilute PM2.5 and PM10 particles, driving dramatic drops in AQI.
    """)

with col2:
    st.markdown("""
    #### 3. Seasonal & Inversion Effects (`is_crop_burning_season`, `is_winter`)
    - In North India (Delhi, Ahmedabad), winter and post-monsoon (Oct-Nov) trigger severe thermal inversion and crop residue burning.
    - Cold air traps particulate matter near the ground, multiplying AQI values 3x to 5x.
    
    #### 4. Humidity & Photochemistry (`relative_humidity_2m`)
    - High relative humidity can promote secondary aerosol formation and hygroscopic growth of particulate matter.
    """)
