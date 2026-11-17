"""Pipeline Status, Data Quality, and Cloud Architecture page."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from src.storage import storage

st.set_page_config(page_title="Pipeline Status | AQI Pipeline", page_icon="⚙️", layout="wide")

st.title("⚙️ Pipeline Status & Cloud Infrastructure")
st.markdown("System health, data quality validation logs, and AWS resource architecture.")

status = storage.load_json("metrics/pipeline_status.json")

if not status:
    st.warning("No pipeline status metrics found. Run data validation first.")
    st.stop()

# 1. High Level Status Badges
overall = status.get("overall_status", "UNKNOWN")
badge_color = "#2E7D32" if overall == "PASS" else "#D32F2F"

st.markdown(f"""
<div style="background-color: #1E232A; padding: 15px 25px; border-radius: 8px; border-left: 8px solid {badge_color}; margin-bottom: 20px;">
    <h3 style="margin: 0; color: white;">Pipeline Health: <span style="color: {badge_color}; font-weight: bold;">{overall}</span></h3>
    <p style="margin: 5px 0 0 0; color: #BBB;">Last Ingestion & Validation Run: <b>{status.get('last_run_timestamp', 'N/A')}</b></p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{status.get('total_records', 0):,}")
with col2:
    st.metric("Cities Covered", f"{status.get('cities_covered', 0)}")
with col3:
    date_range = status.get("date_range", ["N/A", "N/A"])
    st.metric("Timeline Span", f"{date_range[0]} to {date_range[1]}")
with col4:
    storage_info = status.get("storage_stats", {})
    st.metric("Storage Volume", f"{storage_info.get('total_size_mb', 0)} MB", f"{storage_info.get('mode', 'local').upper()}")

st.divider()

# 2. Automated Data Quality Validation Checks
st.subheader("🛡️ Automated Data Quality Validation Checks")
checks = status.get("checks", [])
if checks:
    df_checks = pd.DataFrame(checks)
    st.dataframe(
        df_checks[["check", "status", "description", "details"]].rename(
            columns={
                "check": "Quality Check",
                "status": "Validation Result",
                "description": "Rule Description",
                "details": "Details",
            }
        ),
        use_container_width=True,
    )

st.divider()

# 3. AWS Cloud Architecture Overview
st.subheader("☁️ AWS Cloud Deployment Architecture")
st.markdown("""
```
                  ┌───────────────────────────┐
                  │  Open-Meteo APIs          │
                  │  (Air Quality + Weather)  │
                  └─────────────┬─────────────┘
                                │ HTTPS Daily Ingestion
                                ▼
┌───────────────────────────────────────────────────────────┐
│                   AWS EC2 (Ubuntu VM)                     │
│                                                           │
│  1. collect.py   → Ingestion with exponential backoff     │
│  2. clean.py     → Parquet transformation & deduplication │
│  3. validate.py  → Automated data quality schema checks   │
│  4. features.py  → Time-series lag & rolling feature eng  │
│  5. train.py     → Time-series CV & champion promotion    │
│  6. predict.py   → 7-day multi-horizon inference          │
│  7. evaluate.py  → Ground-truth & PSI drift monitoring    │
│  8. Streamlit Dashboard (Port 8501 via systemd)           │
└───────────────┬───────────────────────────────┬───────────┘
                │ boto3 (IAM Instance Role)     │ HTTP :8501
                ▼                               ▼
┌───────────────────────────────┐        ┌──────────────┐
│          AWS S3 Bucket        │        │   End Users  │
│  raw/        processed/       │        │  (Browsers)  │
│  models/     predictions/     │        └──────────────┘
│  metrics/    logs/            │
└───────────────────────────────┘
```
""")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    #### 🚀 Compute & Scheduling
    - **AWS EC2 (Ubuntu 22.04 LTS / 24.04 LTS):** Hosts the automated pipeline and dashboard.
    - **Systemd Service:** Ensures 24/7 background operation of the Streamlit application with auto-restart.
    - **Cron Daemon:** Triggers daily data collection at 01:00 IST and weekly retraining on Sundays.
    """)

with col_b:
    st.markdown("""
    #### 🔒 Security & Resource Management
    - **AWS IAM Role (`AqiPipelineEC2Role`):** Grants scoped least-privilege S3 access without storing hardcoded API credentials.
    - **AWS Security Groups:** Restricts SSH (port 22) to admin IP and opens port 8501 for public dashboard traffic.
    - **CloudWatch & AWS Budgets:** Tracks CPU/network utilization and fires email notifications on abnormal usage.
    """)
