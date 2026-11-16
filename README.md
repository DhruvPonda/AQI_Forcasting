# Air Quality Forecasting Pipeline on AWS 🌫️☁️

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![AWS S3](https://img.shields.io/badge/AWS-S3%20Data%20Lake-569A31.svg)](https://aws.amazon.com/s3/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Compute-FF9900.svg)](https://aws.amazon.com/ec2/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-success.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, cloud-deployed data engineering and time-series machine learning system built with **100% Python** on **AWS**. Ingests daily atmospheric and meteorological variables across major Indian cities from Open-Meteo REST APIs, validates data schemas, trains multi-horizon ML models, versions champions, and serves real-time 7-day AQI forecasts via an interactive multi-page **Streamlit** dashboard.

---

## 🏗️ System Architecture

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

Cross-Cutting: IAM Role · Security Group · CloudWatch · AWS Budgets · Cron / Systemd
```

---

## 🌟 Key Features

1. **Production-Grade Data Engineering:**
   - Partitioned S3 Data Lake layout (`raw/`, `processed/`, `models/`, `predictions/`, `metrics/`).
   - Columnar **Apache Parquet** storage with Snappy compression for 90%+ storage savings.
   - Exponential backoff retries and idempotent API ingestion.
   - Automated data validation checks (schema, impossible values, freshness, missingness).

2. **Time-Series Machine Learning:**
   - **Direct Multi-Horizon Forecasting:** Separate models trained for $h \in [1..7]$ days ahead to eliminate error accumulation.
   - **Leakage Prevention:** Features computed strictly on information available up to anchor date $t$.
   - **Candidate Models:** Naive Persistence, 7-Day Seasonal Naive, Ridge Regression, Random Forest, and XGBoost.
   - **Expanding-Window Cross-Validation:** No future information leaked into validation folds.
   - **Skill Score:** Model improvement explicitly benchmarked against persistence baselines.

3. **Continuous MLOps & Monitoring:**
   - Automated weekly retraining.
   - Champion/Challenger versioning with rollback protection.
   - Continuous ground-truth matching and rolling MAE computation.
   - Statistical data drift monitoring via **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests.

4. **Multi-Page Streamlit Dashboard:**
   - **Overview:** Interactive 7-day forecasts with dynamic uncertainty bands.
   - **Trends & EDA:** Timeline, seasonal heatmap, and meteorological correlations (wind, temperature).
   - **Forecast vs Actual:** Real-time tracking against incoming ground truth.
   - **Model Leaderboard:** Transparent multi-horizon MAE/RMSE comparisons.
   - **Explainability:** Feature importance analysis identifying environmental pollution drivers.
   - **Accuracy Over Time:** Rolling error trajectories and distribution drift alarms.
   - **Pipeline Status:** System health, data lake telemetry, and cloud architecture specs.

---

## 📂 Project Structure

```
AQI_Forcasting/
├── README.md
├── requirements.txt
├── config.yaml                # Cities, API endpoints, S3 bucket, ML parameters
├── pytest.ini                 # Pytest configuration
├── Makefile                   # Automation shortcuts
├── Dockerfile                 # Optional container deployment
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration loader & env overrides
│   ├── utils.py               # Logging, timezone (IST), AQI category mappings
│   ├── storage.py             # Dual-mode storage manager (Local disk / AWS S3)
│   ├── models.py              # Custom baseline forecasters
│   ├── collect.py             # Daily API ingestion with retry logic
│   ├── backfill.py            # Historical bootstrapping script
│   ├── clean.py               # Data sanitization, deduplication, & daily aggregation
│   ├── validate.py            # Automated quality assurance checks
│   ├── features.py            # Time-series lag, rolling, seasonal feature engineering
│   ├── train.py               # Multi-horizon training, time-series CV & versioning
│   ├── predict.py             # 7-day multi-horizon inference engine
│   └── evaluate.py            # Forecast-vs-actual evaluation & statistical drift
├── app/
│   ├── Home.py                # Streamlit entry point (Overview & Forecast)
│   └── pages/
│       ├── 1_Trends.py
│       ├── 2_Forecast_vs_Actual.py
│       ├── 3_Model_Leaderboard.py
│       ├── 4_Explainability.py
│       ├── 5_Accuracy_Over_Time.py
│       └── 6_Pipeline_Status.py
├── tests/
│   ├── test_clean.py
│   ├── test_features.py
│   └── test_validate.py
├── scripts/
│   ├── run_pipeline.sh        # Daily pipeline execution
│   ├── retrain.sh             # Weekly model retraining
│   └── setup_ec2.sh           # Automated EC2 server bootstrap
└── deploy/
    ├── aqi-dashboard.service  # Systemd daemon configuration
    ├── crontab.txt            # Cron scheduling configuration
    ├── iam_policy.json        # Least-privilege IAM policy for S3
    └── nginx.conf             # Optional reverse proxy configuration
```

---

## ⚡ Quick Start (Local Laptop)

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/aqi-forecast-pipeline.git
cd aqi-forecast-pipeline

python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Historical Backfill (Bootstrap)
```bash
python -m src.backfill --days 90
```

### 3. Run Pipeline Steps
```bash
# Clean raw data into processed daily Parquet
python -m src.clean

# Run automated validation
python -m src.validate

# Train candidate models & version champion
python -m src.train

# Generate 7-day multi-horizon forecasts
python -m src.predict

# Evaluate accuracy & drift
python -m src.evaluate
```

### 4. Run Unit Tests
```bash
pytest
```

### 5. Launch the Dashboard
```bash
streamlit run app/Home.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ AWS Deployment Walkthrough

### 1. S3 Data Lake Setup
1. Create a bucket (e.g. `aqi-pipeline-yourname`) in `ap-south-1` (Mumbai).
2. Enable SSE-S3 encryption. Keep **Block all public access = ON**.

### 2. IAM Least-Privilege Role
1. Create policy using `deploy/iam_policy.json` (grants `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` strictly for your bucket).
2. Create role `AqiPipelineEC2Role` for EC2 and attach this policy.

### 3. Security Group Rules
| Type | Port | Source | Description |
|---|---|---|---|
| SSH | 22 | My IP | Administrative SSH access |
| Custom TCP | 8501 | 0.0.0.0/0 | Public Streamlit dashboard access |

### 4. Launch EC2 Instance
- **AMI:** Ubuntu Server 22.04 / 24.04 LTS
- **Instance Type:** `t3.small` (2 vCPU, 2GB RAM)
- **IAM Instance Profile:** `AqiPipelineEC2Role`
- **Storage:** 20-30 GB gp3 root volume

### 5. Deploy Code & Run Bootstrap
SSH into EC2:
```bash
ssh -i key.pem ubuntu@<EC2-PUBLIC-IP>
git clone https://github.com/<your-username>/aqi-forecast-pipeline.git
cd aqi-forecast-pipeline
chmod +x scripts/*.sh
./scripts/setup_ec2.sh
```

---

## 📊 Evaluation & Model Performance

| Horizon | Champion Model | Test MAE (AQI pts) | Skill Score vs Persistence |
|---|---|---|---|
| **Day +1** | Random Forest / XGBoost | **~19.8** | **+23.5%** |
| **Day +2** | Random Forest | **~24.8** | **+18.2%** |
| **Day +3** | XGBoost | **~24.5** | **+16.4%** |
| **Day +4** | Random Forest | **~25.3** | **+12.1%** |
| **Day +5** | XGBoost | **~24.1** | **+11.8%** |
| **Day +6** | Baseline Persistence | **~25.9** | **0.0%** |
| **Day +7** | Baseline Persistence | **~26.0** | **0.0%** |

---

## 🛡️ License

This project is licensed under the MIT License.
