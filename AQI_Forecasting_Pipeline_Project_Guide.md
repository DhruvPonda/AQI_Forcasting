# Air Quality Forecasting Pipeline on AWS
### A cloud-deployed data engineering + machine learning project (100% Python)

> **Purpose:** College cloud computing assignment *and* a resume/portfolio project.
> **Cloud:** AWS (Free account plan, $100 credits)
> **Language:** Python only

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Mapping to the Assignment Requirements](#2-mapping-to-the-assignment-requirements)
3. [Complete Technology Stack](#3-complete-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [Data Source Details](#5-data-source-details)
6. [Data Engineering Layer](#6-data-engineering-layer)
7. [Machine Learning Layer](#7-machine-learning-layer)
8. [Dashboard (Streamlit)](#8-dashboard-streamlit)
9. [Project Folder Structure](#9-project-folder-structure)
10. [AWS Services: What, Why, How](#10-aws-services-what-why-how)
11. [AWS Account, Credits and Cost Plan](#11-aws-account-credits-and-cost-plan)
12. [Step-by-Step Build Plan (4 Weeks)](#12-step-by-step-build-plan-4-weeks)
13. [AWS Setup Walkthrough (Detailed)](#13-aws-setup-walkthrough-detailed)
14. [Deployment on EC2](#14-deployment-on-ec2)
15. [Scheduling and Automation](#15-scheduling-and-automation)
16. [Monitoring and Resource Management](#16-monitoring-and-resource-management)
17. [Security Practices](#17-security-practices)
18. [Testing and Quality](#18-testing-and-quality)
19. [Optional Upgrades (Resume Boosters)](#19-optional-upgrades-resume-boosters)
20. [What to Submit / Demo Checklist](#20-what-to-submit--demo-checklist)
21. [Report Outline](#21-report-outline)
22. [Resume Content](#22-resume-content)
23. [Likely Viva Questions](#23-likely-viva-questions)
24. [Risks and Honest Limitations](#24-risks-and-honest-limitations)
25. [Cleanup Checklist](#25-cleanup-checklist)
26. [Glossary](#26-glossary)

---

## 1. Project Overview

### 1.1 One-line description
An automated pipeline that collects daily air quality and weather data for several Indian cities, stores it in AWS, trains and compares forecasting models, retrains them automatically, and serves 7-day AQI forecasts through a public dashboard that also exposes model performance and pipeline health.

### 1.2 What makes it different from a "normal AQI website"

| | Normal AQI website | This project |
|---|---|---|
| Data | Shows today's value from an API | Builds **its own historical dataset** by collecting every day |
| Future | Cannot tell tomorrow's AQI | **Forecasts the next 7 days** per city |
| Intelligence | None | Models that learn patterns and are **retrained** as data grows |
| Reliability | Breaks if the API fails | Validation, retries, deduplication, alerts |
| Transparency | Only a number | Shows **forecast vs actual, model leaderboard, feature importance, accuracy over time** |

### 1.3 Goals
- Build a real end-to-end data pipeline (ingest, clean, validate, store, serve).
- Apply proper time-series machine learning (feature engineering, baselines, time-aware validation).
- Deploy on AWS with compute, storage, security, monitoring, and cost control.
- Produce a GitHub repository and write-up that can go straight on a resume.

### 1.4 Cities (default set)

| City | State | Latitude | Longitude |
|---|---|---|---|
| Delhi | Delhi | 28.6139 | 77.2090 |
| Mumbai | Maharashtra | 19.0760 | 72.8777 |
| Ahmedabad | Gujarat | 23.0225 | 72.5714 |
| Bhavnagar | Gujarat | 21.7645 | 72.1519 |
| Kolkata | West Bengal | 22.5726 | 88.3639 |
| Bengaluru | Karnataka | 12.9716 | 77.5946 |

Cities are stored in a config file, so adding or removing one is a one-line change.

---

## 2. Mapping to the Assignment Requirements

| Assignment requirement | How this project satisfies it |
|---|---|
| Create and deploy a basic web application on a cloud platform | Streamlit dashboard deployed on AWS EC2 |
| Configure compute services (VM / app service) | EC2 Ubuntu instance running the collector, training jobs, and dashboard |
| Configure storage services (cloud storage / databases) | S3 for raw data, processed data, models, and metrics |
| Demonstrate successful hosting | Live public URL `http://<EC2-public-IP>:8501` |
| Accessibility over the internet | Security group opens the dashboard port; tested from a phone on mobile data |
| Basic resource management | IAM role, security groups, CloudWatch monitoring, AWS Budgets alert, S3 lifecycle rule, instance stop/terminate |

---

## 3. Complete Technology Stack

### 3.1 Languages and runtime
| Item | Use |
|---|---|
| Python 3.11 (or 3.10+) | Entire project |
| Bash | Small helper scripts, cron, deployment commands |

### 3.2 Python libraries

| Library | Used for |
|---|---|
| `requests` | Calling the Open-Meteo APIs |
| `pandas` | Cleaning, transformation, feature engineering |
| `numpy` | Numerical operations |
| `pyarrow` | Reading/writing Parquet files |
| `boto3` | Talking to AWS S3 |
| `scikit-learn` | Baselines, Random Forest, time-series CV, metrics |
| `xgboost` (or `lightgbm`) | Gradient boosting model |
| `prophet` (optional) | Additional time-series model for comparison |
| `shap` (optional) | Model explainability |
| `streamlit` | Dashboard (frontend + backend in Python) |
| `plotly` | Interactive charts |
| `pyyaml` | Reading the config file |
| `joblib` | Saving/loading models |
| `tenacity` | Retry logic for API calls |
| `pytest` | Unit tests |
| `python-dotenv` | Local environment variables (laptop only) |
| `pandera` or custom checks | Data validation |

### 3.3 Tools
| Tool | Use |
|---|---|
| Git + GitHub | Version control, portfolio |
| VS Code | Editor |
| AWS Console | Resource setup and screenshots |
| AWS CLI (optional) | Command-line AWS control |
| Draw.io / Excalidraw | Architecture diagram |
| tmux / systemd | Keeping the dashboard running |
| Docker (optional) | Containerization upgrade |

### 3.4 External services
| Service | Purpose | Cost |
|---|---|---|
| Open-Meteo Air Quality API | Pollutants and AQI | Free for non-commercial use, no key (verify current terms) |
| Open-Meteo Weather APIs (archive + forecast) | Temperature, wind, humidity, rain, pressure | Free for non-commercial use |
| OpenAQ (optional) | Ground-station measurements for validation | Free, requires API key |

---

## 4. System Architecture

```
                  ┌───────────────────────────┐
                  │  Open-Meteo APIs          │
                  │  (air quality + weather)  │
                  └─────────────┬─────────────┘
                                │ HTTPS (daily, via cron)
                                ▼
┌───────────────────────────────────────────────────────────┐
│                   AWS EC2 (Ubuntu VM)                     │
│                                                           │
│  1. collect.py   → fetch data, retry, save raw            │
│  2. clean.py     → clean, validate, build processed set   │
│  3. features.py  → feature engineering                    │
│  4. train.py     → train + compare models, save best      │
│  5. predict.py   → generate 7-day forecasts               │
│  6. evaluate.py  → compare past forecasts to actuals      │
│  7. Streamlit dashboard (port 8501)                       │
└───────────────┬───────────────────────────────┬───────────┘
                │ boto3 (IAM role)              │ HTTP :8501
                ▼                               ▼
┌───────────────────────────────┐        ┌──────────────┐
│          AWS S3 bucket        │        │   End users  │
│  raw/        processed/       │        │  (browser)   │
│  models/     predictions/     │        └──────────────┘
│  metrics/    logs/            │
└───────────────────────────────┘

Cross-cutting:  IAM role · Security Group · CloudWatch · AWS Budgets · cron/systemd
```

### 4.1 Data flow (daily)
1. **01:00 IST** cron triggers `collect.py`. Latest air quality + weather for each city is fetched.
2. Raw JSON is saved to `s3://<bucket>/raw/...` untouched.
3. `clean.py` merges, cleans, validates, and writes the processed Parquet dataset.
4. `predict.py` loads the current best model and writes new 7-day forecasts.
5. `evaluate.py` compares forecasts made earlier with the actual values that have now arrived and updates metrics.
6. Weekly, `train.py` retrains all models, compares them, and promotes the best one.
7. The dashboard reads processed data, forecasts, and metrics from S3 and shows them.

---

## 5. Data Source Details

### 5.1 Air quality API (Open-Meteo)
- Endpoint: `https://air-quality-api.open-meteo.com/v1/air-quality`
- Key parameters: `latitude`, `longitude`, `hourly`, `past_days`, `forecast_days`, `timezone`
- Useful hourly variables: `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`, `us_aqi`
- Example request:
```
https://air-quality-api.open-meteo.com/v1/air-quality
  ?latitude=28.6139&longitude=77.2090
  &hourly=pm10,pm2_5,nitrogen_dioxide,ozone,us_aqi
  &past_days=7&timezone=Asia%2FKolkata
```

### 5.2 Weather APIs (Open-Meteo)
- Historical: `https://archive-api.open-meteo.com/v1/archive` (start/end dates)
- Forecast: `https://api.open-meteo.com/v1/forecast` (also supports `past_days`)
- Variables: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `wind_direction_10m`, `precipitation`, `surface_pressure`, `cloud_cover`

### 5.3 Important notes on the data
- Open-Meteo air quality values are **model-based estimates (CAMS atmospheric model)**, not readings from a physical sensor at your exact location. State this honestly in your report.
- How far back historical air quality data goes depends on the API and region. Check the Open-Meteo documentation on day 1 and use the longest history available. A longer history gives better models.
- `us_aqi` is the US AQI scale. India's official CPCB AQI uses a different scale. Mention this clearly. You can optionally compute a CPCB-style AQI from pollutant values as an extra feature.
- Optional: use OpenAQ ground-station data for one or two cities as a sanity check on the model-based data.

### 5.4 Granularity decision
Hourly data is pulled, then **aggregated to daily values** (mean, max, min) for modelling. Daily forecasts for 7 days are the main target. This keeps the models simple and the dashboard readable.

---

## 6. Data Engineering Layer

### 6.1 Storage layers (S3 "data lake" layout)

```
s3://aqi-pipeline-<yourname>/
├── raw/
│   ├── air_quality/city=delhi/date=2026-11-14/data.json
│   └── weather/city=delhi/date=2026-11-14/data.json
├── processed/
│   ├── daily_dataset.parquet          # full cleaned dataset
│   └── features/features_latest.parquet
├── models/
│   ├── v20261114_0100/
│   │   ├── model_h1.joblib ... model_h7.joblib
│   │   └── metadata.json              # features, metrics, train dates
│   └── current.json                   # pointer to the active version
├── predictions/
│   └── date=2026-11-14/forecast.parquet
├── metrics/
│   ├── model_leaderboard.json
│   ├── accuracy_over_time.parquet
│   └── pipeline_status.json
└── logs/
```

Why layers?
- **raw/** is the untouched source of truth, so you can always reprocess.
- **processed/** is clean and analysis-ready.
- Partitioning by `city` and `date` makes data easy to find and cheap to manage.

### 6.2 Ingestion (`collect.py`)
- Reads the city list and parameters from `config.yaml`.
- Calls the APIs for each city, with **retries and exponential backoff** (`tenacity`) and timeouts.
- Saves the raw response to S3 exactly as received, plus a fetch timestamp.
- **Idempotent**: running it twice for the same date overwrites the same S3 key rather than creating duplicates.
- Logs successes and failures per city.
- A one-time `backfill.py` script downloads historical data to bootstrap the dataset.

### 6.3 Cleaning (`clean.py`)
| Problem | Handling |
|---|---|
| Missing hourly values | Short gaps interpolated; long gaps flagged and left missing |
| Duplicates | Dropped by key (city, timestamp) |
| Impossible values | Negative concentrations or AQI > 500 flagged/removed |
| Timezones | Everything converted to Asia/Kolkata (IST) |
| Units | Kept consistent (µg/m³ for pollutants) |
| Outliers / sensor-like spikes | Detected with rolling median/IQR rules and flagged, not blindly deleted |

Then hourly data is aggregated to daily (mean, max, min per variable) and merged with weather.

### 6.4 Validation (data quality checks)
Run automatically after each ingestion. Examples:
- Schema check: expected columns and types are present.
- Range check: AQI between 0 and 500; humidity between 0 and 100.
- Freshness check: today's (or yesterday's) data exists for every city.
- Completeness check: fewer than X% missing values per city per day.
- Volume check: row count is within the expected range.

Results are written to `metrics/pipeline_status.json` and shown on the dashboard. On failure, the pipeline logs the error and (optionally) sends an email through AWS SNS.

### 6.5 Orchestration
Simple and explainable: **cron** runs a `run_pipeline.sh` that calls the scripts in order and stops on error. (An upgrade path to Apache Airflow or AWS Step Functions is listed in section 19.)

### 6.6 Logging
Python `logging` with timestamps. Logs are written to local files and optionally to CloudWatch Logs.

---

## 7. Machine Learning Layer

### 7.1 Problem definition
- **Task:** time-series forecasting, supervised regression.
- **Target:** daily mean US AQI (optionally also PM2.5) per city.
- **Horizon:** 1 to 7 days ahead.
- **Approach:** *direct multi-horizon forecasting*: one model per horizon (h = 1 ... 7), each trained to predict AQI `h` days ahead. This avoids error accumulating from feeding predictions back in.

### 7.2 Feature engineering
| Group | Features |
|---|---|
| Lag features | AQI at t-1, t-2, t-3, t-7, t-14 |
| Rolling statistics | 3, 7, 14-day rolling mean, std, max |
| Trend | Difference between today and 7 days ago |
| Weather | Temperature, humidity, wind speed/direction, rain, pressure, cloud cover |
| Weather lags/rolling | Previous-day weather and 3-day averages |
| Calendar | Day of week, month, day of year, is_weekend |
| Seasonal | Season (winter / summer / monsoon / post-monsoon), Diwali-period and crop-burning-season flag (Oct to Nov) |
| Pollutants | PM2.5, PM10, NO2, O3 lags (as available) |
| City | City encoded (one-hot) if one global model is trained, or separate models per city |

**Leakage prevention:** every feature uses only information that would be available at forecast time. For future weather, the weather **forecast** is used in production. During training, historical weather stands in for the forecast, which makes offline scores slightly optimistic. Mention this in your report.

### 7.3 Models to compare
| Model | Why include it |
|---|---|
| **Naive persistence** ("tomorrow = today") | Baseline every model must beat |
| **Seasonal naive** ("same as 7 days ago") | Second baseline |
| **Ridge / Linear regression** | Simple, interpretable benchmark |
| **Random Forest** | Strong non-linear model, robust |
| **XGBoost / LightGBM** | Usually the best on tabular time-series features |
| **Prophet** (optional) | Classical decomposition-based forecaster |

### 7.4 Validation strategy
- **Never use random train/test splits** on time series.
- Use **`TimeSeriesSplit`** (expanding window) or a rolling-origin evaluation.
- Final hold-out: the most recent 10 to 15% of the timeline.
- Light hyperparameter tuning (`RandomizedSearchCV` with time-series CV, or Optuna if you want an extra).

### 7.5 Metrics
| Metric | Meaning |
|---|---|
| MAE | Average absolute error in AQI points (easy to explain) |
| RMSE | Penalizes large errors |
| MAPE / sMAPE | Percentage error (watch out for near-zero values) |
| Skill score vs baseline | `1 - MAE_model / MAE_baseline`; this is the headline number for your resume |
| Category accuracy | Whether the predicted AQI category (Good, Moderate, Poor...) is correct |

Report metrics **per horizon** (h=1 to h=7). Error grows with horizon, and showing that is a sign of maturity.

### 7.6 Model selection and versioning
1. `train.py` trains every candidate and writes a leaderboard.
2. The best model per horizon (lowest validation MAE) is chosen.
3. Models and `metadata.json` (feature list, training date range, metrics) are saved to `models/vYYYYMMDD_HHMM/`.
4. `models/current.json` is updated to point at the new version, **only if the new model beats the current one** (a simple champion/challenger rule).
5. Old versions are kept for rollback.

### 7.7 Retraining policy
- **Weekly** scheduled retraining.
- **Triggered** retraining if monitored error exceeds a threshold for several consecutive days.

### 7.8 Monitoring (the MLOps part)
- `evaluate.py` matches stored forecasts to actual values once they arrive.
- Rolling 7-day and 30-day MAE are saved in `metrics/accuracy_over_time.parquet`.
- **Drift check:** compare recent feature distributions to training distributions (for example Population Stability Index or a Kolmogorov-Smirnov test). Flag drift on the dashboard.

### 7.9 Explainability
- Feature importance from tree models.
- Optional SHAP summary plot to show which factors (wind, temperature, last week's AQI) drive predictions.

---

## 8. Dashboard (Streamlit)

Multi-page Streamlit app. The AQI display is intentionally only one small part. The rest proves the engineering and ML work.

| Page | Contents |
|---|---|
| **1. Overview** | City selector, current AQI with category colour, 7-day forecast chart with uncertainty band, quick city comparison |
| **2. Trends & EDA** | Historical AQI, seasonal patterns, heatmap of AQI by month and city, correlation with weather |
| **3. Forecast vs Actual** | Overlay of what the model predicted versus what actually happened, per horizon |
| **4. Model Leaderboard** | Table of all models with MAE, RMSE, skill score vs baseline, per horizon |
| **5. Explainability** | Feature importance / SHAP, "what drives AQI in this city" |
| **6. Accuracy Over Time** | Rolling error, drift indicators, retraining history |
| **7. Pipeline Status** | Last successful run, records collected today, data-quality check results, S3 data size, model version in use |
| **8. About / Architecture** | Architecture diagram and explanation (great for demos and viva) |

Implementation notes:
- Use `st.cache_data` with a time-to-live (for example 1 hour) when reading from S3, so the app is fast and S3 requests stay low.
- Read files with `boto3` (or `s3fs`) using the EC2 IAM role. No keys anywhere in code.
- Charts with Plotly for interactivity.
- A health-style sidebar shows the model version and last data update.

---

## 9. Project Folder Structure

```
aqi-forecast-pipeline/
├── README.md
├── requirements.txt
├── config.yaml                # cities, API params, S3 bucket, thresholds
├── .gitignore                 # includes .env, data/, venv/
├── .env.example               # local-only variables
├── Makefile                   # shortcuts: make train, make run, make test
├── src/
│   ├── __init__.py
│   ├── config.py              # loads config.yaml
│   ├── storage.py             # S3 + local read/write helpers (boto3)
│   ├── collect.py             # ingestion
│   ├── backfill.py            # historical bootstrap
│   ├── clean.py               # cleaning + aggregation
│   ├── validate.py            # data-quality checks
│   ├── features.py            # feature engineering
│   ├── train.py               # train, compare, version models
│   ├── predict.py             # produce forecasts
│   ├── evaluate.py            # forecast vs actual + drift
│   └── utils.py               # logging, retries, date helpers
├── app/
│   ├── Home.py                # Streamlit entry point
│   └── pages/
│       ├── 1_Trends.py
│       ├── 2_Forecast_vs_Actual.py
│       ├── 3_Model_Leaderboard.py
│       ├── 4_Explainability.py
│       ├── 5_Accuracy_Over_Time.py
│       └── 6_Pipeline_Status.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_experiments.ipynb
├── tests/
│   ├── test_clean.py
│   ├── test_features.py
│   └── test_validate.py
├── scripts/
│   ├── run_pipeline.sh        # daily job
│   ├── retrain.sh             # weekly job
│   └── setup_ec2.sh           # server bootstrap
├── deploy/
│   ├── aqi-dashboard.service  # systemd unit
│   ├── crontab.txt
│   ├── iam_policy.json
│   └── nginx.conf             # optional
├── docs/
│   ├── architecture.png
│   └── screenshots/
└── Dockerfile                 # optional
```

### 9.1 `config.yaml` (example)
```yaml
cities:
  - {name: Delhi,     lat: 28.6139, lon: 77.2090}
  - {name: Mumbai,    lat: 19.0760, lon: 72.8777}
  - {name: Ahmedabad, lat: 23.0225, lon: 72.5714}
  - {name: Bhavnagar, lat: 21.7645, lon: 72.1519}
  - {name: Kolkata,   lat: 22.5726, lon: 88.3639}
  - {name: Bengaluru, lat: 12.9716, lon: 77.5946}

timezone: Asia/Kolkata
aws:
  region: ap-south-1
  bucket: aqi-pipeline-yourname
forecast:
  horizons: [1, 2, 3, 4, 5, 6, 7]
  target: us_aqi_mean
training:
  test_fraction: 0.15
  n_splits: 5
monitoring:
  retrain_if_mae_above: 25
```

### 9.2 `requirements.txt` (starting point)
```
pandas
numpy
pyarrow
requests
tenacity
boto3
pyyaml
scikit-learn
xgboost
joblib
streamlit
plotly
shap
pytest
python-dotenv
# optional: prophet, pandera, lightgbm
```
Pin exact versions once the project works (`pip freeze`).

---

## 10. AWS Services: What, Why, How

### 10.1 Services in the core design

| Service | Role in this project | How it is used |
|---|---|---|
| **EC2** | Compute | One Ubuntu VM runs collector, training, prediction, and dashboard |
| **EBS** | VM disk | Root volume (gp3, ~20 to 30 GB) attached to EC2 |
| **S3** | Object storage | Data lake: raw, processed, models, predictions, metrics |
| **IAM** | Access control | EC2 role allows only this bucket (no access keys in code) |
| **VPC / Security Group** | Network security | Opens only ports 22 (your IP) and 8501 (dashboard) |
| **Elastic IP** | Stable address | Keeps the public IP fixed across stop/start (note: public IPv4 addresses may carry a small hourly charge; check current pricing) |
| **CloudWatch** | Monitoring | CPU, network, disk metrics; optional logs; alarms |
| **AWS Budgets** | Cost control | Alert before credits are drained |
| **Cron (on EC2)** | Scheduling | Runs the daily and weekly jobs |

### 10.2 Optional services (resume boosters)
| Service | Use |
|---|---|
| **SNS** | Email alerts for failed pipeline runs or budget alerts |
| **Lambda + EventBridge** | Serverless replacement for cron-based collection |
| **Route 53 + ACM + ALB/Nginx** | Custom domain and HTTPS |
| **ECR** | Store Docker image |
| **Systems Manager Parameter Store** | Store configuration values securely |
| **Athena** | Run SQL queries directly on the Parquet files in S3 |

### 10.3 Recommended region
`ap-south-1` (Mumbai) gives low latency from India. Compare prices with `us-east-1` before deciding, since regions differ in cost. Stay in one region for everything.

### 10.4 Instance choice
| Option | Notes |
|---|---|
| `t3.micro` (1 GB RAM) | Cheapest. May struggle with XGBoost/Prophet training and Streamlit at the same time. Add swap space if used. |
| **`t3.small` (2 GB RAM)** | **Recommended.** Comfortable for this project. |
| `t3.medium` (4 GB RAM) | Only if you add Prophet or SHAP-heavy jobs and notice memory pressure. |

Instance types available on the Free plan depend on when the account was created. For accounts created on or after 15 July 2025, AWS lists `t3.micro`, `t3.small`, `t4g.micro`, `t4g.small`, `c7i-flex.large` and `m7i-flex.large` as free-tier eligible. Confirm in your own console (the instance launch screen marks eligible types).

Tip: training can also be done on your laptop. You then only upload the model to S3, and the VM stays small. For the assignment, though, running training on EC2 is a stronger demonstration.

---

## 11. AWS Account, Credits and Cost Plan

### 11.1 How the current AWS free plan works (verify at sign-up)
- New accounts receive **$100 in credits** at sign-up, and can earn **up to $100 more** by completing onboarding activities (for example launching an EC2 instance, creating a budget alert, and trying services such as RDS or Lambda).
- **Free account plan:** you are not charged; the plan lasts **up to 6 months or until credits run out, whichever comes first**, and then the account is **closed automatically** unless you upgrade to the paid plan. Some services are restricted.
- **Paid account plan:** full service access; credits are applied automatically to bills, and you pay normal rates after credits end.
- Credits expire **12 months** after account creation.
- AWS also keeps a set of always-free services with monthly limits.

> **Rules change.** Read the plan details on the AWS sign-up page and the AWS Free Tier page when you register, and follow what they say.

### 11.2 What this means for your project
1. **Your project must be finished and demonstrated inside the 6-month window**, and on the Free plan the account closes when that ends or when credits run out. Plan accordingly.
2. **Before the account closes, save your proof:** screenshots, a screen-recorded demo video, the architecture diagram, an export of key S3 data, and the full code on GitHub. This is what makes the project resume-worthy even after AWS shuts down.
3. If you want the live site running long-term, you would need to upgrade to a paid plan and accept normal charges, or move the dashboard to a free host (for example Streamlit Community Cloud) reading from a small public dataset.

### 11.3 Rough cost estimate (always check the AWS Pricing pages)
These are ballpark figures for planning only:

| Item | Approximate cost |
|---|---|
| EC2 `t3.small`, running 24/7 | roughly $15 to $20 / month |
| EC2 `t3.micro`, running 24/7 | roughly $8 to $10 / month |
| EBS 20 to 30 GB gp3 | roughly $2 to $3 / month |
| S3 (a few GB, light requests) | well under $1 / month |
| CloudWatch (basic metrics) | free; extra logs/alarms are cheap |
| Data transfer out | small for a class demo (first GB-range is low/free) |
| Public IPv4 address | small hourly charge; check current pricing |

A 24/7 `t3.small` for 3 to 4 months lands comfortably inside $100, especially if you **stop the instance when you aren't using it**. Compute is the biggest cost, which is why stopping the VM matters.

### 11.4 Cost-control rules
1. Create an **AWS Budget** (for example $20/month) with email alerts at 50%, 80%, 100% on day one (this is also one of the credit-earning activities).
2. Use **one region** only.
3. Stop the instance when not working on it. A stopped instance does not charge for compute, but the EBS volume and any Elastic IP still cost something.
4. Keep S3 clean with a **lifecycle rule** (for example, move raw data older than 90 days to a cheaper storage class or delete very old logs).
5. Avoid unneeded services such as NAT Gateways, large RDS instances, and unused load balancers, which are classic surprise-cost sources.
6. Check the **Billing and Cost Management** dashboard and Free Tier usage page weekly.
7. Enable MFA on the root account and don't use the root account for daily work.

---

## 12. Step-by-Step Build Plan (4 Weeks)

Steps 1 to 5 run entirely on your laptop for free. AWS is only needed at step 6.

### Week 1: Data foundation (laptop)
1. Create the GitHub repo, folder structure, virtual environment, `requirements.txt`.
2. Write `config.yaml` and `config.py`.
3. Write `collect.py` and `backfill.py`; download historical data for all cities (save locally first).
4. Write `clean.py` and `validate.py`; produce the daily dataset.
5. Notebook `01_eda.ipynb`: trends, seasonality, missingness, correlations. Save the best charts for the report.

### Week 2: Machine learning (laptop)
1. Write `features.py` (lags, rolling windows, weather, calendar).
2. Implement baselines, then Ridge, Random Forest, XGBoost (optionally Prophet).
3. Evaluate with `TimeSeriesSplit`; record MAE/RMSE per horizon.
4. Build model versioning and the champion/challenger logic in `train.py`.
5. Write `predict.py` and `evaluate.py`.

### Week 3: Dashboard + cloud storage
1. Build the Streamlit pages using local files.
2. Write `storage.py` so the same code can use local disk or S3 (switch with config).
3. **Create the AWS account**, set up the budget, S3 bucket, IAM role (section 13).
4. Upload the backfilled data and models to S3 and make the dashboard read from S3 locally to test.

### Week 4: Deploy, automate, polish
1. Launch EC2, deploy code, run the dashboard (section 14).
2. Set up cron jobs and systemd service (section 15).
3. Configure CloudWatch and alarms (section 16).
4. Let the pipeline run for several days so you have real collected data and a real accuracy history.
5. Write the README, capture screenshots, record a demo video, prepare the report.

---

## 13. AWS Setup Walkthrough (Detailed)

### Step 0: Account and security basics
1. Sign up at aws.amazon.com and choose the plan (Free account plan or Paid plan; see section 11).
2. Enable **MFA** on the root user.
3. Create an **IAM admin user** for daily use (or IAM Identity Center) and stop using root.
4. Choose the region **ap-south-1** (top right of the console).
5. Create the **budget alert** (Billing, then Budgets).

### Step 1: Create the S3 bucket
1. S3, then **Create bucket**.
2. Name: globally unique, for example `aqi-pipeline-<yourname>`.
3. Region: ap-south-1.
4. Keep **Block all public access = ON** (the dashboard reads data through the IAM role; the bucket never needs to be public).
5. Enable **default encryption** (SSE-S3).
6. Optional: enable **versioning**.
7. Add a **lifecycle rule** (for example, expire `logs/` after 60 days; transition old `raw/` data to an infrequent-access class after 90 days).

### Step 2: Create the IAM policy and role for EC2
Policy (least privilege, only this bucket). Save as `deploy/iam_policy.json`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ListBucket",
      "Effect": "Allow",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::aqi-pipeline-yourname"
    },
    {
      "Sid": "ObjectAccess",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::aqi-pipeline-yourname/*"
    }
  ]
}
```
1. IAM, then Policies, then **Create policy** (JSON tab), paste, name it `AqiPipelineS3Access`.
2. IAM, then Roles, then **Create role**, trusted entity **AWS service, EC2**.
3. Attach `AqiPipelineS3Access`. Also attach `CloudWatchAgentServerPolicy` if you will use the CloudWatch agent.
4. Name it `AqiPipelineEC2Role`.

### Step 3: Create a key pair
EC2, then Key Pairs, then **Create key pair** (RSA, `.pem`). Store it safely and never commit it to Git. On Linux/macOS run `chmod 400 key.pem`.

### Step 4: Create the security group
| Type | Port | Source | Reason |
|---|---|---|---|
| SSH | 22 | **My IP** only | Admin access |
| Custom TCP | 8501 | 0.0.0.0/0 | Public dashboard |
| (optional) HTTP/HTTPS | 80/443 | 0.0.0.0/0 | If you add Nginx/HTTPS |

Outbound: default allow-all (the VM needs to reach the APIs and S3).

### Step 5: Launch the EC2 instance
1. EC2, then **Launch instance**.
2. Name: `aqi-pipeline-server`.
3. AMI: **Ubuntu Server 22.04 or 24.04 LTS**.
4. Instance type: `t3.small` (or per section 10.4).
5. Key pair: the one you created.
6. Network: default VPC, auto-assign public IP, select your security group.
7. Storage: 20 to 30 GB **gp3**.
8. Advanced details, **IAM instance profile**: `AqiPipelineEC2Role`.
9. Launch.

### Step 6: (Optional) Elastic IP
EC2, then Elastic IPs, then Allocate and Associate with your instance. This keeps the public IP stable across restarts. Release it when you finish, so it doesn't continue to cost money.

---

## 14. Deployment on EC2

### 14.1 Connect and prepare the server
```bash
ssh -i key.pem ubuntu@<EC2-PUBLIC-IP>

sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git tmux htop
```

### 14.2 (Only if using 1 GB RAM) Add swap
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 14.3 Get the code and install dependencies
```bash
git clone https://github.com/<you>/aqi-forecast-pipeline.git
cd aqi-forecast-pipeline
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 14.4 Verify S3 access through the IAM role
```bash
python -c "import boto3; print(boto3.client('s3').list_objects_v2(Bucket='aqi-pipeline-yourname', MaxKeys=3))"
```
If this works with no credentials configured, the IAM role is set up correctly. Take a screenshot.

### 14.5 Run the first pipeline manually
```bash
python -m src.backfill        # only once
python -m src.collect
python -m src.clean
python -m src.train
python -m src.predict
```

### 14.6 Run the dashboard as a service (systemd)
`/etc/systemd/system/aqi-dashboard.service`:
```ini
[Unit]
Description=AQI Streamlit Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/aqi-forecast-pipeline
ExecStart=/home/ubuntu/aqi-forecast-pipeline/venv/bin/streamlit run app/Home.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now aqi-dashboard
sudo systemctl status aqi-dashboard
```
Open `http://<EC2-PUBLIC-IP>:8501` in your browser. Also test it from a phone on mobile data to prove public accessibility.

### 14.7 Updating the app later
```bash
cd ~/aqi-forecast-pipeline && git pull
source venv/bin/activate && pip install -r requirements.txt
sudo systemctl restart aqi-dashboard
```

---

## 15. Scheduling and Automation

EC2 servers run in UTC by default. IST = UTC + 5:30. Either set the server timezone (`sudo timedatectl set-timezone Asia/Kolkata`) or convert times.

### 15.1 `scripts/run_pipeline.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/aqi-forecast-pipeline
source venv/bin/activate

python -m src.collect
python -m src.clean
python -m src.validate
python -m src.predict
python -m src.evaluate
```

### 15.2 Crontab (`crontab -e`, with the server on IST)
```
# Daily pipeline at 01:00 IST
0 1 * * *  /home/ubuntu/aqi-forecast-pipeline/scripts/run_pipeline.sh >> /home/ubuntu/logs/pipeline.log 2>&1

# Weekly retraining, Sundays at 03:00 IST
0 3 * * 0  /home/ubuntu/aqi-forecast-pipeline/scripts/retrain.sh >> /home/ubuntu/logs/retrain.log 2>&1
```
Create the log directory first: `mkdir -p ~/logs`.

### 15.3 Failure handling
- `set -e` stops the script on the first error.
- Failures write to `pipeline_status.json` and logs.
- Optional: publish an **SNS** email on failure.
- Optional: a "heartbeat" check that flags if no successful run occurred in 36 hours.

---

## 16. Monitoring and Resource Management

### 16.1 What to show (this satisfies "basic resource management")
| Item | What to demonstrate |
|---|---|
| **CloudWatch metrics** | CPU utilization, network in/out, disk graphs for the EC2 instance |
| **CloudWatch alarm** | Alarm if CPU is above 80% for 10 minutes (and optionally status-check failure), with an email notification |
| **CloudWatch Logs (optional)** | Pipeline logs shipped via the CloudWatch agent |
| **Memory/disk metrics** | Need the CloudWatch agent (not collected by default) |
| **AWS Budgets** | Budget with email alerts, plus the cost explorer view |
| **Free Tier / credits page** | Remaining credits and usage |
| **S3 metrics and lifecycle** | Bucket size, lifecycle rule screenshot |
| **IAM** | Role and policy showing least privilege |
| **Security Group** | Restricted inbound rules |
| **Start/Stop/Terminate** | Demonstrate controlling the instance life cycle |

### 16.2 Scaling discussion (for the report and viva)
- **Vertical scaling:** change to a larger instance type (stop, change type, start).
- **Horizontal scaling:** multiple instances behind an Application Load Balancer with Auto Scaling (not required to build; explain how it would work).
- **Storage scaling:** S3 scales automatically.
- **Decoupling:** move scheduled jobs to Lambda/EventBridge and training to a separate instance or SageMaker so the dashboard stays responsive.

---

## 17. Security Practices

1. **No access keys in code or Git.** Use the EC2 IAM role locally on the server and `aws configure`/environment variables only on your laptop (never committed).
2. `.gitignore` includes `.env`, `*.pem`, `data/`, `venv/`.
3. **Least-privilege IAM** policy limited to one bucket.
4. **S3 Block Public Access** stays on; encryption on.
5. **SSH only from your IP**; key-based login only.
6. MFA on the root account; daily work through an IAM user.
7. Keep the OS patched (`apt upgrade`).
8. Optional: Nginx reverse proxy with HTTPS (Let's Encrypt) on a domain name; the dashboard then never exposes port 8501 directly.
9. The dashboard is read-only and has no user input that reaches a database or shell.

---

## 18. Testing and Quality

| Test type | Examples |
|---|---|
| Unit tests (`pytest`) | Cleaning removes negatives; duplicates dropped; lag features computed correctly; no future leakage in features |
| Data validation | Schema, range, freshness checks run on every ingestion |
| Model tests | Model beats the naive baseline; prediction output has the right shape and 7 horizons |
| Smoke test | Dashboard starts and each page loads without error |
| Code quality | `black`, `ruff`/`flake8`, type hints on key functions |
| Reproducibility | Fixed random seeds, pinned dependencies, config-driven runs |

Optional: GitHub Actions workflow that runs `pytest` on each push (a nice CI badge in the README).

---

## 19. Optional Upgrades (Resume Boosters)

Pick one or two *after* the core works.

| Upgrade | Value |
|---|---|
| **Docker** container for app and jobs | Shows containerization skills |
| **GitHub Actions CI/CD** (test, then deploy to EC2 over SSH) | Shows DevOps skills |
| **FastAPI prediction endpoint** (`/forecast?city=Delhi`) | Shows API design |
| **Lambda + EventBridge** instead of cron | Serverless architecture |
| **Athena queries** on S3 Parquet | Data lake skills |
| **Airflow / Prefect** orchestration | Workflow orchestration |
| **MLflow** experiment tracking | MLOps |
| **Prediction intervals** (quantile regression) | Uncertainty-aware ML |
| **Alerts** (email/Telegram when forecast AQI exceeds a threshold) | Real-world usefulness |
| **Nginx + HTTPS + custom domain** | Production-style deployment |
| **Terraform** for the infrastructure | Infrastructure as code |

---

## 20. What to Submit / Demo Checklist

**Documents and links**
- [ ] GitHub repository (clean README, architecture diagram, screenshots)
- [ ] Project report (section 21)
- [ ] Architecture diagram image
- [ ] Demo video (screen recording of the dashboard plus AWS console tour)

**Screenshots of AWS**
- [ ] EC2 instance (running, with instance type and public IP)
- [ ] Security group inbound rules
- [ ] IAM role and policy
- [ ] S3 bucket with folder structure and some files
- [ ] S3 lifecycle rule
- [ ] CloudWatch CPU graph and alarm
- [ ] AWS Budgets configuration
- [ ] Billing / credits page
- [ ] SSH session showing cron jobs and the systemd service status

**Live demo flow (5 to 7 minutes)**
1. Show the architecture diagram.
2. Open the public URL (also on a phone using mobile data).
3. Walk through the dashboard pages: forecast, forecast vs actual, leaderboard, pipeline status.
4. Show S3 data folders growing daily.
5. SSH into EC2: show cron, logs, service status.
6. Show CloudWatch, IAM, security group, budget.
7. Explain scaling and cost management.

---

## 21. Report Outline

1. **Title page and abstract**
2. **Introduction:** problem (air pollution in India), motivation, objectives
3. **Cloud computing concepts used:** IaaS, object storage, IAM, elasticity, pay-as-you-go, shared responsibility model
4. **System design:** architecture diagram, data flow, why each AWS service was chosen
5. **Data:** sources, variables, collection method, limitations
6. **Data engineering:** ingestion, storage layers, cleaning, validation, automation
7. **Machine learning:** features, models, validation strategy, results (tables and charts), explainability
8. **Deployment:** AWS configuration steps with screenshots
9. **Monitoring, security and cost management**
10. **Results and discussion:** model performance, dashboard, lessons learned
11. **Challenges and solutions**
12. **Limitations and future work**
13. **Conclusion**
14. **References** (AWS docs, Open-Meteo docs, scikit-learn, etc.)
15. **Appendix:** key code snippets, config files, full screenshots

---

## 22. Resume Content

### 22.1 Project title options
- *Automated Air Quality Forecasting Pipeline on AWS*
- *End-to-End AQI Forecasting System (Data Engineering + ML + Cloud Deployment)*

### 22.2 Resume bullets (replace the X values with your **real** results)
- Built an automated data pipeline ingesting daily air quality and weather data for 6 Indian cities via REST APIs into an AWS S3 data lake (raw and processed Parquet layers) using scheduled jobs on EC2.
- Implemented data validation, deduplication, and gap handling checks, with pipeline health tracked and surfaced on a dashboard.
- Engineered lag, rolling-window, weather, and seasonal features; compared baseline, Ridge, Random Forest, and XGBoost models with time-series cross-validation, improving 7-day forecast MAE by X% over a naive baseline.
- Automated weekly retraining with model versioning and champion/challenger promotion; monitored forecast accuracy and data drift over time.
- Deployed a public multi-page Streamlit dashboard on AWS EC2 with least-privilege IAM roles, restricted security groups, CloudWatch monitoring, and AWS Budgets cost alerts.

### 22.3 Skills line
`Python, pandas, scikit-learn, XGBoost, time-series forecasting, feature engineering, ETL / data pipelines, AWS (EC2, S3, IAM, CloudWatch, Budgets), Streamlit, Git, Linux, cron, systemd`

### 22.4 Portfolio tips
- A pinned GitHub repo with a polished README (badges, architecture image, GIF of the dashboard, results table).
- Include real numbers: data volume (rows, days collected), number of cities, MAE and skill score.
- Add a short demo video link, since the live AWS link may stop working when the account or credits end.
- Write a short LinkedIn or blog post explaining one interesting finding (for example, how Diwali and winter affect AQI in Delhi).

---

## 23. Likely Viva Questions

| Question | Short answer direction |
|---|---|
| Why EC2 and not Lambda or Elastic Beanstalk? | Full control, simple to explain IaaS, runs long jobs and dashboard together; serverless is a future upgrade |
| Why S3 and not a database? | Cheap, durable object storage fits files, Parquet data lakes and model artifacts; analytics use case is batch, not transactional |
| How is the IAM role better than access keys? | Temporary credentials, nothing to leak, least privilege |
| Why time-series CV instead of random split? | Random splits leak future information into training |
| How do you avoid data leakage? | Features use only past information; weather forecast handled explicitly |
| Why are baselines important? | They prove the ML model adds value |
| How do you know when to retrain? | Weekly schedule plus error threshold and drift checks |
| What happens if the API fails? | Retries with backoff, logging, validation alert, idempotent re-run |
| How would you scale this? | Bigger instance, ALB + auto scaling, separate training job, serverless ingestion |
| What are the costs and how do you control them? | Budgets, stop instance, lifecycle rules, single region |
| What is the shared responsibility model? | AWS secures the infrastructure; you secure your data, IAM, OS, and application |
| Limitations? | Model-based data, not ground truth; limited history; training weather is not true forecast |

---

## 24. Risks and Honest Limitations

| Risk | Mitigation |
|---|---|
| Free plan account closes after 6 months or when credits are used | Finish early; save screenshots, video, GitHub code, and data exports; upgrade only if you want it live |
| Unexpected AWS charges | Budget alerts, stop instance, avoid unneeded services, check billing weekly |
| API changes or downtime | Retries, raw data stored, easy to swap data source |
| Model-based air quality data, not sensor data | Disclose in the report; optionally cross-check with OpenAQ |
| Short history limits model quality | Backfill as far as possible; report uncertainty honestly |
| Small instance runs out of memory | Use `t3.small`, add swap, train on laptop if needed |
| Public dashboard exposed | Read-only app, no secrets, restricted SSH, optional HTTPS |
| Scope creep | Finish the core first; upgrades come last |
| Open-Meteo terms (non-commercial use) | Fine for coursework; check terms before any commercial use |

---

## 25. Cleanup Checklist

When the assignment is graded (and you've saved all proof):
- [ ] Stop (or terminate) the EC2 instance
- [ ] Release any Elastic IP
- [ ] Delete unneeded EBS volumes and snapshots
- [ ] Empty and delete the S3 bucket (after downloading what you need)
- [ ] Delete unused IAM roles, users, and key pairs
- [ ] Remove CloudWatch alarms and log groups
- [ ] Confirm in Billing that nothing is still running

---

## 26. Glossary

| Term | Meaning |
|---|---|
| **AQI** | Air Quality Index, a number summarizing pollution level |
| **PM2.5 / PM10** | Fine particles under 2.5 / 10 micrometres |
| **EC2** | AWS virtual machines (IaaS) |
| **S3** | AWS object storage |
| **IAM** | AWS identity and access management |
| **Security Group** | Virtual firewall for an instance |
| **CloudWatch** | AWS monitoring and logging |
| **ETL / ELT** | Extract, Transform, Load pipelines |
| **Parquet** | Efficient columnar file format |
| **Data lake** | Storage of raw and processed data in files (here, S3) |
| **Idempotent** | Running twice gives the same result as running once |
| **Lag feature** | A past value used as an input feature |
| **Data leakage** | Using information in training that wouldn't exist at prediction time |
| **Baseline** | Simple model that every real model must beat |
| **MAE / RMSE** | Common error metrics |
| **Model drift** | Model getting worse because the data changed |
| **Champion/challenger** | Replace the live model only if the new one is better |
| **IaaS** | Infrastructure as a Service |
| **Elasticity** | Ability to scale resources up or down on demand |

---

*Next step: start with Week 1: repo setup, `config.yaml`, `collect.py`, `backfill.py`, and `clean.py`. These can all be built and tested on your laptop before touching AWS.*
