.PHONY: help install backfill collect clean validate train predict evaluate run-all test dashboard

help:
	@echo "AQI Forecasting Pipeline Make Commands:"
	@echo "  make install    - Install python requirements"
	@echo "  make backfill   - Bootstrap 90 days of historical raw data"
	@echo "  make collect    - Daily collection of AQ and weather"
	@echo "  make clean      - Clean and aggregate raw data to daily Parquet"
	@echo "  make validate   - Run automated data quality checks"
	@echo "  make train      - Train models and update champion"
	@echo "  make predict    - Generate 7-day multi-horizon forecasts"
	@echo "  make evaluate   - Evaluate ground truth error and drift"
	@echo "  make run-all    - Execute entire pipeline end-to-end"
	@echo "  make test       - Run pytest test suite"
	@echo "  make dashboard  - Launch Streamlit dashboard"

install:
	pip install -r requirements.txt

backfill:
	python -m src.backfill --days 90

collect:
	python -m src.collect

clean:
	python -m src.clean

validate:
	python -m src.validate

train:
	python -m src.train

predict:
	python -m src.predict

evaluate:
	python -m src.evaluate

run-all: collect clean validate predict evaluate

test:
	pytest tests/

dashboard:
	streamlit run app/Home.py
