# Stock Quantitative Analysis & Automated Data Pipeline Platform

This repository contains the Big Data Mini Project for the **HCMUT-CO5135** course. It features a FastAPI-based quantitative analysis API and an automated Apache Airflow data ingestion pipeline designed for the Vietnamese stock market.

---

## 📂 Project Architecture

The project is structured into two main independent components:

```
.
├── quant_api/                  # FastAPI Quantitative Service
│   ├── api.py                  # API endpoints, quantitative metrics, backtests
│   ├── news_scoring.py         # Keyword, Hybrid, LSH news scoring algorithms
│   └── README.md               # API-specific instructions
│
└── data-pipeline/              # Airflow Ingestion & Embedding Pipeline
    ├── dags/                   # Orchestrated workflow definitions
    │   ├── assets_dimension_etl.py      # Weekly listed instruments ETL (VN master tickers)
    │   ├── market_news_morning.py       # News crawlers, sentiment analysis & embeddings
    │   ├── market_data_prices_daily.py  # Daily OHLCV price sync
    │   ├── refresh_historical_prices.py # Daily event-driven rebuilds (6-year history)
    │   └── etl_modules/                 # Transformers, adapters, cache and fetchers
    │
    ├── tests/                  # Unit and integration test suite
    ├── Dockerfile              # Custom Airflow image (includes webclaw and uv installer)
    ├── docker-compose.yml      # Multi-service Celery-executor data stack
    └── README.md               # Data pipeline details and operations
```

---

## 🚀 Part 1: Quantitative Analysis API (`quant_api/`)

A FastAPI quantitative analysis engine providing technical indicators, portfolio risk, backtesting, and AI-driven news impact recommendation.

### 1. Local Quick Start
Ensure python 3.12+ is installed, then run:
```bash
# Navigate and install dependencies
cd quant_api
pip install fastapi uvicorn vnstock ta scipy statsmodels scikit-learn \
            psycopg2-binary pgvector sentence-transformers python-dotenv supabase

# Setup configuration
cp .env.example .env
# Open .env and fill in your SUPABASE_URL and SUPABASE_SERVICE_KEY credentials

# Run FastAPI reload server
uvicorn quant_api.api:app --reload
```

### 2. Core API Endpoints for Evaluation
Once running, check the docs at `http://127.0.0.1:8000/docs`. Key endpoints include:
* **GET** `/api/v1/stock/{symbol}/price` — Fetches current and historical price series.
* **GET** `/api/v1/stock/{symbol}/eda` — Returns exploratory data stats (mean, volatility, skewness, kurtosis).
* **GET** `/api/v1/stock/{symbol}/technical` — Technical indicator series (RSI, MACD, Bollinger Bands, SMA/EMA).
* **GET** `/api/v1/stock/{symbol}/risk` — Volatility metrics, Value at Risk (VaR), and Conditional Value at Risk (CVaR).
* **GET** `/api/v1/stock/{symbol}/backtest` — Evaluates a SMA crossover or RSI breakout strategy on historical prices.
* **POST** `/api/v1/news/score/hybrid` — Evaluates financial news using a hybrid Keyword + TF-IDF model.
* **POST** `/api/v1/news/ticker-impact` — Extracts stock ticker impact recommendations from news text.

---

## ⚙️ Part 2: Airflow ETL Ingestion & Embedding Pipeline (`data-pipeline/`)

An Apache Airflow Celery data pipeline containerized for ingestion, sentiment scoring, and generating Gemini vector embeddings.

### 1. Pipeline Inventory
We migrated the following core workflows:
1. **`assets_dimension_etl`** (Weekly: Sunday 2 AM ICT)
   - Synchronizes listed instruments on the HOSE/HNX exchanges.
   - Refreshes master tickers database for 1,600+ VN symbols.
2. **`market_news_morning`** (Mon-Fri 7 AM ICT)
   - Crawls fresh stock market news using scrapers.
   - Batch scores sentiments using `gemini-2.0-flash`.
3. **`market_news_embedding`** (Post-Ingestion Task)
   - Embedded directly inside `market_news_morning.py` as `embed_news`.
   - Generates 768-d Gemini `text-embedding-004` vectors for ingested news and upserts to Supabase pgvector.
4. **`market_data_prices_daily`** (Mon-Fri 6 PM ICT)
   - Pulls EOD prices for active VN equities.
5. **`refresh_historical_prices`** (Daily 6:30 PM ICT)
   - Detects corporate events (splits, dividend adjustments) and triggers full 6-year history rebuilds.

### 2. Running the Pipeline locally
To evaluate the pipeline in a fully isolated container:
```bash
cd data-pipeline

# Configure environment variables
cp .env.example .env
# Edit .env and supply SUPABASE_DB_URL, GEMINI_API_KEY, and Telegram keys (if needed)

# Initialize database, bootstrap pools, and spin up Airflow Services
docker compose up -d --build
```
Airflow Web UI will be available at `http://localhost:8080` (Default login: `admin` / password configured in `.env`).

### 3. Running Validation & Testing
We have copied a selective unit and integration smoke test suite covering the 5 DAGs and their modules:
```bash
# Run tests inside the built container environment
docker compose run --rm airflow-worker pytest tests/unit

# Run import smoke checks
docker compose run --rm airflow-worker pytest tests/integration/test_dag_import_smoke.py
```

---

## 🛠️ Verification Checklist for Evaluators
1. **FastAPI Quantitative analysis endpoints** parse/retrieve data properly from the backend.
2. **DAGs parse cleanly**: `test_dag_import_smoke.py` verifies all 4 python DAG definitions load in the Airflow environment with zero parsing errors.
3. **ETL Modules isolation**: The `etl_modules/` folder is cleanly isolated at the root level, meaning no cross-service contamination.