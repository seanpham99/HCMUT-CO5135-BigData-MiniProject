# quant_api

FastAPI-based quantitative analysis API for Vietnamese stocks.

## Files
| File | Description |
|------|-------------|
| `api.py` | FastAPI app — stock price, EDA, technical indicators, risk, backtesting, news scoring, ticker-impact endpoints |
| `news_scoring.py` | Core news scoring library — Keyword Baseline (A), Hybrid Scorer (B), Random Projection LSH (C), TickerImpactRecommender |

## Quick start
```bash
pip install fastapi uvicorn vnstock ta scipy statsmodels scikit-learn \
            psycopg2-binary pgvector sentence-transformers python-dotenv supabase
cp .env.example .env   # fill in SUPABASE_URL, SUPABASE_SERVICE_KEY
uvicorn quant_api.api:app --reload
```

## API overview
| Group | Endpoints |
|-------|-----------|
| Stock Data | GET `/api/v1/stock/{symbol}/price` |
| Analysis | GET `/api/v1/stock/{symbol}/eda` · `technical` · `risk` · `backtest` · `analyze` |
| Market | GET `/api/v1/market/overview` · `correlation` · `contagion` · `portfolio/risk` |
| Embeddings | POST `/api/v1/embeddings/index/{symbol}` · GET `search` · `similar/{symbol}` |
| News Scoring | POST `/api/v1/news/score/keyword` · `hybrid` · `lsh/build` · `lsh/query` |
| Ticker Impact | POST `/api/v1/news/ticker-impact` · GET `/{news_id}/ticker-impact` |
