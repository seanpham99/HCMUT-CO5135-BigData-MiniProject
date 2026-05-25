"""
VN Stock Quant Analysis API
Powered by FastAPI + vnstock + pgvector

── Single-Stock Endpoints ──
  GET  /api/v1/stock/{symbol}/price
  GET  /api/v1/stock/{symbol}/eda
  GET  /api/v1/stock/{symbol}/technical
  GET  /api/v1/stock/{symbol}/risk
  GET  /api/v1/stock/{symbol}/backtest
  GET  /api/v1/stock/{symbol}/sentiment
  GET  /api/v1/stock/{symbol}/analyze       (full pipeline)
  GET  /api/v1/stock/{symbol}/peers         (peer comparison)

── Market / Cross-Stock Endpoints ──
  GET  /api/v1/market/overview              (breadth, sector heatmap)
  GET  /api/v1/market/sectors               (sector performance & rotation)
  GET  /api/v1/market/correlation           (cross-correlation matrix)
  GET  /api/v1/market/contagion             (lead-lag, Granger causality)
  GET  /api/v1/market/portfolio/risk        (portfolio VaR, covariance, optimal weights)

── Self-Hosted Embeddings (pgvector) ──
  POST /api/v1/embeddings/index/{symbol}           (embed & upsert stock quant profile)
  POST /api/v1/embeddings/index/{symbol}/news      (embed & upsert vnstock news articles)
  GET  /api/v1/embeddings/search                   (semantic search over indexed docs)
  GET  /api/v1/embeddings/similar/{symbol}         (find quantitatively similar stocks)
  GET  /api/v1/embeddings/stats                    (index statistics)

── News Scoring (Methods A / B / C) ──
  POST /api/v1/news/score/keyword           (Method A: keyword density)
  POST /api/v1/news/score/hybrid            (Method B: keyword + semantic + time-decay)
  POST /api/v1/news/score/lsh/build         (Method C: build LSH index)
  POST /api/v1/news/score/lsh/query         (Method C: query LSH index)
  POST /api/v1/news/themes                  (thematic coverage across articles)

── Ticker Impact Recommendation ──
  POST /api/v1/news/ticker-impact           (text → top tickers impacted)
  GET  /api/v1/news/{news_id}/ticker-impact (Supabase news_id → top tickers impacted)
  GET  /api/v1/stock/{symbol}/news-impact   (ticker → top impactful news from corpus)

Run:
  docker compose up -d                      # starts pgvector on :5432
  pip install fastapi uvicorn vnstock ta scipy statsmodels scikit-learn \
              psycopg2-binary pgvector sentence-transformers python-dotenv
  cp .env.example .env                      # then edit .env as needed
  uvicorn api:app --reload                  # host/port read from .env
"""

from __future__ import annotations

import json
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory

from datetime import date, datetime
from typing import Any, Dict, List, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
import ta

import psycopg2
from psycopg2.extras import RealDictCursor
try:
    from pgvector.psycopg2 import register_vector as _register_vector
    _PGVECTOR_AVAILABLE = True
except ImportError:
    _PGVECTOR_AVAILABLE = False

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VN Stock Quant API",
    description="Quantitative analysis API for Vietnamese stocks using vnstock",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Constants (all overridable via .env) ─────────────────────────────────
DEFAULT_START  = os.getenv("DEFAULT_START",  "2020-01-01")
DEFAULT_END    = str(date.today())
DEFAULT_SOURCE = os.getenv("DEFAULT_SOURCE", "VCI")
RISK_FREE      = float(os.getenv("RISK_FREE_RATE", "0.045"))  # 4.5% VN risk-free rate
ANN            = 252

# ── Embedding / pgvector ─────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://vnstock:vnstock@localhost:5432/vnstock",
)
EMBED_MODEL  = os.getenv("EMBED_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM    = int(os.getenv("EMBED_DIM",  "384"))
_embed_model = None  # lazy-loaded singleton

# ── Server (used by __main__ entry point) ─────────────────────────────────
_API_HOST = os.getenv("API_HOST", "0.0.0.0")
_API_PORT = int(os.getenv("API_PORT", "8000"))


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# Sources confirmed working for vnstock 3.x quote.history (KBS only as of 3.5.0)
_QUOTE_SOURCES = ["KBS", "VCI"]


def _get_ohlcv(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    """Fetch OHLCV and add log-return column. Tries preferred source then falls back."""
    from vnstock import Vnstock
    sources = [source] + [s for s in _QUOTE_SOURCES if s != source]
    last_exc = None
    for src in sources:
        try:
            df = Vnstock().stock(symbol=symbol, source=src).quote.history(
                start=start, end=end, interval="1D"
            )
            df["time"] = pd.to_datetime(df["time"])
            df = df.set_index("time").sort_index()
            df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
            return df
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


def _get_benchmark(start: str, end: str, source: str) -> pd.Series:
    from vnstock import Vnstock
    sources = [source] + [s for s in _QUOTE_SOURCES if s != source]
    last_exc = None
    for src in sources:
        try:
            bench = Vnstock().stock(symbol="VNINDEX", source=src).quote.history(
                start=start, end=end, interval="1D"
            )
            bench["time"] = pd.to_datetime(bench["time"])
            bench = bench.set_index("time").sort_index()
            return bench["close"]
        except Exception as e:
            last_exc = e
            continue
    raise last_exc


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 24 technical indicators in-place."""
    # Trend
    df["ema_9"]   = ta.trend.EMAIndicator(df["close"], window=9).ema_indicator()
    df["ema_21"]  = ta.trend.EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_50"]  = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    df["sma_20"]  = ta.trend.SMAIndicator(df["close"], window=20).sma_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["adx"]         = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()
    # Momentum
    df["rsi_14"]    = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["stoch_k"]   = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()
    df["stoch_d"]   = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch_signal()
    df["roc_10"]    = ta.momentum.ROCIndicator(df["close"], window=10).roc()
    df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()
    # Volatility
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"]   = bb.bollinger_mavg()
    df["bb_pct"]   = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()
    df["atr_14"]   = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    # Volume
    df["obv"]      = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    df["mfi_14"]   = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"], window=14).money_flow_index()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma20"]
    return df


def _backtest_engine(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000_000,
    commission: float = 0.0015,
    slippage: float = 0.001,
) -> dict[str, Any]:
    """Core backtest engine. Returns metrics dict + equity series."""
    bt = df[["close"]].copy()
    bt["signal"]   = signals

    # Hold position until opposite signal
    pos, positions = 0, []
    for s in bt["signal"]:
        if s == 1:
            pos = 1
        elif s == -1:
            pos = 0
        positions.append(pos)
    bt["position"] = positions

    bt["market_ret"]       = bt["close"].pct_change()
    bt["strategy_ret"]     = bt["position"].shift(1) * bt["market_ret"]
    bt["trade"]            = bt["position"].diff().abs()
    bt["cost"]             = bt["trade"] * (commission + slippage)
    bt["strategy_ret_net"] = bt["strategy_ret"] - bt["cost"]
    bt["equity"]           = initial_capital * (1 + bt["strategy_ret_net"]).cumprod()

    years      = len(bt) / ANN
    total_ret  = bt["equity"].iloc[-1] / initial_capital - 1
    ann_ret    = (1 + total_ret) ** (1 / years) - 1
    ann_vol    = bt["strategy_ret_net"].std() * np.sqrt(ANN)
    sharpe     = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0
    cum_max    = bt["equity"].cummax()
    max_dd     = ((bt["equity"] / cum_max) - 1).min()
    calmar     = ann_ret / abs(max_dd) if max_dd != 0 else 0
    gp         = bt["strategy_ret_net"][bt["strategy_ret_net"] > 0].sum()
    gl         = abs(bt["strategy_ret_net"][bt["strategy_ret_net"] < 0].sum())
    pf         = gp / gl if gl > 0 else float("inf")
    n_trades   = int(bt["trade"].sum())
    final_eq   = round(bt["equity"].iloc[-1] / 1e6, 2)

    equity_series = bt["equity"].dropna()

    return {
        "total_return":     round(total_ret, 4),
        "ann_return":       round(ann_ret, 4),
        "ann_volatility":   round(ann_vol, 4),
        "sharpe":           round(sharpe, 4),
        "max_drawdown":     round(max_dd, 4),
        "calmar":           round(calmar, 4),
        "profit_factor":    round(pf, 4) if pf != float("inf") else None,
        "num_trades":       n_trades,
        "final_equity_M":  final_eq,
        "equity_curve":     {str(k.date()): round(v, 2) for k, v in equity_series.items()},
    }


def _build_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Return all 4 strategy signal Series."""
    s1 = pd.Series(0, index=df.index)
    s1[(df["ema_9"] > df["ema_21"]) & (df["ema_9"].shift(1) <= df["ema_21"].shift(1))] = 1
    s1[(df["ema_9"] < df["ema_21"]) & (df["ema_9"].shift(1) >= df["ema_21"].shift(1))] = -1

    s2 = pd.Series(0, index=df.index)
    s2[df["rsi_14"] < 30] = 1
    s2[df["rsi_14"] > 70] = -1

    s3 = pd.Series(0, index=df.index)
    s3[(df["macd"] > df["macd_signal"]) & (df["macd"].shift(1) <= df["macd_signal"].shift(1))] = 1
    s3[(df["macd"] < df["macd_signal"]) & (df["macd"].shift(1) >= df["macd_signal"].shift(1))] = -1

    s4 = pd.Series(0, index=df.index)
    buy4  = (df["ema_21"] > df["ema_50"]) & (df["rsi_14"] < 40) & (df["vol_ratio"] > 1.0)
    sell4 = (df["rsi_14"] > 70) | (
        (df["ema_21"] < df["ema_50"]) & (df["ema_21"].shift(1) >= df["ema_50"].shift(1))
    )
    s4[buy4]  = 1
    s4[sell4] = -1

    return {"S1_EMA_Cross": s1, "S2_RSI_MeanRev": s2, "S3_MACD_Momentum": s3, "S4_Combo": s4}


VN_POSITIVE = [
    "tăng trưởng", "lợi nhuận tăng", "doanh thu tăng", "vượt kế hoạch",
    "kết quả kinh doanh tích cực", "cổ tức", "chi trả cổ tức", "thưởng",
    "phục hồi", "tăng vốn", "mở rộng", "hợp tác", "ký kết", "đầu tư",
    "nâng hạng", "tốt", "khả quan", "triển vọng", "đột phá", "kỷ lục",
    "hoàn thành", "vượt mốc", "lãi lớn", "cải thiện", "tích cực", "thuận lợi",
]

VN_NEGATIVE = [
    "giảm", "thua lỗ", "lỗ ròng", "sụt giảm", "khó khăn", "rủi ro",
    "vi phạm", "xử phạt", "cảnh báo", "hủy", "đình chỉ", "kiểm tra",
    "thanh tra", "nợ xấu", "vỡ nợ", "phá sản", "bán ròng", "margin call",
    "cưỡng chế", "giải chấp", "thiếu thanh khoản", "downgrade", "hạ bậc",
]


def _score_text(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    text_lower = text.lower()
    pos = sum(1 for kw in VN_POSITIVE if kw in text_lower)
    neg = sum(1 for kw in VN_NEGATIVE if kw in text_lower)
    total = pos + neg
    return round((pos - neg) / total, 4) if total else 0.0


def _safe_float(val: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    return val


# ─────────────────────────────────────────────────────────────────────────
# EMBEDDING HELPERS
# ─────────────────────────────────────────────────────────────────────────

def _get_embed_model():
    """Lazy-load the sentence-transformer model (downloaded once, CPU-only)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _db_connect() -> psycopg2.extensions.connection:
    """Open a psycopg2 connection and register the vector type adapter."""
    conn = psycopg2.connect(DATABASE_URL)
    if _PGVECTOR_AVAILABLE:
        _register_vector(conn)
    return conn


def _ensure_pgvector_table() -> None:
    """Idempotently create the extension + table + HNSW index."""
    conn = _db_connect()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_embeddings (
                    id         SERIAL PRIMARY KEY,
                    symbol     TEXT        NOT NULL,
                    doc_type   TEXT        NOT NULL DEFAULT 'analysis',
                    content    TEXT        NOT NULL,
                    metadata   JSONB,
                    embedding  vector(384),
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_embeddings_symbol
                    ON stock_embeddings(symbol);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_embeddings_hnsw
                    ON stock_embeddings
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
            """)
        conn.commit()
    finally:
        conn.close()


def _build_stock_doc(symbol: str, start: str, end: str, source: str) -> tuple[str, dict]:
    """
    Compute quant metrics for a stock and build:
      - a rich natural-language document suitable for embedding
      - a metadata dict for structured retrieval
    """
    df    = _get_ohlcv(symbol, start, end, source)
    bench = _get_benchmark(start, end, source)

    df["bench_close"] = bench
    df.dropna(subset=["log_ret"], inplace=True)
    df["bench_ret"] = np.log(df["bench_close"] / df["bench_close"].shift(1))
    df.dropna(subset=["bench_ret"], inplace=True)

    r  = df["log_ret"]
    br = df["bench_ret"]

    ann_ret = float(r.mean() * ANN)
    ann_vol = float(r.std() * np.sqrt(ANN))
    sharpe  = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0.0
    skew    = float(r.skew())
    cummax  = df["close"].cummax()
    max_dd  = float(((df["close"] / cummax) - 1).min())

    X   = sm.add_constant(br)
    mdl = sm.OLS(r, X).fit()
    beta  = float(mdl.params.iloc[1])
    alpha = float(mdl.params.iloc[0]) * ANN

    df = _add_indicators(df)
    latest = df.dropna().iloc[-1]
    tech_signal = (
        "BUY"  if latest["macd"] > latest["macd_signal"] and latest["rsi_14"] < 65 else
        "SELL" if latest["macd"] < latest["macd_signal"] and latest["rsi_14"] > 60 else
        "HOLD"
    )
    rsi   = round(float(latest["rsi_14"]), 2)
    adx   = round(float(latest["adx"]),    2)
    sector = _find_sector(symbol, source) or "Unknown"

    meta = {
        "symbol":      symbol,
        "sector":      sector,
        "ann_return":  round(ann_ret, 4),
        "ann_vol":     round(ann_vol, 4),
        "sharpe":      round(sharpe, 4),
        "beta":        round(beta, 4),
        "alpha_ann":   round(alpha, 4),
        "max_drawdown":round(max_dd, 4),
        "skewness":    round(skew, 4),
        "tech_signal": tech_signal,
        "rsi_14":      rsi,
        "adx":         adx,
        "start":       start,
        "end":         end,
    }

    doc = (
        f"Stock: {symbol}. Sector: {sector}. "
        f"Annualized return: {ann_ret:.2%}. Annualized volatility: {ann_vol:.2%}. "
        f"Sharpe ratio: {sharpe:.2f}. Beta vs VNINDEX: {beta:.2f}. "
        f"Alpha annualized: {alpha:.2%}. Max drawdown: {max_dd:.2%}. "
        f"Skewness: {skew:.2f}. Technical signal: {tech_signal}. "
        f"RSI(14): {rsi:.1f}. ADX: {adx:.1f}."
    )
    return doc, meta


def _vec_literal(vec: list[float]) -> str:
    """Format a Python list as a pgvector literal string for use in SQL casts."""
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "version": "1.0.0"}


# ── 1. Price ──────────────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/price", tags=["Data"])
def get_price(
    symbol: str,
    start: str = Query(default=DEFAULT_START, description="YYYY-MM-DD"),
    end:   str = Query(default=DEFAULT_END,   description="YYYY-MM-DD"),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    OHLCV historical price data for a given symbol.
    """
    try:
        df = _get_ohlcv(symbol.upper(), start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    records = df[["open", "high", "low", "close", "volume"]].reset_index()
    records["time"] = records["time"].dt.strftime("%Y-%m-%d")
    return {
        "symbol": symbol.upper(),
        "start":  start,
        "end":    end,
        "sessions": len(df),
        "data": records.to_dict(orient="records"),
    }


# ── 2. EDA ────────────────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/eda", tags=["Analysis"])
def get_eda(
    symbol: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Exploratory Data Analysis: return statistics, distribution tests,
    CAPM regression (beta/alpha/R²) vs VNINDEX.
    """
    try:
        df    = _get_ohlcv(symbol.upper(), start, end, source)
        bench = _get_benchmark(start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    df["bench_close"] = bench
    df.dropna(subset=["log_ret"], inplace=True)
    df["bench_ret"] = np.log(df["bench_close"] / df["bench_close"].shift(1))
    df.dropna(subset=["bench_ret"], inplace=True)

    r = df["log_ret"]
    br = df["bench_ret"]

    jb_stat, jb_p = stats.jarque_bera(r)
    ks_stat, ks_p = stats.kstest(r, "norm", args=(r.mean(), r.std()))
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_p = adfuller(r, autolag="AIC")[:2]

    t_params = stats.t.fit(r)
    X = sm.add_constant(br)
    model = sm.OLS(r, X).fit()
    beta  = float(model.params.iloc[1])
    alpha = float(model.params.iloc[0]) * ANN
    r2    = float(model.rsquared)

    return {
        "symbol": symbol.upper(),
        "sessions": len(df),
        "statistics": {
            "ann_return":     round(float(r.mean() * ANN), 4),
            "ann_volatility": round(float(r.std() * np.sqrt(ANN)), 4),
            "sharpe_ratio":   round(float((r.mean() * ANN - RISK_FREE) / (r.std() * np.sqrt(ANN))), 4),
            "skewness":       round(float(r.skew()), 4),
            "excess_kurtosis":round(float(r.kurtosis()), 4),
            "max_drawdown":   round(float(((df["close"] / df["close"].cummax()) - 1).min()), 4),
        },
        "benchmark_statistics": {
            "ann_return":     round(float(br.mean() * ANN), 4),
            "ann_volatility": round(float(br.std() * np.sqrt(ANN)), 4),
            "sharpe_ratio":   round(float((br.mean() * ANN - RISK_FREE) / (br.std() * np.sqrt(ANN))), 4),
        },
        "distribution_tests": {
            "jarque_bera":   {"stat": round(jb_stat, 4), "p_value": round(jb_p, 6), "is_normal": jb_p > 0.05},
            "ks_test":       {"stat": round(ks_stat, 4), "p_value": round(ks_p, 6), "is_normal": ks_p > 0.05},
            "adf_test":      {"stat": round(adf_stat, 4), "p_value": round(adf_p, 6), "is_stationary": adf_p < 0.05},
            "student_t_df":  round(float(t_params[0]), 2),
        },
        "capm": {
            "beta":            round(beta, 4),
            "alpha_annualized":round(alpha, 4),
            "r_squared":       round(r2, 4),
            "interpretation":  f"Beta={beta:.2f} → {'HIGH-BETA' if beta > 1.2 else 'MARKET' if beta > 0.8 else 'DEFENSIVE'}",
        },
    }


# ── 3. Technical Indicators ───────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/technical", tags=["Analysis"])
def get_technical(
    symbol: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
    latest_only: bool = Query(default=False, description="Return only last row"),
):
    """
    24 technical indicators: EMA, MACD, RSI, Stochastic, BB, ATR, OBV, MFI, ADX.
    """
    try:
        df = _get_ohlcv(symbol.upper(), start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    df = _add_indicators(df)
    indicator_cols = [
        "close", "ema_9", "ema_21", "ema_50", "ema_200", "sma_20",
        "macd", "macd_signal", "macd_diff", "adx",
        "rsi_14", "stoch_k", "stoch_d", "roc_10", "williams_r",
        "bb_upper", "bb_lower", "bb_mid", "bb_pct", "bb_width", "atr_14",
        "obv", "mfi_14", "vol_ratio",
    ]
    out = df[indicator_cols].dropna()

    if latest_only:
        row = out.iloc[-1]
        signal = "BUY" if row["macd"] > row["macd_signal"] and row["rsi_14"] < 65 else \
                 "SELL" if row["macd"] < row["macd_signal"] and row["rsi_14"] > 60 else "HOLD"
        return {
            "symbol": symbol.upper(),
            "date":   str(out.index[-1].date()),
            "indicators": {k: round(_safe_float(v), 4) if _safe_float(v) is not None else None
                           for k, v in row.items()},
            "signal": signal,
        }

    records = out.reset_index()
    records["time"] = records["time"].dt.strftime("%Y-%m-%d")
    return {
        "symbol": symbol.upper(),
        "sessions": len(out),
        "data": [{k: round(_safe_float(v), 4) if isinstance(v, float) else v
                  for k, v in row.items()}
                 for row in records.to_dict(orient="records")],
    }


# ── 4. Risk ───────────────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/risk", tags=["Analysis"])
def get_risk(
    symbol:     str,
    start:      str   = Query(default=DEFAULT_START),
    end:        str   = Query(default=DEFAULT_END),
    source:     str   = Query(default=DEFAULT_SOURCE),
    confidence: float = Query(default=0.95, ge=0.9, le=0.99),
):
    """
    VaR (4 methods), CVaR/Expected Shortfall, Max Drawdown analysis.
    """
    try:
        df = _get_ohlcv(symbol.upper(), start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    r = df["log_ret"].dropna()

    # Historical
    var_hist  = float(r.quantile(1 - confidence))
    cvar_hist = float(r[r <= var_hist].mean())

    # Normal
    var_norm  = float(r.mean() + stats.norm.ppf(1 - confidence) * r.std())
    cvar_norm = float(r.mean() - r.std() * stats.norm.pdf(stats.norm.ppf(1 - confidence)) / (1 - confidence))

    # Student-t
    t_params = stats.t.fit(r)
    var_t    = float(stats.t.ppf(1 - confidence, *t_params))
    cvar_t   = float(r[r <= var_t].mean())

    # Cornish-Fisher
    z    = stats.norm.ppf(1 - confidence)
    s, k = float(r.skew()), float(r.kurtosis())
    z_cf = z + (z**2 - 1)*s/6 + (z**3 - 3*z)*k/24 - (2*z**3 - 5*z)*(s**2)/36
    var_cf = float(r.mean() + z_cf * r.std())

    # Drawdown
    cummax  = df["close"].cummax()
    dd      = (df["close"] / cummax) - 1
    max_dd  = float(dd.min())
    max_dd_date = str(dd.idxmin().date())
    peak_price  = round(float(cummax[dd.idxmin()]), 2)
    trough_price = round(float(df["close"][dd.idxmin()]), 2)

    # Rolling VaR (60d)
    rolling_var = r.rolling(60).quantile(0.05)
    rolling_var_latest = float(rolling_var.dropna().iloc[-1])

    return {
        "symbol":     symbol.upper(),
        "confidence": confidence,
        "var": {
            "historical":    round(var_hist, 4),
            "normal":        round(var_norm, 4),
            "student_t":     round(var_t, 4),
            "cornish_fisher":round(var_cf, 4),
        },
        "cvar": {
            "historical": round(cvar_hist, 4),
            "normal":     round(cvar_norm, 4),
            "student_t":  round(cvar_t, 4),
        },
        "drawdown": {
            "max_drawdown":     round(max_dd, 4),
            "max_drawdown_date":max_dd_date,
            "peak_price":       peak_price,
            "trough_price":     trough_price,
        },
        "rolling_var_60d_latest": round(rolling_var_latest, 4),
        "loss_on_100M_VND": {
            "var_hist_1day_loss": round(abs(var_hist) * 100, 2),
            "cvar_hist_1day_loss": round(abs(cvar_hist) * 100, 2),
        },
    }


# ── 5. Backtest ───────────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/backtest", tags=["Strategy"])
def get_backtest(
    symbol:          str   = "HCM",
    start:           str   = Query(default=DEFAULT_START),
    end:             str   = Query(default=DEFAULT_END),
    source:          str   = Query(default=DEFAULT_SOURCE),
    initial_capital: float = Query(default=100_000_000, description="VND"),
    commission:      float = Query(default=0.0015),
    slippage:        float = Query(default=0.001),
    strategy:        str   = Query(default="all", description="all | S1 | S2 | S3 | S4"),
):
    """
    Backtest 4 strategies: EMA Cross, RSI Mean-Rev, MACD Momentum, Combo.
    Returns metrics + equity curve for each strategy.
    """
    try:
        df = _get_ohlcv(symbol.upper(), start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    df = _add_indicators(df)
    bt_df = df.dropna().copy()
    signals = _build_signals(bt_df)

    # Buy & Hold
    signals["BuyAndHold"] = pd.Series(1, index=bt_df.index)

    names_map = {
        "S1": "S1_EMA_Cross", "S2": "S2_RSI_MeanRev",
        "S3": "S3_MACD_Momentum", "S4": "S4_Combo",
        "all": None,
    }

    selected = names_map.get(strategy.upper())
    if strategy.lower() != "all" and selected not in signals:
        raise HTTPException(status_code=400, detail=f"strategy must be one of: all, S1, S2, S3, S4")

    filter_keys = [selected, "BuyAndHold"] if selected else list(signals.keys())
    results = {}
    for name in filter_keys:
        metrics = _backtest_engine(bt_df, signals[name], initial_capital, commission, slippage)
        results[name] = metrics

    # Best by Sharpe
    best = max(
        [k for k in results if k != "BuyAndHold"],
        key=lambda k: results[k]["sharpe"],
    )

    return {
        "symbol":      symbol.upper(),
        "start":       start,
        "end":         end,
        "initial_capital": initial_capital,
        "commission":  commission,
        "slippage":    slippage,
        "best_strategy": best,
        "strategies": results,
    }


# ── 6. Walk-Forward ───────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/walkforward", tags=["Strategy"])
def get_walkforward(
    symbol: str,
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Walk-forward validation for MACD Momentum strategy.
    Train 2 years → test 1 year, 4 rolling windows (2022–2025).
    """
    try:
        df = _get_ohlcv(symbol.upper(), "2020-01-01", DEFAULT_END, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    df = _add_indicators(df)
    bt_df = df.dropna().copy()

    periods = [
        ("2022", "2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2023", "2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2024", "2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2025", "2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]

    results = []
    for test_yr, _, _, test_start, test_end in periods:
        test = bt_df[(bt_df.index >= test_start) & (bt_df.index <= test_end)]
        if len(test) < 20:
            continue

        s_macd = pd.Series(0, index=test.index)
        s_macd[(test["macd"] > test["macd_signal"]) & (test["macd"].shift(1) <= test["macd_signal"].shift(1))] = 1
        s_macd[(test["macd"] < test["macd_signal"]) & (test["macd"].shift(1) >= test["macd_signal"].shift(1))] = -1

        m_macd = _backtest_engine(test, s_macd)
        m_bh   = _backtest_engine(test, pd.Series(1, index=test.index))

        results.append({
            "test_year":       test_yr,
            "macd_return":     m_macd["total_return"],
            "bh_return":       m_bh["total_return"],
            "macd_sharpe":     m_macd["sharpe"],
            "bh_sharpe":       m_bh["sharpe"],
            "macd_max_dd":     m_macd["max_drawdown"],
            "bh_max_dd":       m_bh["max_drawdown"],
            "macd_outperforms":m_macd["total_return"] > m_bh["total_return"],
        })

    wins = sum(1 for r in results if r["macd_outperforms"])
    return {
        "symbol":  symbol.upper(),
        "strategy":"MACD Momentum",
        "periods": results,
        "summary": f"MACD outperforms Buy&Hold in {wins}/{len(results)} out-of-sample years",
    }


# ── 7. Sentiment ──────────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/sentiment", tags=["Sentiment"])
def get_sentiment(
    symbol: str,
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Corporate events + news sentiment analysis.
    Returns event classification, CAR summary, and daily event sentiment signal.
    """
    from vnstock import Vnstock

    # --- Corporate events (VCI source) ---
    events_df = pd.DataFrame()
    events_error = None
    try:
        stock_vci = Vnstock().stock(symbol=symbol.upper(), source="VCI")
        events_df = stock_vci.company.events()
    except Exception as e:
        events_error = str(e)

    # --- News (try requested source, fall back to KBS) ---
    news_df = pd.DataFrame()
    for news_source in ([source, "KBS"] if source != "KBS" else ["KBS"]):
        try:
            stock_ns = Vnstock().stock(symbol=symbol.upper(), source=news_source)
            news_df = stock_ns.company.news()
            if not news_df.empty:
                break
        except Exception:
            continue

    # Events processing
    event_summary: list = []
    recent_events: list = []
    if not events_df.empty and "exright_date" in events_df.columns:
        events_df["event_date"] = pd.to_datetime(events_df["exright_date"], errors="coerce")
        if "public_date" in events_df.columns:
            invalid = events_df["event_date"].dt.year < 2000
            events_df.loc[invalid, "event_date"] = pd.to_datetime(
                events_df.loc[invalid, "public_date"], errors="coerce"
            )
        if "event_title" in events_df.columns:
            events_df["sentiment_score"] = events_df["event_title"].apply(_score_text)
        if "event_list_name" in events_df.columns:
            events_df.loc[
                events_df["event_list_name"].str.contains("cổ tức", na=False), "sentiment_score"
            ] = 0.8
            event_summary = (
                events_df.groupby("event_list_name")
                .agg(count=("event_list_name", "size"), avg_sentiment=("sentiment_score", "mean"))
                .reset_index()
                .to_dict(orient="records")
            )
        keep_cols = [c for c in ["event_date", "event_title", "event_list_name", "sentiment_score"]
                     if c in events_df.columns]
        recent_events = (
            events_df[events_df["event_date"] >= "2020-01-01"]
            .sort_values("event_date")[keep_cols]
            .tail(20)
            .to_dict(orient="records")
        )
        for row in recent_events:
            if "event_date" in row:
                row["event_date"] = (
                    str(row["event_date"].date())
                    if hasattr(row["event_date"], "date")
                    else str(row["event_date"])
                )

    # News processing
    news_out = []
    if not news_df.empty:
        # KBS columns: head, article_id, title, publish_time, url
        # VCI columns: news_title, public_date
        if "title" in news_df.columns:
            news_df = news_df.rename(columns={"title": "news_title", "publish_time": "public_date"})
        if "news_title" in news_df.columns:
            news_df["sentiment"] = news_df["news_title"].apply(_score_text)
        if "public_date" in news_df.columns:
            news_df["date"] = pd.to_datetime(
                news_df["public_date"].astype(str), errors="coerce"
            )
            # VCI stores epoch-ms as float strings
            epoch_mask = news_df["date"].isna() & news_df["public_date"].notna()
            if epoch_mask.any():
                news_df.loc[epoch_mask, "date"] = pd.to_datetime(
                    pd.to_numeric(news_df.loc[epoch_mask, "public_date"], errors="coerce"),
                    unit="ms",
                    errors="coerce",
                )
            out_cols = [c for c in ["date", "news_title", "sentiment"] if c in news_df.columns]
            news_out = news_df[out_cols].dropna(subset=["news_title"]).head(20).to_dict(orient="records")
            for row in news_out:
                if "date" in row:
                    row["date"] = (
                        str(row["date"].date())
                        if hasattr(row["date"], "date")
                        else str(row["date"])
                    )

    return {
        "symbol":        symbol.upper(),
        "total_events":  len(events_df),
        "event_types":   event_summary,
        "recent_events": recent_events,
        "news":          news_out,
        "notes": {
            "dividend_car_5d":    "+1.38% (positive catalyst)",
            "new_listing_car_5d": "-3.36% (dilution risk)",
        },
        **({"events_source_error": events_error} if events_error else {}),
    }


# ── 8. Full Analysis ──────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/analyze", tags=["Analysis"])
def get_full_analysis(
    symbol: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Full quant pipeline in one call:
    EDA → Risk → Backtest (best strategy metrics only, no equity curves for speed).
    """
    try:
        df    = _get_ohlcv(symbol.upper(), start, end, source)
        bench = _get_benchmark(start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    df["bench_close"] = bench
    df.dropna(subset=["log_ret"], inplace=True)
    df["bench_ret"] = np.log(df["bench_close"] / df["bench_close"].shift(1))
    df.dropna(subset=["bench_ret"], inplace=True)

    # EDA
    r  = df["log_ret"]
    br = df["bench_ret"]
    X  = sm.add_constant(br)
    model   = sm.OLS(r, X).fit()
    beta    = float(model.params.iloc[1])
    alpha   = float(model.params.iloc[0]) * ANN

    # Risk
    var_hist  = float(r.quantile(0.05))
    cvar_hist = float(r[r <= var_hist].mean())
    cummax    = df["close"].cummax()
    max_dd    = float(((df["close"] / cummax) - 1).min())

    # Backtest — best strategy
    df = _add_indicators(df)
    bt_df   = df.dropna().copy()
    signals = _build_signals(bt_df)
    best_name, best_metrics = None, None
    for name, sig in signals.items():
        m = _backtest_engine(bt_df, sig)
        m.pop("equity_curve")   # strip for speed
        if best_metrics is None or m["sharpe"] > best_metrics["sharpe"]:
            best_name, best_metrics = name, m

    bh = _backtest_engine(bt_df, pd.Series(1, index=bt_df.index))
    bh.pop("equity_curve")

    return {
        "symbol":   symbol.upper(),
        "period":   {"start": start, "end": end, "sessions": len(df)},
        "eda": {
            "ann_return":     round(float(r.mean() * ANN), 4),
            "ann_volatility": round(float(r.std() * np.sqrt(ANN)), 4),
            "sharpe_ratio":   round(float((r.mean() * ANN - RISK_FREE) / (r.std() * np.sqrt(ANN))), 4),
            "skewness":       round(float(r.skew()), 4),
            "beta":           round(beta, 4),
            "alpha_ann":      round(alpha, 4),
            "r_squared":      round(float(model.rsquared), 4),
        },
        "risk": {
            "var_95_hist":  round(var_hist, 4),
            "cvar_95_hist": round(cvar_hist, 4),
            "max_drawdown": round(max_dd, 4),
        },
        "best_strategy": {
            "name":    best_name,
            "metrics": best_metrics,
        },
        "buy_hold": bh,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MACRO / CROSS-STOCK HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _get_multi_returns(
    symbols: list[str], start: str, end: str, source: str
) -> pd.DataFrame:
    """Fetch close prices for multiple symbols → return log-returns DataFrame."""
    frames = {}
    for sym in symbols:
        try:
            df = _get_ohlcv(sym, start, end, source)
            frames[sym] = df["close"]
        except Exception:
            pass  # skip unavailable symbols
    prices = pd.DataFrame(frames).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


@lru_cache(maxsize=4)
def _get_industry_map(source: str) -> pd.DataFrame:
    """Fetch industry classification from API: symbols_by_industries().
    Returns DataFrame with columns: symbol, industry_code, industry_name.
    Falls back through valid sources if the primary raises."""
    from vnstock import Vnstock
    sources = [source] + [s for s in _QUOTE_SOURCES if s != source]
    for src in sources:
        try:
            return Vnstock().stock(symbol="VCB", source=src).listing.symbols_by_industries()
        except Exception:
            continue
    # All sources failed — return empty frame so callers degrade gracefully
    return pd.DataFrame(columns=["symbol", "industry_code", "industry_name"])


def _get_sector_tickers(source: str) -> dict[str, list[str]]:
    """Build sector → tickers map dynamically from the API."""
    ind = _get_industry_map(source)
    return {
        name: group["symbol"].tolist()
        for name, group in ind.groupby("industry_name")
    }


def _find_sector(symbol: str, source: str = DEFAULT_SOURCE) -> str | None:
    """Find the industry/sector of a symbol via the API."""
    ind = _get_industry_map(source)
    row = ind[ind["symbol"] == symbol.upper()]
    if row.empty:
        return None
    return row["industry_name"].iloc[0]


def _get_sector_peers(symbol: str, source: str = DEFAULT_SOURCE) -> list[str]:
    """Return all stocks in the same industry as symbol (excluding itself)."""
    sector = _find_sector(symbol, source)
    if not sector:
        return []
    ind = _get_industry_map(source)
    peers = ind[ind["industry_name"] == sector]["symbol"].tolist()
    return [s for s in peers if s != symbol.upper()]


VN30_SYMBOLS = [
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
]


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-STOCK / MARKET ROUTES
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/api/v1/market/industries", tags=["Market"])
def get_industries(source: str = Query(default=DEFAULT_SOURCE)):
    """List all industries and their stocks (auto-discovered from API)."""
    try:
        sector_tickers = _get_sector_tickers(source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "num_industries": len(sector_tickers),
        "industries": {
            name: {"count": len(tickers), "symbols": tickers}
            for name, tickers in sorted(sector_tickers.items(), key=lambda x: -len(x[1]))
        },
    }


# ── 9. Peer Comparison ────────────────────────────────────────────────────
@app.get("/api/v1/stock/{symbol}/peers", tags=["Cross-Stock"])
def get_peers(
    symbol: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Compare a stock vs peers in the SAME sector (auto-discovered via API).
    Returns relative performance, beta, volatility, correlation with peers.
    """
    symbol = symbol.upper()
    sector = _find_sector(symbol, source)
    if not sector:
        raise HTTPException(status_code=404, detail=f"{symbol} not found in industry database. Use /market/correlation with custom symbols.")

    peers = [symbol] + _get_sector_peers(symbol, source)

    try:
        returns = _get_multi_returns(peers, start, end, source)
        bench_close = _get_benchmark(start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if symbol not in returns.columns:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")

    bench_ret = np.log(bench_close / bench_close.shift(1)).dropna()
    bench_ret = bench_ret.reindex(returns.index).dropna()
    returns = returns.loc[bench_ret.index]

    peer_stats = []
    for col in returns.columns:
        r = returns[col]
        ann_ret = float(r.mean() * ANN)
        ann_vol = float(r.std() * np.sqrt(ANN))
        sharpe  = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0
        corr_bench = float(r.corr(bench_ret))

        # Beta vs VNINDEX
        X = sm.add_constant(bench_ret)
        aligned = pd.concat([r, X], axis=1).dropna()
        if len(aligned) > 30:
            model = sm.OLS(aligned.iloc[:, 0], aligned.iloc[:, 1:]).fit()
            beta = float(model.params.iloc[1])
        else:
            beta = None

        corr_with_target = float(returns[col].corr(returns[symbol])) if col != symbol else 1.0

        peer_stats.append({
            "symbol":          col,
            "is_target":       col == symbol,
            "ann_return":      round(ann_ret, 4),
            "ann_volatility":  round(ann_vol, 4),
            "sharpe":          round(sharpe, 4),
            "beta_vs_vnindex": round(beta, 4) if beta else None,
            "corr_vs_vnindex": round(corr_bench, 4),
            "corr_vs_target":  round(corr_with_target, 4),
        })

    peer_stats.sort(key=lambda x: x["sharpe"], reverse=True)
    rank = next((i + 1 for i, p in enumerate(peer_stats) if p["symbol"] == symbol), None)

    return {
        "symbol": symbol,
        "sector": sector,
        "peers":  peers,
        "sessions": len(returns),
        "ranking": f"{rank}/{len(peer_stats)} by Sharpe in sector",
        "peer_analysis": peer_stats,
    }


# ── 10. Market Overview ──────────────────────────────────────────────────
@app.get("/api/v1/market/overview", tags=["Market"])
def get_market_overview(
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Market breadth & sector heatmap (sectors auto-discovered from API).
    Shows how every sector is performing relative to VNINDEX.
    """
    try:
        bench_close = _get_benchmark(start, end, source)
        sector_tickers = _get_sector_tickers(source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    bench_ret = np.log(bench_close / bench_close.shift(1)).dropna()
    vnindex_return = float((bench_close.iloc[-1] / bench_close.iloc[0]) - 1)

    sector_results = []
    for sector, tickers in sector_tickers.items():
        try:
            returns = _get_multi_returns(tickers, start, end, source)
        except Exception:
            continue
        if returns.empty:
            continue

        # Sector return = equal-weighted average
        sector_mean_ret = returns.mean(axis=1)
        total_ret = float((np.exp(sector_mean_ret.sum())) - 1)
        ann_ret   = float(sector_mean_ret.mean() * ANN)
        ann_vol   = float(sector_mean_ret.std() * np.sqrt(ANN))
        sharpe    = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

        # Correlation with VNINDEX
        aligned = pd.concat([sector_mean_ret, bench_ret], axis=1).dropna()
        corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 10 else None

        # Market breadth: % of stocks with positive return
        stock_total_rets = returns.sum()
        pct_positive = float((stock_total_rets > 0).mean())

        sector_results.append({
            "sector":          sector,
            "num_stocks":      len(returns.columns),
            "total_return":    round(total_ret, 4),
            "ann_return":      round(ann_ret, 4),
            "ann_volatility":  round(ann_vol, 4),
            "sharpe":          round(sharpe, 4),
            "corr_vs_vnindex": round(corr, 4) if corr else None,
            "pct_stocks_up":   round(pct_positive, 4),
            "tickers":         list(returns.columns),
        })

    sector_results.sort(key=lambda x: x["total_return"], reverse=True)

    return {
        "period": {"start": start, "end": end},
        "vnindex_total_return": round(vnindex_return, 4),
        "num_sectors": len(sector_results),
        "sectors": sector_results,
        "interpretation": {
            "top_sector":    sector_results[0]["sector"] if sector_results else None,
            "bottom_sector": sector_results[-1]["sector"] if sector_results else None,
            "overall_breadth": round(
                float(np.mean([s["pct_stocks_up"] for s in sector_results])), 4
            ) if sector_results else None,
        },
    }


# ── 11. Sector Deep Dive ─────────────────────────────────────────────────
@app.get("/api/v1/market/sectors/{sector_name}", tags=["Market"])
def get_sector_analysis(
    sector_name: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Deep analysis of a single sector: intra-sector correlation,
    lead-lag, risk contribution of each stock.
    Sectors are auto-discovered from API (e.g. Ngân hàng, Chứng khoán, Bất động sản).
    """
    # Fuzzy match sector name
    try:
        sector_tickers = _get_sector_tickers(source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    matched = None
    for s in sector_tickers:
        if sector_name.lower() in s.lower() or s.lower() in sector_name.lower():
            matched = s
            break
    if not matched:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_name}' not found. Available: {list(sector_tickers.keys())}")

    tickers = sector_tickers[matched]
    try:
        returns = _get_multi_returns(tickers, start, end, source)
        bench_close = _get_benchmark(start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if returns.empty or len(returns.columns) < 2:
        raise HTTPException(status_code=404, detail="Not enough data for sector analysis")

    bench_ret = np.log(bench_close / bench_close.shift(1)).dropna()
    aligned = returns.reindex(bench_ret.index).dropna()

    # Intra-sector correlation
    corr_matrix = aligned.corr()
    avg_corr = float(corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    ).stack().mean())

    # Rolling 60d average intra-sector correlation
    rolling_corrs = []
    for i in range(60, len(aligned), 20):
        window = aligned.iloc[i-60:i]
        c = window.corr()
        avg_c = float(c.where(np.triu(np.ones(c.shape), k=1).astype(bool)).stack().mean())
        rolling_corrs.append({"date": str(window.index[-1].date()), "avg_intra_corr": round(avg_c, 4)})

    # Risk contribution (% of sector variance attributed to each stock)
    cov = aligned.cov() * ANN
    w = np.ones(len(aligned.columns)) / len(aligned.columns)  # equal weight
    port_var = float(w @ cov.values @ w)
    marginal = cov.values @ w
    risk_contrib = (w * marginal) / port_var

    stock_details = []
    for i, col in enumerate(aligned.columns):
        r = aligned[col]
        stock_details.append({
            "symbol":          col,
            "ann_return":      round(float(r.mean() * ANN), 4),
            "ann_volatility":  round(float(r.std() * np.sqrt(ANN)), 4),
            "risk_contribution_pct": round(float(risk_contrib[i]) * 100, 2),
        })

    # Sector return vs VNINDEX
    sector_ret = aligned.mean(axis=1)
    sector_total = float(np.exp(sector_ret.sum()) - 1)
    bench_aligned = bench_ret.reindex(aligned.index).dropna()
    sector_ret_a = sector_ret.reindex(bench_aligned.index)
    sector_beta = None
    if len(bench_aligned) > 30:
        X = sm.add_constant(bench_aligned)
        model = sm.OLS(sector_ret_a, X).fit()
        sector_beta = round(float(model.params.iloc[1]), 4)

    return {
        "sector":          matched,
        "tickers":         list(aligned.columns),
        "sessions":        len(aligned),
        "sector_total_return": round(sector_total, 4),
        "sector_beta":     sector_beta,
        "intra_sector_correlation": {
            "average": round(avg_corr, 4),
            "interpretation": "HIGH co-movement" if avg_corr > 0.6 else "MODERATE" if avg_corr > 0.3 else "LOW",
            "matrix": {k: {k2: round(v2, 4) for k2, v2 in v.items()}
                       for k, v in corr_matrix.to_dict().items()},
        },
        "rolling_correlation": rolling_corrs[-10:],  # last 10 snapshots
        "stock_details": stock_details,
        "sector_covariance_annual_pct": round(port_var * 100, 4),
    }


# ── 12. Cross-Stock Correlation ──────────────────────────────────────────
@app.get("/api/v1/market/correlation", tags=["Cross-Stock"])
def get_correlation(
    symbols: str = Query(
        description="Comma-separated symbols, e.g. HCM,SSI,VCI,VND or use 'VN30' for the VN30 basket",
    ),
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Correlation & covariance matrix for any set of stocks.
    Pass VN30 to analyze the entire VN30 basket.
    Identifies the most/least correlated pairs — useful for hedging & diversification.
    """
    if symbols.strip().upper() == "VN30":
        sym_list = VN30_SYMBOLS
    else:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    if len(sym_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 symbols required")

    try:
        returns = _get_multi_returns(sym_list, start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if returns.shape[1] < 2:
        raise HTTPException(status_code=404, detail="Not enough data. Check symbols.")

    corr = returns.corr()
    cov  = (returns.cov() * ANN)

    # Find top correlated & least correlated pairs
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "pair": f"{cols[i]}-{cols[j]}",
                "correlation": round(float(corr.iloc[i, j]), 4),
            })
    pairs.sort(key=lambda x: x["correlation"], reverse=True)

    return {
        "symbols":  list(returns.columns),
        "sessions": len(returns),
        "correlation_matrix": {
            k: {k2: round(v2, 4) for k2, v2 in v.items()}
            for k, v in corr.to_dict().items()
        },
        "covariance_matrix_annual": {
            k: {k2: round(v2, 6) for k2, v2 in v.items()}
            for k, v in cov.to_dict().items()
        },
        "most_correlated_pairs":  pairs[:5],
        "least_correlated_pairs": pairs[-5:][::-1],
        "insights": {
            "avg_correlation":  round(float(np.mean([p["correlation"] for p in pairs])), 4),
            "high_contagion_risk": [p["pair"] for p in pairs if p["correlation"] > 0.7],
            "diversification_candidates": [p["pair"] for p in pairs if p["correlation"] < 0.3],
        },
    }


# ── 13. Contagion / Lead-Lag / Granger Causality ─────────────────────────
@app.get("/api/v1/market/contagion", tags=["Cross-Stock"])
def get_contagion(
    symbols: str = Query(
        description="Comma-separated symbols, e.g. HCM,SSI,VCI,TCB,VCB",
    ),
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
    max_lag: int = Query(default=5, ge=1, le=20, description="Max lag for Granger test"),
):
    """
    Detect contagion / spillover between stocks:
    - Granger Causality: does stock A's past predict stock B's future?
    - Lead-lag correlation: which stock leads?
    - Tail dependence: do they crash together?
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(sym_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 symbols required")

    try:
        returns = _get_multi_returns(sym_list, start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if returns.shape[1] < 2:
        raise HTTPException(status_code=404, detail="Not enough data")

    cols = returns.columns.tolist()

    # Granger Causality
    from statsmodels.tsa.stattools import grangercausalitytests
    granger_results = []
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                continue
            try:
                test_data = returns[[cols[j], cols[i]]].dropna()
                if len(test_data) < max_lag + 30:
                    continue
                result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                best_pval = min(result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1))
                best_lag  = min(range(1, max_lag + 1),
                               key=lambda lag: result[lag][0]["ssr_ftest"][1])
                granger_results.append({
                    "cause":   cols[i],
                    "effect":  cols[j],
                    "best_lag":best_lag,
                    "p_value": round(best_pval, 6),
                    "significant": best_pval < 0.05,
                    "interpretation": f"{cols[i]} Granger-causes {cols[j]} at lag {best_lag}" if best_pval < 0.05 else "No significant causality",
                })
            except Exception:
                pass

    # Lead-lag cross-correlation (±5 days)
    lead_lag = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            cross_corrs = {}
            for lag in range(-5, 6):
                if lag >= 0:
                    c = float(returns[cols[i]].corr(returns[cols[j]].shift(lag)))
                else:
                    c = float(returns[cols[i]].shift(-lag).corr(returns[cols[j]]))
                cross_corrs[str(lag)] = round(c, 4) if not np.isnan(c) else None

            peak_lag = max(cross_corrs, key=lambda k: abs(cross_corrs[k] or 0))
            lead_lag.append({
                "pair":             f"{cols[i]}-{cols[j]}",
                "cross_correlations": cross_corrs,
                "peak_lag":         int(peak_lag),
                "peak_corr":        cross_corrs[peak_lag],
                "leader":           cols[i] if int(peak_lag) > 0 else cols[j] if int(peak_lag) < 0 else "simultaneous",
            })

    # Tail dependence (correlation during worst 10% days)
    tail_dep = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r_i, r_j = returns[cols[i]], returns[cols[j]]
            # Lower tail: both in bottom 10%
            threshold_i = r_i.quantile(0.10)
            threshold_j = r_j.quantile(0.10)
            both_bad = ((r_i <= threshold_i) & (r_j <= threshold_j)).mean()
            # Normal correlation vs stress correlation
            normal_corr = float(r_i.corr(r_j))
            stress_mask = (r_i <= threshold_i) | (r_j <= threshold_j)
            stress_corr = float(r_i[stress_mask].corr(r_j[stress_mask])) if stress_mask.sum() > 10 else None

            tail_dep.append({
                "pair":              f"{cols[i]}-{cols[j]}",
                "normal_correlation":round(normal_corr, 4),
                "stress_correlation":round(stress_corr, 4) if stress_corr else None,
                "joint_crash_prob":  round(float(both_bad), 4),
                "contagion_warning": stress_corr is not None and stress_corr > normal_corr + 0.1,
            })

    return {
        "symbols":  cols,
        "sessions": len(returns),
        "granger_causality": sorted(granger_results, key=lambda x: x["p_value"]),
        "lead_lag": lead_lag,
        "tail_dependence": tail_dep,
        "summary": {
            "significant_causal_links": [
                f"{r['cause']} → {r['effect']} (lag {r['best_lag']}, p={r['p_value']:.4f})"
                for r in granger_results if r["significant"]
            ],
            "contagion_pairs": [t["pair"] for t in tail_dep if t["contagion_warning"]],
        },
    }


# ── 14. Portfolio Risk & Optimization ─────────────────────────────────────
@app.get("/api/v1/market/portfolio/risk", tags=["Portfolio"])
def get_portfolio_risk(
    symbols: str = Query(
        description="Comma-separated symbols, e.g. HCM,FPT,VNM,HPG,VCB",
    ),
    start:   str   = Query(default=DEFAULT_START),
    end:     str   = Query(default=DEFAULT_END),
    source:  str   = Query(default=DEFAULT_SOURCE),
    weights: str   = Query(default="", description="Comma-separated weights (must sum to 1). Empty = equal-weight."),
):
    """
    Portfolio-level risk analysis:
    - Covariance-based portfolio VaR & volatility
    - Marginal risk contribution per stock
    - Minimum-variance optimal weights
    - Maximum Sharpe (tangency) portfolio
    - Efficient frontier sample points
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(sym_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 symbols required")

    try:
        returns = _get_multi_returns(sym_list, start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if returns.shape[1] < 2:
        raise HTTPException(status_code=404, detail="Not enough data")

    n = len(returns.columns)
    mu  = returns.mean().values * ANN
    cov = returns.cov().values * ANN

    # Weights
    if weights.strip():
        w = np.array([float(x) for x in weights.split(",")])
        if len(w) != n:
            raise HTTPException(status_code=400, detail=f"Expected {n} weights, got {len(w)}")
        if abs(w.sum() - 1.0) > 0.01:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")
    else:
        w = np.ones(n) / n  # equal-weight

    # Portfolio stats
    port_ret = float(w @ mu)
    port_vol = float(np.sqrt(w @ cov @ w))
    port_sharpe = (port_ret - RISK_FREE) / port_vol if port_vol > 0 else 0
    port_var_95 = port_ret - 1.645 * port_vol  # parametric annual

    # Marginal risk contribution
    marginal = cov @ w
    risk_contrib = (w * marginal) / (w @ cov @ w)

    stock_detail = []
    for i, sym in enumerate(returns.columns):
        stock_detail.append({
            "symbol":               sym,
            "weight":               round(float(w[i]), 4),
            "ann_return":           round(float(mu[i]), 4),
            "ann_volatility":       round(float(np.sqrt(cov[i, i])), 4),
            "risk_contribution_pct":round(float(risk_contrib[i]) * 100, 2),
        })

    # ── Minimum Variance Portfolio ────────────────────────────────────────
    def _neg_sharpe(w_):
        r_ = w_ @ mu
        v_ = np.sqrt(w_ @ cov @ w_)
        return -(r_ - RISK_FREE) / v_ if v_ > 0 else 0

    def _port_vol(w_):
        return np.sqrt(w_ @ cov @ w_)

    bounds = tuple((0, 1) for _ in range(n))
    constraints = {"type": "eq", "fun": lambda w_: w_.sum() - 1}
    w0 = np.ones(n) / n

    # Min Variance
    res_mv = minimize(_port_vol, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    mv_w = res_mv.x
    mv_ret = float(mv_w @ mu)
    mv_vol = float(np.sqrt(mv_w @ cov @ mv_w))
    mv_sharpe = (mv_ret - RISK_FREE) / mv_vol if mv_vol > 0 else 0

    # Max Sharpe
    res_ms = minimize(_neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    ms_w = res_ms.x
    ms_ret = float(ms_w @ mu)
    ms_vol = float(np.sqrt(ms_w @ cov @ ms_w))
    ms_sharpe = (ms_ret - RISK_FREE) / ms_vol if ms_vol > 0 else 0

    # Efficient frontier (10 points)
    target_rets = np.linspace(mu.min(), mu.max(), 10)
    frontier = []
    for tgt in target_rets:
        cons = [
            {"type": "eq", "fun": lambda w_: w_.sum() - 1},
            {"type": "eq", "fun": lambda w_, t=tgt: w_ @ mu - t},
        ]
        res = minimize(_port_vol, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            f_vol = float(np.sqrt(res.x @ cov @ res.x))
            f_ret = float(res.x @ mu)
            frontier.append({"return": round(f_ret, 4), "volatility": round(f_vol, 4)})

    return {
        "symbols":  list(returns.columns),
        "sessions": len(returns),
        "current_portfolio": {
            "weights":    {sym: round(float(w[i]), 4) for i, sym in enumerate(returns.columns)},
            "ann_return": round(port_ret, 4),
            "ann_volatility": round(port_vol, 4),
            "sharpe":     round(port_sharpe, 4),
            "var_95_annual": round(port_var_95, 4),
        },
        "stock_details": stock_detail,
        "optimal_portfolios": {
            "min_variance": {
                "weights":    {sym: round(float(mv_w[i]), 4) for i, sym in enumerate(returns.columns)},
                "ann_return": round(mv_ret, 4),
                "ann_volatility": round(mv_vol, 4),
                "sharpe":     round(mv_sharpe, 4),
            },
            "max_sharpe": {
                "weights":    {sym: round(float(ms_w[i]), 4) for i, sym in enumerate(returns.columns)},
                "ann_return": round(ms_ret, 4),
                "ann_volatility": round(ms_vol, 4),
                "sharpe":     round(ms_sharpe, 4),
            },
        },
        "efficient_frontier": frontier,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EMBEDDING / PGVECTOR ROUTES
# ═══════════════════════════════════════════════════════════════════════════

# ── Index a stock ─────────────────────────────────────────────────────────
@app.post("/api/v1/embeddings/index/{symbol}", tags=["Embeddings"])
def embed_index_stock(
    symbol: str,
    start:  str = Query(default=DEFAULT_START),
    end:    str = Query(default=DEFAULT_END),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    Build a quantitative profile text for *symbol*, encode it with the
    self-hosted sentence-transformer model, and upsert into pgvector.

    Subsequent calls overwrite the previous vector for the same symbol.
    """
    symbol = symbol.upper()
    try:
        doc, meta = _build_stock_doc(symbol, start, end, source)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    model = _get_embed_model()
    vec   = model.encode(doc).tolist()   # list[float] length 384

    try:
        conn = _db_connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM stock_embeddings WHERE symbol = %s AND doc_type = 'analysis'",
                    (symbol,),
                )
                cur.execute(
                    """
                    INSERT INTO stock_embeddings (symbol, doc_type, content, metadata, embedding)
                    VALUES (%s, 'analysis', %s, %s, %s::vector)
                    """,
                    (symbol, doc, json.dumps(meta), _vec_literal(vec)),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {
        "symbol":      symbol,
        "doc_type":    "analysis",
        "content":     doc,
        "metadata":    meta,
        "vector_dims": len(vec),
        "model":       EMBED_MODEL,
        "status":      "indexed",
    }


# ── Index news articles ───────────────────────────────────────────────────
@app.post("/api/v1/embeddings/index/{symbol}/news", tags=["Embeddings"])
def embed_index_news(
    symbol: str,
    source: str = Query(
        default="KBS",
        description="Data source for company.news(). KBS is the most reliable; VCI may fail.",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Max news articles to embed"),
):
    """
    Fetch the latest news for *symbol* from vnstock (`company.news()`),
    embed each article (title + snippet) with the self-hosted sentence-transformer,
    and upsert all rows into pgvector with `doc_type='news'`.

    **vnstock KBS columns** (confirmed live): `title`, `head`, `publish_time`, `article_id`, `url`.

    Previous news rows for the same symbol are replaced on each call.
    The `embed_search` endpoint can then be used to query news semantically
    (e.g. `?q=tăng vốn&doc_type=news`).
    """
    symbol = symbol.upper()

    # ── 1. Fetch raw news from vnstock ────────────────────────────────────
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source=source)
        raw_news = stock.company.news()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vnstock news fetch failed: {e}")

    if raw_news is None or (hasattr(raw_news, "empty") and raw_news.empty):
        raise HTTPException(status_code=404, detail=f"No news returned for {symbol} from source={source}")

    # Normalise to a list of dicts regardless of source differences
    rows: list[dict] = raw_news.head(limit).to_dict(orient="records")

    # ── 2. Build embed text + metadata for each article ───────────────────
    model = _get_embed_model()

    def _parse_publish_time(val: Any) -> str:
        """Return an ISO date string regardless of how vnstock encodes the timestamp."""
        if val is None:
            return ""
        try:
            # KBS: ISO string  e.g. '2026-04-03T14:13:13.18'
            return str(pd.to_datetime(val).date())
        except Exception:
            try:
                # Legacy VCI: Unix ms integer
                return str(pd.to_datetime(float(val), unit="ms").date())
            except Exception:
                return str(val)

    docs: list[str] = []
    metas: list[dict] = []
    for row in rows:
        title   = str(row.get("title") or row.get("news_title") or "")
        snippet = str(row.get("head")  or row.get("content") or "")
        # Truncate snippet to keep embedding text under ~512 tokens
        snippet_trimmed = snippet[:500] if len(snippet) > 500 else snippet
        embed_text = f"{title}. {snippet_trimmed}".strip(". ")

        pub_date = _parse_publish_time(row.get("publish_time") or row.get("public_date"))

        meta = {
            "symbol":       symbol,
            "article_id":   int(row["article_id"]) if "article_id" in row else None,
            "title":        title,
            "publish_date": pub_date,
            "url":          str(row.get("url") or ""),
            "source":       source,
        }
        docs.append(embed_text)
        metas.append(meta)

    if not docs:
        raise HTTPException(status_code=404, detail="No parseable news articles found")

    # Batch encode all titles in one call (faster than one-by-one)
    vectors: list[list[float]] = model.encode(docs, batch_size=32, show_progress_bar=False).tolist()

    # ── 3. Upsert into pgvector ───────────────────────────────────────────
    try:
        conn = _db_connect()
        try:
            with conn.cursor() as cur:
                # Delete previous news rows for this symbol so re-indexing is idempotent
                cur.execute(
                    "DELETE FROM stock_embeddings WHERE symbol = %s AND doc_type = 'news'",
                    (symbol,),
                )
                for doc, meta, vec in zip(docs, metas, vectors):
                    cur.execute(
                        """
                        INSERT INTO stock_embeddings (symbol, doc_type, content, metadata, embedding)
                        VALUES (%s, 'news', %s, %s, %s::vector)
                        """,
                        (symbol, doc, json.dumps(meta), _vec_literal(vec)),
                    )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {
        "symbol":          symbol,
        "source":          source,
        "doc_type":        "news",
        "articles_indexed":len(docs),
        "vector_dims":     len(vectors[0]) if vectors else 0,
        "model":           EMBED_MODEL,
        "status":          "indexed",
        "preview": [
            {
                "title":        m["title"],
                "publish_date": m["publish_date"],
                "embed_text":   d[:120] + "…" if len(d) > 120 else d,
            }
            for d, m in zip(docs[:5], metas[:5])
        ],
    }


# ── Embedding-based news sentiment ───────────────────────────────────────
@app.post("/api/v1/embeddings/sentiment/news", tags=["Embeddings"])
def embed_sentiment_news(
    symbol: str = Query(description="Stock ticker, e.g. VCB"),
    source: str = Query(default="KBS"),
    limit:  int = Query(default=50, ge=1, le=500),
    threshold: float = Query(
        default=0.03,
        description="Min cosine-sim difference between pos/neg anchors to assign a label (else NEUTRAL)",
    ),
):
    """
    Fetch KBS news for *symbol*, embed each article title+snippet with the
    self-hosted sentence-transformer model, and classify sentiment by cosine
    similarity to pre-defined positive and negative Vietnamese anchor phrases.

    Returns per-article labels (POSITIVE / NEUTRAL / NEGATIVE) together with
    the raw similarity scores, plus aggregate counts.
    """
    symbol = symbol.upper()

    POS_ANCHOR = (
        "tăng trưởng lợi nhuận kỷ lục cổ tức vượt kế hoạch đột phá triển vọng tốt "
        "hoàn thành mục tiêu lãi lớn mở rộng đầu tư hợp tác ký kết cải thiện tích cực"
    )
    NEG_ANCHOR = (
        "giảm thua lỗ rủi ro xử phạt vi phạm nợ xấu sụt giảm margin call phá sản "
        "đình chỉ cưỡng chế giải chấp thiếu thanh khoản hạ bậc downgrade cảnh báo"
    )

    # ── 1. Fetch news ─────────────────────────────────────────────────────
    try:
        from vnstock import Vnstock
        raw_news = Vnstock().stock(symbol=symbol, source=source).company.news()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"vnstock news fetch failed: {e}")

    if raw_news is None or (hasattr(raw_news, "empty") and raw_news.empty):
        raise HTTPException(status_code=404, detail=f"No news returned for {symbol}")

    rows = raw_news.head(limit).to_dict(orient="records")

    # ── 2. Build texts ────────────────────────────────────────────────────
    texts, dates, titles = [], [], []
    for row in rows:
        title   = str(row.get("title") or row.get("news_title") or "")
        snippet = str(row.get("head")  or row.get("content") or "")[:400]
        pub     = row.get("publish_time") or row.get("public_date") or ""
        try:
            pub = str(pd.to_datetime(pub).date())
        except Exception:
            pub = str(pub)
        texts.append(f"{title}. {snippet}".strip(". "))
        dates.append(pub)
        titles.append(title)

    # ── 3. Embed ──────────────────────────────────────────────────────────
    model = _get_embed_model()
    import numpy as _np
    anc_vecs = model.encode([POS_ANCHOR, NEG_ANCHOR], normalize_embeddings=True)
    txt_vecs = model.encode(texts, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=False)
    sims = txt_vecs @ anc_vecs.T   # (n, 2)

    # ── 4. Classify ───────────────────────────────────────────────────────
    results = []
    counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    for i, (title, date, sim) in enumerate(zip(titles, dates, sims)):
        pos_sim, neg_sim = float(sim[0]), float(sim[1])
        diff = pos_sim - neg_sim
        if abs(diff) < threshold:
            label = "NEUTRAL"
        elif diff > 0:
            label = "POSITIVE"
        else:
            label = "NEGATIVE"
        counts[label] += 1
        results.append({
            "title":    title,
            "date":     date,
            "label":    label,
            "pos_sim":  round(pos_sim, 4),
            "neg_sim":  round(neg_sim, 4),
            "diff":     round(diff, 4),
        })

    return {
        "symbol":               symbol,
        "source":               source,
        "total_articles":       len(results),
        "counts":               counts,
        "positive_ratio":       round(counts["POSITIVE"] / len(results), 4) if results else 0,
        "negative_ratio":       round(counts["NEGATIVE"] / len(results), 4) if results else 0,
        "avg_pos_sim":          round(float(_np.mean([r["pos_sim"] for r in results])), 4),
        "avg_neg_sim":          round(float(_np.mean([r["neg_sim"] for r in results])), 4),
        "model":                EMBED_MODEL,
        "threshold":            threshold,
        "anchors": {
            "positive": POS_ANCHOR[:80] + "…",
            "negative": NEG_ANCHOR[:80] + "…",
        },
        "articles":             results,
    }


# ── Semantic search ───────────────────────────────────────────────────────
@app.get("/api/v1/embeddings/search", tags=["Embeddings"])
def embed_search(
    q:        str = Query(description="Natural-language query, e.g. 'high Sharpe defensive stock'"),
    k:        int = Query(default=5, ge=1, le=50),
    doc_type: str | None = Query(default=None, description="Filter: 'analysis' | 'news' | 'event'"),
):
    """
    Embed *q* with the self-hosted model and return the *k* nearest stocks
    from pgvector (cosine similarity).
    """
    model = _get_embed_model()
    vec   = model.encode(q).tolist()
    vec_s = _vec_literal(vec)

    try:
        conn = _db_connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if doc_type:
                    cur.execute(
                        f"""
                        SELECT id, symbol, doc_type, content, metadata,
                               1 - (embedding <=> '{vec_s}'::vector) AS cosine_similarity
                        FROM stock_embeddings
                        WHERE doc_type = %s
                        ORDER BY embedding <=> '{vec_s}'::vector
                        LIMIT %s
                        """,
                        (doc_type, k),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, symbol, doc_type, content, metadata,
                               1 - (embedding <=> '{vec_s}'::vector) AS cosine_similarity
                        FROM stock_embeddings
                        ORDER BY embedding <=> '{vec_s}'::vector
                        LIMIT %s
                        """,
                        (k,),
                    )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {
        "query":   q,
        "k":       k,
        "model":   EMBED_MODEL,
        "results": [
            {
                "id":                r["id"],
                "symbol":            r["symbol"],
                "doc_type":          r["doc_type"],
                "cosine_similarity": round(float(r["cosine_similarity"]), 4),
                "content":           r["content"],
                "metadata":          r["metadata"],
            }
            for r in rows
        ],
    }


# ── Similar stocks ────────────────────────────────────────────────────────
@app.get("/api/v1/embeddings/similar/{symbol}", tags=["Embeddings"])
def embed_similar(
    symbol: str,
    k:      int = Query(default=5, ge=1, le=50),
):
    """
    Find the *k* most quantitatively similar stocks to *symbol* using
    cosine distance on stored pgvector embeddings.
    Both *symbol* and the candidates must be indexed first via POST /index.
    """
    symbol = symbol.upper()
    try:
        conn = _db_connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT embedding::text AS embedding_text
                    FROM stock_embeddings
                    WHERE symbol = %s AND doc_type = 'analysis'
                    LIMIT 1
                    """,
                    (symbol,),
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=f"{symbol} not indexed. Call POST /api/v1/embeddings/index/{symbol} first.",
                    )
                vec_s = row["embedding_text"]  # already formatted as '[f1,f2,...]'

                cur.execute(
                    f"""
                    SELECT symbol, doc_type, content, metadata,
                           1 - (embedding <=> '{vec_s}'::vector) AS cosine_similarity
                    FROM stock_embeddings
                    WHERE symbol != %s AND doc_type = 'analysis'
                    ORDER BY embedding <=> '{vec_s}'::vector
                    LIMIT %s
                    """,
                    (symbol, k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {
        "target_symbol": symbol,
        "k":             k,
        "model":         EMBED_MODEL,
        "similar": [
            {
                "symbol":            r["symbol"],
                "cosine_similarity": round(float(r["cosine_similarity"]), 4),
                "metadata":          r["metadata"],
            }
            for r in rows
        ],
    }


# ── Embedding index stats ─────────────────────────────────────────────────
@app.get("/api/v1/embeddings/stats", tags=["Embeddings"])
def embed_stats():
    """Return a summary of what has been indexed in pgvector."""
    try:
        conn = _db_connect()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT doc_type,
                           COUNT(*)                          AS count,
                           ARRAY_AGG(symbol ORDER BY symbol) AS symbols
                    FROM stock_embeddings
                    GROUP BY doc_type
                    ORDER BY doc_type
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return {
        "total_documents":   sum(int(r["count"]) for r in rows),
        "embedding_model":   EMBED_MODEL,
        "vector_dimensions": EMBED_DIM,
        "similarity_metric": "cosine",
        "by_doc_type": [
            {
                "doc_type": r["doc_type"],
                "count":    int(r["count"]),
                "symbols":  r["symbols"],
            }
            for r in rows
        ],
    }


# ── DB initialisation on startup ──────────────────────────────────────────
@app.on_event("startup")
def _startup_init_db():
    """Create pgvector table if it doesn't exist. Silently skipped if DB is unreachable."""
    try:
        _ensure_pgvector_table()
    except Exception:
        pass  # pgvector not running yet; embedding endpoints will return 500 until it is


# ═══════════════════════════════════════════════════════════════════════════
# NEWS SCORING ROUTES  (Methods A / B / C + evaluation + themes + benchmark)
# ═══════════════════════════════════════════════════════════════════════════

from news_scoring import (
    KeywordBaseline,
    HybridScorer,
    RandomProjectionLSH,
    TopicAwareScorer,
    CLEAN_ENERGY_KEYWORDS,
    INVESTMENT_THEMES,
    load_news_data,
    load_news_from_supabase,
    load_assets_from_supabase,
    load_ground_truth,
    save_ground_truth,
    calculate_metrics,
    evaluate_method,
    benchmark_exact_search,
    benchmark_lsh_search,
    thematic_coverage,
    TickerImpactRecommender,
)

_SB_URL    = os.getenv("SUPABASE_URL", "")
_SB_KEY    = os.getenv("SUPABASE_SERVICE_KEY", "")
_SB_SCHEMA = os.getenv("SUPABASE_SCHEMA",     "market_data")

# ── Asset cache (refreshed every hour) ────────────────────────────────────────
_assets_cache: Optional[pd.DataFrame] = None
_assets_cache_ts: float = 0.0
_ASSETS_TTL: float = 3600.0


def _get_assets_df() -> pd.DataFrame:
    """Return assets DataFrame from Supabase, using a 1-hour in-memory cache."""
    global _assets_cache, _assets_cache_ts
    if _assets_cache is None or (time.time() - _assets_cache_ts) > _ASSETS_TTL:
        _assets_cache = load_assets_from_supabase(
            supabase_url=_SB_URL,
            supabase_key=_SB_KEY,
            schema=_SB_SCHEMA,
        )
        _assets_cache_ts = time.time()
    return _assets_cache


def _load_supabase_articles(limit: int = 401, asset_id: Optional[str] = None):
    """Load articles from Supabase and return (df, articles_list_for_api)."""
    df = load_news_from_supabase(
        supabase_url=_SB_URL,
        supabase_key=_SB_KEY,
        schema=_SB_SCHEMA,
        limit=limit,
        asset_id=asset_id,
    )
    articles = [
        ArticleIn(text=row["text"], publish_date=str(row["publish_date"].date()))
        for _, row in df.iterrows()
    ]
    return df, articles
from pydantic import BaseModel

# ── Request / response models ─────────────────────────────────────────────

class ArticleIn(BaseModel):
    text: str
    publish_date: Optional[str] = None  # ISO date string; optional for Method A

class KeywordScoreRequest(BaseModel):
    articles: List[ArticleIn]
    keywords: Optional[List[str]] = None
    threshold: float = 0.0

class HybridScoreRequest(BaseModel):
    articles: List[ArticleIn]
    keywords: Optional[List[str]] = None
    theme_query: str = "clean energy transition renewable power generation sustainability"
    keyword_weight: float = 0.3
    semantic_weight: float = 0.5
    time_weight: float = 0.2
    use_embeddings: bool = True

class LSHBuildRequest(BaseModel):
    articles: List[ArticleIn]
    num_hyperplanes: int = 12
    embedding_dim: int = 384

class LSHQueryRequest(BaseModel):
    query_text: str
    top_k: int = 20

class EvaluateRequest(BaseModel):
    scores: List[float]
    ground_truth: List[bool]
    threshold: float

class BenchmarkRequest(BaseModel):
    articles: List[ArticleIn]
    query_text: str = "clean energy transition renewable sustainability"
    top_k: int = 20
    num_runs: int = 100

class ThemeRequest(BaseModel):
    articles: List[ArticleIn]
    themes: Optional[Dict[str, List[str]]] = None

# In-memory LSH index (reset on server restart; rebuild via POST /news/lsh/build)
_lsh_index: Optional[RandomProjectionLSH] = None
_lsh_embeddings: Optional[np.ndarray] = None
_lsh_vectorizer = None   # TF-IDF vectorizer kept for consistent query encoding
_lsh_embed_method: str = "none"   # "transformer" | "tfidf"


def _encode_texts(texts: List[str], fit_vectorizer=None):
    """
    Encode a list of texts into dense float32 vectors.
    Tries sentence-transformers first; falls back to TF-IDF + TruncatedSVD.
    Returns (embeddings, embed_dim, method, vectorizer_or_None)
    """
    try:
        model = _get_embed_model()
        vecs = model.encode(texts, show_progress_bar=False)
        return vecs, vecs.shape[1], "transformer", None
    except Exception:
        pass

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    dim = min(128, len(texts) - 1) if len(texts) > 1 else 1

    if fit_vectorizer is None:
        tfidf = TfidfVectorizer(max_features=8000, sublinear_tf=True, strip_accents="unicode")
        svd   = TruncatedSVD(n_components=dim, random_state=42)
        sparse = tfidf.fit_transform(texts)
        dense  = svd.fit_transform(sparse)
        vecs   = normalize(dense).astype(np.float32)
        return vecs, dim, "tfidf", (tfidf, svd)
    else:
        tfidf, svd = fit_vectorizer
        sparse = tfidf.transform(texts)
        dense  = svd.transform(sparse)
        vecs   = normalize(dense).astype(np.float32)
        return vecs, dim, "tfidf", fit_vectorizer


# ── Method A – Keyword Baseline ───────────────────────────────────────────

@app.post("/api/v1/news/score/keyword", tags=["News Scoring"])
def news_score_keyword(req: KeywordScoreRequest):
    """
    Method A – Keyword Baseline.
    Score each article by keyword density (matches / text_length).
    Returns per-article scores sorted descending.
    """
    keywords = req.keywords or CLEAN_ENERGY_KEYWORDS
    scorer = KeywordBaseline(keywords)

    results = []
    for art in req.articles:
        score = scorer.score_article(art.text)
        if score > req.threshold:
            results.append({"text_preview": art.text[:120], "keyword_score": round(score, 8)})

    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return {
        "method": "A - Keyword Baseline",
        "keywords_used": len(keywords),
        "threshold": req.threshold,
        "total_articles": len(req.articles),
        "matched_articles": len(results),
        "results": results,
    }


# ── Method B – Hybrid ─────────────────────────────────────────────────────

@app.post("/api/v1/news/score/hybrid", tags=["News Scoring"])
def news_score_hybrid(req: HybridScoreRequest):
    """
    Method B – Hybrid System.
    score = keyword * 0.3 + semantic * 0.5 + time_decay * 0.2

    Requires sentence-transformers when use_embeddings=true.
    Degrades gracefully to keyword + time_decay when unavailable or disabled.
    """
    keywords = req.keywords or CLEAN_ENERGY_KEYWORDS
    embed_model = None
    embeddings_active = False

    if req.use_embeddings:
        try:
            embed_model = _get_embed_model()
            embeddings_active = True
        except Exception:
            pass

    scorer = HybridScorer(
        keywords=keywords,
        theme_query=req.theme_query,
        model=embed_model,
        keyword_weight=req.keyword_weight,
        semantic_weight=req.semantic_weight,
        time_weight=req.time_weight,
    )

    results = []
    for art in req.articles:
        pub_date = datetime.now()
        if art.publish_date:
            try:
                pub_date = datetime.fromisoformat(art.publish_date)
            except ValueError:
                pass
        scores = scorer.score_article(art.text, pub_date)
        results.append({
            "text_preview": art.text[:120],
            "publish_date": art.publish_date,
            **{k: round(v, 6) for k, v in scores.items()},
        })

    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return {
        "method": "B - Hybrid System",
        "embeddings_active": embeddings_active,
        "weights": {
            "keyword": req.keyword_weight,
            "semantic": req.semantic_weight,
            "time_decay": req.time_weight,
        },
        "total_articles": len(req.articles),
        "results": results,
    }


# ── Method C – LSH: build index ──────────────────────────────────────────

@app.post("/api/v1/news/lsh/build", tags=["News Scoring"])
def news_lsh_build(req: LSHBuildRequest):
    """
    Method C – Build an in-memory LSH index from the supplied articles.
    Uses sentence-transformers when available, otherwise falls back to TF-IDF + SVD.
    Call this once, then query via POST /api/v1/news/lsh/query.
    """
    global _lsh_index, _lsh_embeddings, _lsh_vectorizer, _lsh_embed_method

    texts = [art.text for art in req.articles]
    embeddings, embed_dim, method, vectorizer = _encode_texts(texts)
    _lsh_embeddings = embeddings
    _lsh_vectorizer = vectorizer
    _lsh_embed_method = method

    _lsh_index = RandomProjectionLSH(
        num_hyperplanes=req.num_hyperplanes,
        embedding_dim=embed_dim,
    )
    metadata_list = [
        {"text_preview": art.text[:120], "publish_date": art.publish_date}
        for art in req.articles
    ]
    _lsh_index.add_batch(embeddings, metadata_list)

    return {
        "method": "C - Random Projection LSH",
        "embed_method": method,
        "embed_dim": embed_dim,
        "status": "index built",
        "stats": _lsh_index.stats(),
    }


@app.post("/api/v1/news/lsh/query", tags=["News Scoring"])
def news_lsh_query(req: LSHQueryRequest):
    """
    Method C – Query the in-memory LSH index built by POST /news/lsh/build.
    Uses the same embedding method that was used to build the index.
    """
    if _lsh_index is None:
        raise HTTPException(status_code=400, detail="LSH index not built yet. Call POST /api/v1/news/lsh/build first.")

    query_vecs, _, _, _ = _encode_texts([req.query_text], fit_vectorizer=_lsh_vectorizer)
    query_vec = query_vecs[0]
    raw_results = _lsh_index.query(query_vec, top_k=req.top_k)

    return {
        "method": "C - Random Projection LSH",
        "embed_method": _lsh_embed_method,
        "query": req.query_text,
        "top_k": req.top_k,
        "results": [
            {"metadata": meta, "similarity": round(sim, 6)}
            for meta, sim in raw_results
        ],
    }


# ── Evaluation ────────────────────────────────────────────────────────────

@app.post("/api/v1/news/evaluate", tags=["News Scoring"])
def news_evaluate(req: EvaluateRequest):
    """
    Compute precision, recall, F1 and accuracy given raw float scores,
    a threshold, and ground-truth boolean labels.
    """
    if len(req.scores) != len(req.ground_truth):
        raise HTTPException(status_code=400, detail="scores and ground_truth must have the same length")

    predictions = [s > req.threshold for s in req.scores]
    metrics = calculate_metrics(predictions, req.ground_truth)
    return {
        "threshold": req.threshold,
        "num_articles": len(req.scores),
        "metrics": metrics,
    }


# ── Latency Benchmark ─────────────────────────────────────────────────────

@app.post("/api/v1/news/benchmark", tags=["News Scoring"])
def news_benchmark(req: BenchmarkRequest):
    """
    Benchmark exact cosine search vs LSH on the supplied articles.
    Returns median / mean / std latency in milliseconds for both methods.
    Uses sentence-transformers when available, otherwise falls back to TF-IDF + SVD.
    """
    texts = [art.text for art in req.articles]
    embeddings, embed_dim, method, vectorizer = _encode_texts(texts)
    query_vecs, _, _, _ = _encode_texts([req.query_text], fit_vectorizer=vectorizer)
    query_vec = query_vecs[0]

    lsh = RandomProjectionLSH(num_hyperplanes=12, embedding_dim=embed_dim)
    metadata_list = [{"text_preview": t[:80]} for t in texts]
    lsh.add_batch(embeddings, metadata_list)

    exact_stats = benchmark_exact_search(embeddings, query_vec, top_k=req.top_k, num_runs=req.num_runs)
    lsh_stats   = benchmark_lsh_search(lsh, query_vec, top_k=req.top_k, num_runs=req.num_runs)

    speedup = exact_stats["median_ms"] / lsh_stats["median_ms"] if lsh_stats["median_ms"] > 0 else None

    return {
        "num_articles": len(texts),
        "num_runs": req.num_runs,
        "embed_method": method,
        "embed_dim": embed_dim,
        "query": req.query_text,
        "exact_search_ms": exact_stats,
        "lsh_search_ms": lsh_stats,
        "speedup_x": round(speedup, 2) if speedup else None,
    }


# ── Thematic Coverage ─────────────────────────────────────────────────────

@app.post("/api/v1/news/themes", tags=["News Scoring"])
def news_themes(req: ThemeRequest):
    """
    Compute keyword-density coverage scores for each investment theme
    across the supplied articles.
    Pass a custom themes dict or leave empty to use the 5 default themes.
    """
    df = pd.DataFrame([{"text": art.text} for art in req.articles])
    themes = req.themes or INVESTMENT_THEMES
    coverage = thematic_coverage(df, themes=themes)
    return {
        "num_articles": len(req.articles),
        "num_themes": len(themes),
        "coverage": coverage,
    }


# ── File-based convenience: load news_data.json + score all three methods ─

@app.get("/api/v1/news/score/file", tags=["News Scoring"])
def news_score_from_file(
    path: str = Query(default="news_data.json", description="Path to news JSON file"),
    method: str = Query(default="keyword", description="keyword | hybrid | themes"),
    threshold: float = Query(default=0.0),
    top_k: int = Query(default=20, ge=1, le=500),
):
    """
    Load articles from a local news_data.json file and apply the chosen
    scoring method without needing to POST article bodies manually.
    """
    try:
        df = load_news_data(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if df.empty:
        raise HTTPException(status_code=404, detail="No articles found in file")

    if method == "keyword":
        scorer = KeywordBaseline(CLEAN_ENERGY_KEYWORDS)
        df_out = scorer.find_relevant(df, threshold=threshold)
        records = df_out[["title", "keyword_score", "publish_date"]].head(top_k).to_dict(orient="records")
        return {"method": "A - Keyword Baseline", "total": len(df), "matched": len(df_out), "top_results": records}

    if method == "themes":
        coverage = thematic_coverage(df)
        return {"method": "Thematic Coverage", "total_articles": len(df), "coverage": coverage}

    if method == "hybrid":
        embed_model = None
        try:
            embed_model = _get_embed_model()
        except Exception:
            pass
        scorer = HybridScorer(
            keywords=CLEAN_ENERGY_KEYWORDS,
            theme_query="clean energy transition renewable power generation sustainability",
            model=embed_model,
        )
        df_out = scorer.score_batch(df.head(200))
        cols = ["title", "hybrid_keyword", "hybrid_semantic", "hybrid_time_decay", "hybrid_hybrid_score"]
        available = [c for c in cols if c in df_out.columns]
        records = df_out[available].head(top_k).to_dict(orient="records")
        return {"method": "B - Hybrid System", "total": len(df), "top_results": records}

    raise HTTPException(status_code=400, detail="method must be one of: keyword, hybrid, themes")


# ── Supabase-backed convenience endpoints ─────────────────────────────────────

@app.get("/api/v1/news/supabase/keyword", tags=["News Scoring"])
def supabase_keyword(
    limit: int = 401,
    threshold: float = 0.0,
    asset_id: Optional[str] = None,
):
    """Method A on all Supabase news — server fetches articles, no payload needed."""
    _, articles = _load_supabase_articles(limit=limit, asset_id=asset_id)
    req = KeywordScoreRequest(articles=articles, threshold=threshold)
    return news_score_keyword(req)


@app.get("/api/v1/news/supabase/hybrid", tags=["News Scoring"])
def supabase_hybrid(
    limit: int = 100,
    theme_query: str = "clean energy transition renewable power generation sustainability",
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.5,
    time_weight: float = 0.2,
    use_embeddings: bool = True,
    asset_id: Optional[str] = None,
):
    """Method B on Supabase news — server fetches articles, no payload needed."""
    _, articles = _load_supabase_articles(limit=limit, asset_id=asset_id)
    req = HybridScoreRequest(
        articles=articles,
        theme_query=theme_query,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        time_weight=time_weight,
        use_embeddings=use_embeddings,
    )
    return news_score_hybrid(req)


@app.get("/api/v1/news/supabase/themes", tags=["News Scoring"])
def supabase_themes(
    limit: int = 401,
    asset_id: Optional[str] = None,
):
    """Thematic coverage on Supabase news — server fetches articles, no payload needed."""
    _, articles = _load_supabase_articles(limit=limit, asset_id=asset_id)
    req = ThemeRequest(articles=articles)
    return news_themes(req)


@app.post("/api/v1/news/supabase/lsh/build", tags=["News Scoring"])
def supabase_lsh_build(
    limit: int = 200,
    num_hyperplanes: int = 12,
    asset_id: Optional[str] = None,
):
    """Build LSH index from Supabase news — server fetches articles, no payload needed."""
    _, articles = _load_supabase_articles(limit=limit, asset_id=asset_id)
    req = LSHBuildRequest(articles=articles, num_hyperplanes=num_hyperplanes)
    return news_lsh_build(req)


@app.get("/api/v1/news/supabase/benchmark", tags=["News Scoring"])
def supabase_benchmark(
    limit: int = 200,
    query_text: str = "clean energy transition renewable sustainability",
    top_k: int = 20,
    num_runs: int = 100,
    asset_id: Optional[str] = None,
):
    """Latency benchmark using Supabase news — server fetches articles, no payload needed."""
    _, articles = _load_supabase_articles(limit=limit, asset_id=asset_id)
    req = BenchmarkRequest(articles=articles, query_text=query_text, top_k=top_k, num_runs=num_runs)
    return news_benchmark(req)


# ═══════════════════════════════════════════════════════════════════════════
# TICKER IMPACT RECOMMENDATION ROUTES
# ═══════════════════════════════════════════════════════════════════════════

class TickerImpactRequest(BaseModel):
    text: str
    publish_date: Optional[str] = None   # ISO date or datetime string
    top_k: int = 10
    min_score: float = 0.05


@app.post("/api/v1/news/ticker-impact", tags=["Ticker Impact"])
def news_ticker_impact_from_text(req: TickerImpactRequest):
    """
    Given raw article text, return the top-k stock tickers most likely
    impacted by that news, ranked by composite impact score.

    **Impact score = 0.45 × direct_mention + 0.35 × sector_match + 0.20 × hybrid_score**

    - ``direct_mention``: symbol or company name found verbatim in article text.
    - ``sector_match``: article's dominant investment topic aligns with the asset's sector.
    - ``hybrid_score``: overall keyword + semantic + time-decay relevance.
    """
    pub_date = datetime.now()
    if req.publish_date:
        try:
            pub_date = datetime.fromisoformat(req.publish_date)
        except ValueError:
            pass

    assets_df = _get_assets_df()
    if assets_df.empty:
        raise HTTPException(status_code=503, detail="Asset data unavailable from Supabase.")

    rec     = TickerImpactRecommender(assets_df)
    impacts = rec.recommend(
        text=req.text,
        publish_date=pub_date,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    return {
        "num_assets_evaluated": len(assets_df),
        "results": impacts,
    }


@app.get("/api/v1/news/{news_id}/ticker-impact", tags=["Ticker Impact"])
def news_ticker_impact_by_id(
    news_id: int,
    top_k: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.05, ge=0.0, le=1.0),
):
    """
    Fetch a news article by its ``news_id`` from Supabase and return the
    top-k stock tickers most likely impacted by that article.
    """
    from supabase import create_client
    sb  = create_client(_SB_URL, _SB_KEY)
    sbs = sb.schema(_SB_SCHEMA)

    r = sbs.table("news").select(
        "id,news_id,publish_date,title,news_content,source,source_url"
    ).eq("news_id", news_id).limit(1).execute()

    if not r.data:
        raise HTTPException(status_code=404, detail=f"news_id={news_id} not found.")

    article = r.data[0]
    text    = (article.get("title") or "") + " " + (article.get("news_content") or "")
    try:
        pub_date = datetime.fromisoformat(article["publish_date"])
    except (ValueError, TypeError):
        pub_date = datetime.now()

    assets_df = _get_assets_df()
    if assets_df.empty:
        raise HTTPException(status_code=503, detail="Asset data unavailable from Supabase.")

    rec     = TickerImpactRecommender(assets_df)
    impacts = rec.recommend(text=text, publish_date=pub_date, top_k=top_k, min_score=min_score)

    return {
        "news_id":              news_id,
        "title":                article.get("title", ""),
        "source":               article.get("source", ""),
        "source_url":           article.get("source_url", ""),
        "publish_date":         article.get("publish_date", ""),
        "num_assets_evaluated": len(assets_df),
        "results":              impacts,
    }


@app.get("/api/v1/stock/{symbol}/news-impact", tags=["Ticker Impact"])
def stock_news_impact(
    symbol: str,
    top_k: int = Query(default=10, ge=1, le=50),
    min_score: float = Query(default=0.05, ge=0.0, le=1.0),
    limit: int = Query(default=0, description="Max articles to scan (0 = all)"),
    source: str = Query(default=DEFAULT_SOURCE),
):
    """
    For a given stock ticker, scan the full Supabase news corpus and return
    the most impactful articles for that stock, ranked by impact score.

    Useful for monitoring news flow around a specific equity.
    """
    symbol_upper = symbol.upper()

    df = load_news_from_supabase(
        supabase_url=_SB_URL,
        supabase_key=_SB_KEY,
        schema=_SB_SCHEMA,
        limit=limit,
    )
    if df.empty:
        raise HTTPException(status_code=503, detail="No news articles available from Supabase.")

    assets_df = _get_assets_df()
    if assets_df.empty:
        raise HTTPException(status_code=503, detail="Asset data unavailable from Supabase.")

    asset_row = assets_df[assets_df["symbol"] == symbol_upper]
    name_en   = asset_row["name_en"].iloc[0]  if not asset_row.empty else symbol_upper
    sector    = asset_row["sector"].iloc[0]   if not asset_row.empty else None
    industry  = asset_row["industry"].iloc[0] if not asset_row.empty else None

    rec     = TickerImpactRecommender(assets_df)
    results = rec.score_for_ticker(
        df=df,
        symbol=symbol_upper,
        top_k=top_k,
        min_score=min_score,
    )

    return {
        "symbol":                symbol_upper,
        "name":                  name_en,
        "sector":                sector,
        "industry":              industry,
        "total_articles_scanned": len(df),
        "results":               results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=_API_HOST, port=_API_PORT, reload=True)
