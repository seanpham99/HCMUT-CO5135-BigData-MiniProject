import logging
from datetime import datetime

import psycopg2
import requests
from dotenv import load_dotenv

from dags.etl_modules.adapters.notification_adapter import (
    CallableNewsSummaryProvider,
    TelegramNotificationAdapter,
)
from dags.etl_modules.orchestrators import notifications_orchestrator
from dags.etl_modules.settings import get_env

load_dotenv()


def get_latest_stock_data(tickers):
    """
    Fetches latest stock data and nearest fundamental snapshot from Supabase
    Postgres tables for the given tickers.

    Reads from normalized Supabase tables:
    - prices: latest daily close/volume by asset_id
    - financial_ratios: latest fiscal snapshot as-of trading_date by asset_id
    - assets: symbol/sector/industry metadata
    Returns a dictionary with ticker as key and fundamental data as value.
    """
    if not tickers:
        return {}

    db_url = get_env("SUPABASE_DB_URL")
    if not db_url:
        logging.error("SUPABASE_DB_URL is not set; cannot fetch stock context")
        return {}

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH target_assets AS (
                        SELECT
                            a.id AS asset_id,
                            a.symbol,
                            a.sector,
                            a.industry
                        FROM market_data.assets a
                        WHERE a.symbol = ANY(%s)
                          AND a.asset_class = 'STOCK'
                          AND a.market = 'VN'
                    ),
                    latest_price AS (
                        SELECT DISTINCT ON (p.asset_id)
                            p.asset_id,
                            p.trading_date,
                            p.close,
                            p.volume
                        FROM market_data.prices p
                        JOIN target_assets ta
                          ON ta.asset_id = p.asset_id
                        ORDER BY p.asset_id, p.trading_date DESC
                    )
                    SELECT
                        ta.symbol AS ticker,
                        lp.trading_date::text,
                        lp.close,
                        lp.volume,
                        CASE
                            WHEN p20.close IS NULL OR p20.close = 0 THEN 0
                            ELSE ((lp.close - p20.close) / p20.close) * 100
                        END AS return_1m,
                        COALESCE(ta.sector, 'N/A') AS sector,
                        COALESCE(ta.industry, 'N/A') AS industry,
                        COALESCE(
                            r.pe_ratio,
                            CASE
                                WHEN COALESCE(r.eps, 0) = 0 THEN 0
                                ELSE (lp.close * 1000) / r.eps
                            END,
                            0
                        ) AS pe_ratio,
                        COALESCE(r.roe, 0) AS roe,
                        COALESCE(r.roic, 0) AS roic,
                        COALESCE(r.debt_to_equity, 0) AS debt_to_equity,
                        COALESCE(r.net_profit_margin, 0) AS net_profit_margin,
                        COALESCE(r.eps, 0) AS eps
                    FROM latest_price lp
                    JOIN target_assets ta
                      ON ta.asset_id = lp.asset_id
                    LEFT JOIN LATERAL (
                        SELECT p2.close
                        FROM market_data.prices p2
                        WHERE p2.asset_id = lp.asset_id
                          AND p2.trading_date < lp.trading_date
                        ORDER BY p2.trading_date DESC
                        OFFSET 19 LIMIT 1
                    ) p20 ON TRUE
                    LEFT JOIN LATERAL (
                        SELECT
                            fr.pe_ratio,
                            fr.roe,
                            fr.roic,
                            fr.debt_to_equity,
                            fr.net_profit_margin,
                            fr.eps
                        FROM market_data.financial_ratios fr
                        WHERE fr.asset_id = lp.asset_id
                          AND fr.fiscal_date <= lp.trading_date
                        ORDER BY fr.fiscal_date DESC
                        LIMIT 1
                    ) r ON TRUE
                    """,
                    (tickers,),
                )
                rows = cur.fetchall()

        stock_data = {}
        for row in rows:
            stock_data[row[0]] = {
                "trading_date": str(row[1]),
                "close": float(row[2]) if row[2] is not None else 0.0,
                "volume": int(row[3]) if row[3] is not None else 0,
                "return_1m": float(row[4]) if row[4] is not None else 0.0,
                "sector": str(row[5]) if row[5] else "N/A",
                "industry": str(row[6]) if row[6] else "N/A",
                "pe_ratio": float(row[7]) if row[7] is not None else 0.0,
                "roe": float(row[8]) if row[8] is not None else 0.0,
                "roic": float(row[9]) if row[9] is not None else 0.0,
                "debt_to_equity": float(row[10]) if row[10] is not None else 0.0,
                "net_profit_margin": float(row[11]) if row[11] is not None else 0.0,
                "eps": float(row[12]) if row[12] is not None else 0.0,
            }

        return stock_data

    except Exception as e:
        logging.error(f"Failed to fetch stock data from database: {e}")
        return {}


def summarize_news_with_gemini(news_data):
    """
    Use Gemini AI to generate a summary of news with technical context.
    """
    gemini_api_key = get_env("GEMINI_API_KEY")

    if not gemini_api_key:
        logging.warning("Gemini API key not found. Using basic summary.")
        return None

    # Prepare news data for Gemini
    news_by_ticker = {}
    for item in news_data:
        ticker = item.get("ticker")
        if ticker not in news_by_ticker:
            news_by_ticker[ticker] = []
        news_by_ticker[ticker].append(
            {
                "title": item.get("title", ""),
                "price_change": item.get("price_change_ratio", 0),
                "source": item.get("source", ""),
            }
        )

    # Fetch latest stock data from database
    tickers = list(news_by_ticker.keys())
    stock_data = get_latest_stock_data(tickers)

    # Create enhanced prompt for Gemini with technical data
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = (
        "You are a financial analyst summarizing Vietnamese stock market news for "
        f"{current_date}.\n\n"
        "Here is today's news data with current technical and fundamental "
        "indicators:\n\n"
    )

    for ticker, articles in sorted(news_by_ticker.items()):
        prompt += f"\n{ticker}:"

        # Add technical data if available
        if ticker in stock_data:
            data = stock_data[ticker]
            prompt += f"""
  Current Price: {data["close"]:.2f} VND (1000s)
  1-Month Return: {data["return_1m"]:.2f}%
  P/E Ratio: {data["pe_ratio"]:.2f}
  ROE: {data["roe"]:.2f}%, ROIC: {data["roic"]:.2f}%
  Debt/Equity: {data["debt_to_equity"]:.2f}
  Sector: {data["sector"]}, Industry: {data["industry"]}
"""

        prompt += "\n  News Articles:\n"
        for i, article in enumerate(articles[:5], 1):
            prompt += (
                f"  {i}. {article['title']} "
                f"(Price change at publish: {article['price_change']:.2f}%)\n"
            )

        if len(articles) > 5:
            prompt += f"  ...and {len(articles) - 5} more articles\n"

    prompt += """

Please provide:
1. A brief overall market sentiment (2-3 sentences)
2. Key highlights for each stock with technical analysis context:
    - Interpret the news in light of current technical indicators
      (RSI, MACD, moving averages)
   - Mention if the stock is overbought/oversold, trending up/down
   - Note any divergences between news sentiment and technical signals
3. Any notable trends or patterns you observe
4. Brief investment considerations based on the combination of news and technicals

Keep the summary concise, professional, and actionable for investors.
Format using simple text, no markdown."""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite-preview:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": gemini_api_key}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 800},
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extract the generated text
        if "candidates" in result and len(result["candidates"]) > 0:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
            logging.info("Gemini summary generated successfully")
            return text
        else:
            logging.warning("No summary generated by Gemini")
            return None

    except Exception as e:
        logging.error(f"Failed to generate Gemini summary: {e}")
        return None


def send_telegram_news_summary(news_data):
    """
    Sends a formatted news summary to Telegram with Gemini AI analysis.
    """
    sender = TelegramNotificationAdapter()
    summarizer = CallableNewsSummaryProvider(summarize_fn=summarize_news_with_gemini)
    notifications_orchestrator.send_news_summary_notification(
        news_data,
        sender=sender,
        summarizer=summarizer,
    )


def send_telegram_message(context, status):
    """
    Sends a Telegram notification based on the DAG run status.
    """
    sender = TelegramNotificationAdapter()
    notifications_orchestrator.send_dag_status_notification(
        context,
        status,
        sender=sender,
    )


def send_success_notification(context):
    send_telegram_message(context, status="SUCCESS")


def send_failure_notification(context):
    send_telegram_message(context, status="FAILED")
