import os
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras
from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import task
from pendulum import timezone

from dags.etl_modules.fetcher import fetch_stock_price
from dags.etl_modules.notifications import (
    send_failure_notification,
    send_success_notification,
)
from dags.etl_modules.refresh_historical_trigger import (
    ensure_corporate_events_table_exists,
    fetch_tickers_for_refresh,
)

# CONFIG
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
CORPORATE_EVENTS_LOOKBACK_DAYS = int(os.getenv("CORPORATE_EVENTS_LOOKBACK_DAYS", "14"))

default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": send_failure_notification,
}

with DAG(
    "refresh_historical_prices",
    default_args=default_args,
    description=(
        "Refreshes full historical prices (OHLCV) when new corporate "
        "events are detected"
    ),
    schedule="30 11 * * *",  # 18:30 VN time (UTC+7)
    start_date=datetime(2025, 1, 1, tzinfo=timezone("UTC")),
    catchup=False,
    tags=["market_data", "maintenance"],
    on_success_callback=send_success_notification,
) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end")

    @task
    def get_unprocessed_tickers():
        print("Connecting to Supabase to find assets that need historical refresh...")
        conn = psycopg2.connect(SUPABASE_DB_URL)
        assets = []
        try:
            with conn:
                with conn.cursor() as cur:
                    ensure_corporate_events_table_exists(cur)
                    print(
                        "Using trigger source: market_data.corporate_events "
                        f"(events lookback={CORPORATE_EVENTS_LOOKBACK_DAYS} days)"
                    )
                    assets = fetch_tickers_for_refresh(
                        cur,
                        events_lookback_days=CORPORATE_EVENTS_LOOKBACK_DAYS,
                    )
        finally:
            conn.close()

        print(f"Found {len(assets)} assets needing historical refresh.")
        return assets

    @task
    def refresh_ticker_history(assets: list):
        if not assets:
            print("No assets to process. Exiting.")
            return

        end_date = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")
        symbols = [asset.get("symbol") for asset in assets if asset.get("symbol")]
        print(
            "Fetching 6-year history "
            f"({start_date} to {end_date}) for symbols: {symbols}"
        )

        conn = psycopg2.connect(SUPABASE_DB_URL)
        try:
            with conn.cursor() as source_cur:
                ensure_corporate_events_table_exists(source_cur)

            for asset in assets:
                symbol = asset.get("symbol")
                asset_id = asset.get("asset_id")
                if not symbol or not asset_id:
                    print(f"Warning: Invalid asset payload skipped: {asset}")
                    continue
                print(f"Processing {symbol} ({asset_id})...")
                df_price = fetch_stock_price(symbol, asset_id, start_date, end_date)

                if df_price.empty:
                    print(f"Warning: No data fetched for {symbol}. Skipping.")
                    continue

                # Prepare for bulk upsert
                price_cols = [
                    "trading_date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "asset_id",
                    "source",
                ]

                rows = []
                for _, row in df_price.iterrows():
                    # Handle Pandas Timestamp conversion
                    if isinstance(row.get("trading_date"), pd.Timestamp):
                        row["trading_date"] = row["trading_date"].date()
                    rows.append(tuple(row.get(col) for col in price_cols))

                print(f"Upserting {len(rows)} rows for {symbol}...")
                with conn:
                    with conn.cursor() as cur:
                        psycopg2.extras.execute_values(
                            cur,
                            """
                            INSERT INTO market_data.prices
                                (
                                    trading_date, open, high, low, close,
                                    volume, asset_id, source
                                )
                            VALUES %s
                            ON CONFLICT (asset_id, trading_date) DO UPDATE SET
                                open          = EXCLUDED.open,
                                high          = EXCLUDED.high,
                                low           = EXCLUDED.low,
                                close         = EXCLUDED.close,
                                volume        = EXCLUDED.volume,
                                source        = EXCLUDED.source,
                                ingested_at   = NOW()
                            """,
                            rows,
                        )

        finally:
            conn.close()

    # Define execution graph
    tickers_to_process = get_unprocessed_tickers()
    refresh_task = refresh_ticker_history(tickers_to_process)  # type: ignore[arg-type]

    _ = start >> tickers_to_process >> refresh_task >> end
