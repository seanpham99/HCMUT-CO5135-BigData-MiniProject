from datetime import datetime, timedelta

from airflow import DAG
from airflow.sdk import task
from pendulum import timezone

from dags.etl_modules.notifications import (
    send_failure_notification,
    send_success_notification,
)
from dags.etl_modules.orchestrators import prices_orchestrator
from dags.etl_modules.settings import get_env, get_env_int

# CONFIG
SUPABASE_DB_URL = get_env("SUPABASE_DB_URL")
DB_UPSERT_BATCH_SIZE = get_env_int("DB_UPSERT_BATCH_SIZE", 100)
VCI_GRAPHQL_POOL = "vci_graphql"
PRICE_INDICATOR_LOOKBACK_DAYS = get_env_int("PRICE_INDICATOR_LOOKBACK_DAYS", 250)
PRICE_LOAD_WINDOW_DAYS = get_env_int("PRICE_LOAD_WINDOW_DAYS", 7)
PRICE_COLUMNS = (
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "asset_id",
    "source",
)
PRICES_UPSERT_SQL = """
INSERT INTO market_data.prices
    (trading_date, open, high, low, close, volume, asset_id, source)
VALUES %s
ON CONFLICT (asset_id, trading_date) DO UPDATE SET
    open          = EXCLUDED.open,
    high          = EXCLUDED.high,
    low           = EXCLUDED.low,
    close         = EXCLUDED.close,
    volume        = EXCLUDED.volume,
    source        = EXCLUDED.source,
    ingested_at   = NOW()
"""

default_args = {
    "owner": "data_engineer",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

local_tz = timezone("Asia/Bangkok")


with DAG(
    dag_id="market_data_prices_daily",
    default_args=default_args,
    schedule="0 18 * * 1-5",  # 6 PM Vietnam Time Mon-Fri
    start_date=datetime(2024, 1, 1, tzinfo=local_tz),
    catchup=False,
    tags=["stock-price", "supabase", "evening-batch"],
    on_success_callback=send_success_notification,
    on_failure_callback=send_failure_notification,
) as dag:

    @task(show_return_value_in_logs=False)
    def list_price_assets():
        return prices_orchestrator.list_price_assets()

    @task(show_return_value_in_logs=False)
    def chunk_price_assets(assets):
        return prices_orchestrator.chunk_assets(
            assets,
            chunk_size=DB_UPSERT_BATCH_SIZE,
        )

    @task
    def process_price_chunk(chunk_payload):
        return prices_orchestrator.process_price_chunk(
            chunk_payload,
            db_url=SUPABASE_DB_URL,
            batch_size=DB_UPSERT_BATCH_SIZE,
            lookback_days=PRICE_INDICATOR_LOOKBACK_DAYS,
            load_window_days=PRICE_LOAD_WINDOW_DAYS,
            upsert_sql=PRICES_UPSERT_SQL,
            price_columns=PRICE_COLUMNS,
        )

    @task(trigger_rule="all_done")
    def finalize_prices_load(chunk_results):
        return prices_orchestrator.finalize_prices_load(chunk_results)

    price_assets = list_price_assets()
    price_chunks = chunk_price_assets(price_assets)
    chunk_summaries = process_price_chunk.override(pool=VCI_GRAPHQL_POOL).expand(
        chunk_payload=price_chunks
    )
    final_summary = finalize_prices_load(chunk_summaries)

    _ = price_assets >> price_chunks >> chunk_summaries >> final_summary
