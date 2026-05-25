"""
Asset Dimension ETL DAG
Story: prep-3-seed-asset-data
Architecture: Single Table Inheritance (STI)

Fetches:
- VN listed instruments (VCI provider)
    -> mapped to supported asset_class values, market=VN
- US stocks (Wikipedia S&P 500 + yfinance) → asset_class=STOCK, market=US
- Crypto (CoinGecko) → asset_class=CRYPTO, market=NULL
- Precious metals (yfinance futures mapping) → asset_class=COMMODITY, market=NULL

Schedule: Weekly (Sunday 2 AM)
"""

import logging
import os
import re
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import psycopg2
import psycopg2.extras
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from dags.etl_modules.vci_provider import (
    fetch_vn_industry_metadata,
    fetch_vn_listing_symbols,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-pipeline",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "assets_dimension_etl",
    default_args=default_args,
    description="Refresh asset master data (STI pattern: asset_class + market)",
    schedule="0 2 * * 0",  # Weekly Sunday 2 AM
    catchup=False,
    tags=["assets", "dimension", "etl", "sti"],
)


def _get_conn():
    db_url = os.getenv("SUPABASE_DB_URL")
    if not db_url:
        raise RuntimeError("SUPABASE_DB_URL environment variable is not set")
    return psycopg2.connect(db_url)


def upsert_assets_records(records: list[dict]) -> int:
    """Upsert asset records directly into market_data.assets using Supabase Postgres."""
    if not records:
        return 0

    def _map_row(rec: dict) -> tuple:
        market = rec.get("market")
        return (
            rec.get("symbol"),
            rec.get("name_en") or "",
            rec.get("name_local") or None,
            rec.get("asset_class"),
            market if market else None,
            rec.get("currency") or "",
            rec.get("exchange") or None,
            rec.get("sector") or None,
            rec.get("industry") or None,
            rec.get("industry_code") or None,
            rec.get("logo_url") or None,
            psycopg2.extras.Json(
                rec.get("external_api_metadata") or rec.get("metadata") or {}
            ),
            rec.get("source") or None,
        )

    rows_with_market = []
    rows_without_market = []
    for rec in records:
        mapped = _map_row(rec)
        if mapped[4] is None:
            rows_without_market.append(mapped)
        else:
            rows_with_market.append(mapped)

    with _get_conn() as conn:
        with conn.cursor() as cur:
            if rows_with_market:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO market_data.assets
                        (symbol, name_en, name_local, asset_class, market, currency,
                         exchange, sector, industry, industry_code,
                         logo_url, metadata, source)
                    VALUES %s
                    ON CONFLICT (symbol, market, asset_class)
                    WHERE market IS NOT NULL
                    DO UPDATE SET
                        name_en = EXCLUDED.name_en,
                        name_local = EXCLUDED.name_local,
                        currency = EXCLUDED.currency,
                        exchange = EXCLUDED.exchange,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        industry_code = EXCLUDED.industry_code,
                        logo_url = EXCLUDED.logo_url,
                        metadata = EXCLUDED.metadata,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    """,
                    rows_with_market,
                )

            if rows_without_market:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO market_data.assets
                        (symbol, name_en, name_local, asset_class, market, currency,
                         exchange, sector, industry, industry_code,
                         logo_url, metadata, source)
                    VALUES %s
                    ON CONFLICT (symbol, asset_class)
                    WHERE market IS NULL
                    DO UPDATE SET
                        name_en = EXCLUDED.name_en,
                        name_local = EXCLUDED.name_local,
                        currency = EXCLUDED.currency,
                        exchange = EXCLUDED.exchange,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        industry_code = EXCLUDED.industry_code,
                        logo_url = EXCLUDED.logo_url,
                        metadata = EXCLUDED.metadata,
                        source = EXCLUDED.source,
                        updated_at = NOW()
                    """,
                    rows_without_market,
                )

    return len(records)


def _map_vnstock_type_to_asset_class(
    vnstock_type: str | None,
    symbol: str | None = None,
    product_grp_id: str | None = None,
    organ_name: str | None = None,
) -> str:
    """
    Map vnstock listing metadata to supported assets.asset_class values.

    Uses provider `type` first, then falls back to symbol/name/product-group
    heuristics for rows where type is missing or inconsistent.
    """
    normalized = ""
    if vnstock_type is not None and not pd.isna(vnstock_type):
        normalized = str(vnstock_type).strip().upper()
    symbol_upper = str(symbol or "").strip().upper()
    product_group = str(product_grp_id or "").strip().upper()
    name_upper = str(organ_name or "").strip().upper()

    direct_map = {
        "STOCK": "STOCK",
        "ETF": "ETF",
        "FUND": "FUND",
        "UNIT_TRUST": "FUND",
        "BOND": "BOND",
        "DEBENTURE": "BOND",
        "INDEX": "INDEX",
        # Tradable contracts with leverage/expiry are derivatives.
        "FU": "DERIVATIVE",
        "FU_INDEX": "DERIVATIVE",
        "FU_BOND": "DERIVATIVE",
        "CW": "DERIVATIVE",
    }
    if normalized in direct_map:
        return direct_map[normalized]

    if any(token in normalized for token in ["FU", "WARRANT", "CW"]):
        return "DERIVATIVE"
    if "BOND" in normalized:
        return "BOND"
    if product_group in {"FIO", "FBX"}:
        return "DERIVATIVE"
    if product_group == "HCX":
        return "BOND"
    if "CHỨNG QUYỀN" in name_upper:
        return "DERIVATIVE"
    if re.match(r"^\d+[A-Z]\d+G\d+$", symbol_upper):
        return "DERIVATIVE"
    if re.match(r"^C[A-Z]{3,}\d{2,}$", symbol_upper):
        return "DERIVATIVE"

    logger.warning(
        "Unknown vnstock symbol type '%s'; defaulting asset_class to STOCK",
        normalized,
    )
    return "STOCK"


def deactivate_stale_vn_stock_rows(non_stock_symbols: list[str]) -> int:
    """
    Placeholder hook for stale STOCK reclassification.

    We intentionally avoid status mutation here because deployments may enforce
    strict status check constraints. Downstream readers now filter out
    non-equity symbols directly (metadata + symbol heuristics), which prevents
    bad rows from entering stock-only pipelines without violating constraints.
    """
    if non_stock_symbols:
        logger.info(
            "Detected %s VN symbols classified as non-STOCK; stock readers will "
            "exclude them via strict symbol-type filtering.",
            len(non_stock_symbols),
        )
    return 0


def fetch_vn_instruments(**context):
    """
    Fetch VN listed instruments from the VCI provider and map provider types
    to supported assets.asset_class values.
    """
    logger.info("Fetching VN instrument list from VCI provider...")

    df_list = fetch_vn_listing_symbols()
    df_list = df_list[df_list["exchange"].str.upper() != "DELISTED"]
    if "type" not in df_list.columns:
        logger.warning(
            "VCI provider returned no type column; "
            "defaulting all VN instruments to STOCK"
        )
        df_list["type"] = "STOCK"
    df_list["asset_class"] = df_list.apply(
        lambda row: _map_vnstock_type_to_asset_class(
            vnstock_type=row.get("type"),
            symbol=row.get("symbol"),
            product_grp_id=row.get("product_grp_id"),
            organ_name=row.get("organ_name"),
        ),
        axis=1,
    )

    logger.info(f"Found {len(df_list)} active VN instruments")

    # Get industry data
    try:
        df_ind = fetch_vn_industry_metadata()
        if "icb_code" not in df_ind.columns:
            code_candidates = [
                column
                for column in ["icb_code4", "icb_code3", "icb_code2", "icb_code1"]
                if column in df_ind.columns
            ]
            if code_candidates:
                df_ind = df_ind.copy()
                df_ind["icb_code"] = df_ind[code_candidates].bfill(axis=1).iloc[:, 0]
            else:
                df_ind = df_ind.copy()
                df_ind["icb_code"] = None

        industry_cols = [
            column
            for column in ["symbol", "icb_name2", "icb_name3", "icb_code"]
            if column in df_ind.columns
        ]
        for required in ["icb_name2", "icb_name3", "icb_code"]:
            if required not in industry_cols:
                df_ind[required] = None
                industry_cols.append(required)

        df = pd.merge(
            df_list[["symbol", "organ_name", "exchange", "type", "asset_class"]],
            df_ind[industry_cols],
            on="symbol",
            how="left",
        )
        logger.info("Successfully merged industry data")
    except Exception as e:
        logger.warning(f"Failed to fetch industries: {e}, using fallback")
        df = df_list[["symbol", "organ_name", "exchange", "type", "asset_class"]].copy()
        df["icb_name2"] = "Unknown"
        df["icb_name3"] = "Unknown"
        df["icb_code"] = None

    # Transform to STI schema
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "symbol": str(row["symbol"]),
                # vnstock only provides 'organ_name' (Vietnamese).
                # We use it for both name_en and name_local
                # to ensure UI has a display name.
                # TODO: (Improvement) find api to get correct name_en of vn stocks
                "name_en": str(row["organ_name"])
                if pd.notna(row["organ_name"])
                else "",
                "name_local": str(row["organ_name"])
                if pd.notna(row["organ_name"])
                else "",
                "asset_class": str(row["asset_class"])
                if pd.notna(row["asset_class"])
                else "STOCK",
                "market": "VN",  # Market code
                "currency": "VND",
                "exchange": str(row["exchange"])
                if pd.notna(row["exchange"])
                else "HOSE",
                "sector": str(row["icb_name2"])
                if pd.notna(row["icb_name2"])
                else "Unknown",
                "industry": str(row["icb_name3"])
                if pd.notna(row["icb_name3"])
                else "Unknown",
                "industry_code": str(row["icb_code"])
                if pd.notna(row["icb_code"])
                else None,
                "logo_url": "",
                "description": "",
                "external_api_metadata": {
                    "source_api": "vci",
                    "symbol_type": str(row["type"])
                    if pd.notna(row["type"])
                    else "STOCK",
                },
                "source": "vci",
                "is_active": 1,
            }
        )

    non_stock_symbols = sorted(
        {
            record["symbol"]
            for record in records
            if record.get("market") == "VN" and record.get("asset_class") != "STOCK"
        }
    )
    deactivated = deactivate_stale_vn_stock_rows(non_stock_symbols)
    if deactivated:
        logger.info(
            "Marked %s stale VN STOCK rows inactive due to reclassification.",
            deactivated,
        )

    upserted = upsert_assets_records(records)
    class_distribution = df["asset_class"].value_counts().sort_index().to_dict()
    logger.info(
        "✅ Upserted %s VN instruments with asset class distribution: %s",
        upserted,
        class_distribution,
    )
    return upserted


def fetch_us_stocks(**context):
    """
    Fetch US stocks using HYBRID APPROACH (Wikipedia S&P 500 + yfinance)
    → asset_class = 'STOCK', market = 'US'
    """
    import yfinance as yf

    logger.info("Fetching S&P 500 constituents from Wikipedia...")

    # Step 1: Get official S&P 500 constituents
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    sp500_df = pd.read_html(StringIO(response.text))[0]

    logger.info(f"Found {len(sp500_df)} S&P 500 constituents")

    # Step 2: Fetch metadata and market caps
    enriched_data = []
    for idx, (_, row) in enumerate(sp500_df.iterrows(), start=1):
        symbol = row["Symbol"]
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            enriched_data.append(
                {
                    "symbol": symbol,
                    "name_en": info.get("longName", row["Security"]),
                    "market_cap": info.get("marketCap", 0) or 0,
                    "sector": info.get("sector", row.get("GICS Sector", "Unknown")),
                    "industry": info.get(
                        "industry", row.get("GICS Sub-Industry", "Unknown")
                    ),
                    "exchange": info.get("exchange", "NASDAQ"),
                    "external_api_metadata": {
                        "yfinance_ticker": symbol,
                        "isin": info.get("isin", ""),
                        "cik": str(row.get("CIK", "")),
                    },
                }
            )

            if idx % 50 == 0:
                logger.info(f"Processed {idx}/{len(sp500_df)} stocks...")

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch {symbol}: {e}")

    logger.info(f"Successfully enriched {len(enriched_data)} stocks")

    # Step 3: Use all S&P 500 stocks
    df = pd.DataFrame(enriched_data)

    # Step 4: Format for STI schema
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "symbol": str(row["symbol"]),
                "name_en": str(row["name_en"]) if pd.notna(row["name_en"]) else "",
                "name_local": "",  # US stocks don't have local names
                "asset_class": "STOCK",  # STI discriminator
                "market": "US",  # Market code
                "currency": "USD",
                "exchange": str(row["exchange"])
                if pd.notna(row["exchange"])
                else "NASDAQ",
                "sector": str(row["sector"]) if pd.notna(row["sector"]) else "Unknown",
                "industry": str(row["industry"])
                if pd.notna(row["industry"])
                else "Unknown",
                "logo_url": "",
                "description": "",
                "external_api_metadata": row["external_api_metadata"],
                "source": "yfinance",
                "is_active": 1,
            }
        )

    upserted = upsert_assets_records(records)
    logger.info(f"✅ Upserted {upserted} US stocks (asset_class=STOCK, market=US)")
    return upserted


def fetch_crypto(**context):
    """
    Fetch top 500 crypto from CoinGecko
    → asset_class = 'CRYPTO', market = '' (empty/NULL)
    NOTE: 'coingecko_id' in metadata is CRITICAL for MarketDataService price fetching.
    """
    import time

    from pycoingecko import CoinGeckoAPI

    cg = CoinGeckoAPI()

    logger.info("Fetching top 500 crypto from CoinGecko...")

    all_coins = []
    # Fetch top 500 (2 pages of 250)
    for page in range(1, 3):
        try:
            logger.info(f"Fetching page {page}...")
            coins = cg.get_coins_markets(
                vs_currency="usd",
                order="market_cap_desc",
                per_page=250,
                page=page,
                sparkline=False,
            )
            all_coins.extend(coins)
            time.sleep(1)  # Rate limit protection
        except Exception as e:
            logger.error(f"Failed to fetch page {page}: {e}")

    logger.info(f"Fetched {len(all_coins)} crypto assets")

    # Known stablecoins
    stablecoins = {
        "USDT",
        "USDC",
        "DAI",
        "BUSD",
        "TUSD",
        "USDD",
        "FRAX",
        "USDP",
        "PYUSD",
    }

    records = []
    # Track symbols to prevent duplicates if any
    seen_symbols = set()

    for coin in all_coins:
        symbol_upper = coin["symbol"].upper()

        if symbol_upper in seen_symbols:
            continue
        seen_symbols.add(symbol_upper)

        is_stable = "1" if symbol_upper in stablecoins else "0"

        records.append(
            {
                "symbol": symbol_upper,
                "name_en": coin["name"],
                "name_local": "",
                "asset_class": "CRYPTO",  # STI discriminator
                "market": "",  # No market for crypto
                "currency": "USD",
                "exchange": "",  # Crypto trades on multiple exchanges
                "sector": "Cryptocurrency",
                "industry": "",
                "logo_url": coin.get("image", ""),
                "description": "",
                "external_api_metadata": {
                    "coingecko_id": coin["id"],
                    "chain": "",  # Would require additional API call
                    "is_stablecoin": is_stable,
                    "market_cap_rank": str(coin.get("market_cap_rank", 0)),
                },
                "source": "coingecko",
                "is_active": 1,
            }
        )

    upserted = upsert_assets_records(records)
    logger.info(f"✅ Upserted {upserted} crypto (asset_class=CRYPTO)")
    return upserted


def fetch_precious_metals(**context):
    """
    Seed precious metals dimensions for AI Risk Intelligence.

    Mapping:
    - XAU -> yfinance source symbol GC=F (Gold futures)
    - XAG -> yfinance source symbol SI=F (Silver futures)

    Stored as asset_class='COMMODITY' so downstream systems can classify
    metals independently from stocks/crypto.
    """
    import yfinance as yf

    metals_mapping = [
        {
            "symbol": "XAU",
            "name_en": "Gold (XAU/USD)",
            "name_local": "Gold",
            "source_symbol": "GC=F",
            "display_exchange": "COMEX",
        },
        {
            "symbol": "XAG",
            "name_en": "Silver (XAG/USD)",
            "name_local": "Silver",
            "source_symbol": "SI=F",
            "display_exchange": "COMEX",
        },
    ]

    records = []
    for metal in metals_mapping:
        exchange = metal["display_exchange"]
        provider_name = metal["name_en"]

        # Best-effort metadata enrichment from yfinance.
        try:
            info = yf.Ticker(metal["source_symbol"]).info or {}
            provider_name = str(info.get("longName") or provider_name)
            exchange = str(info.get("exchange") or exchange)
        except Exception as e:
            logger.warning(
                "Could not enrich "
                f"{metal['symbol']} from yfinance "
                f"{metal['source_symbol']}: {e}"
            )

        records.append(
            {
                "symbol": metal["symbol"],
                # Keep explicit XAU/XAG in names so symbol and keyword search both work.
                "name_en": metal["name_en"],
                "name_local": metal["name_local"],
                "asset_class": "COMMODITY",
                "market": "",
                "currency": "USD",
                "exchange": exchange,
                "sector": "Precious Metals",
                "industry": "Commodity",
                "logo_url": "",
                "description": provider_name,
                "external_api_metadata": {
                    "source_api": "yfinance",
                    "source_symbol": metal["source_symbol"],
                    "provider_symbol": metal["source_symbol"],
                    "commodity_type": "precious_metal",
                    "unit": "troy_ounce",
                    "quote_pair": f"{metal['symbol']}/USD",
                },
                "source": "yfinance",
                "is_active": 1,
            }
        )

    upserted = upsert_assets_records(records)
    logger.info(f"Upserted {upserted} precious metals (asset_class=COMMODITY)")
    return upserted


# Define tasks
task_vn = PythonOperator(
    task_id="fetch_vn_instruments",
    python_callable=fetch_vn_instruments,
    dag=dag,
)

task_us = PythonOperator(
    task_id="fetch_us_stocks",
    python_callable=fetch_us_stocks,
    dag=dag,
)

task_crypto = PythonOperator(
    task_id="fetch_crypto",
    python_callable=fetch_crypto,
    dag=dag,
)

task_precious_metals = PythonOperator(
    task_id="fetch_precious_metals",
    python_callable=fetch_precious_metals,
    dag=dag,
)

# All tasks run in parallel
_ = [task_vn, task_us, task_crypto, task_precious_metals]
