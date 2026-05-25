from datetime import datetime, timedelta
from typing import Any, TypeAlias

from dags.etl_modules.adapters.market_data_repository import MarketDataRepository
from dags.etl_modules.adapters.price_provider_adapter import PriceProviderAdapter
from dags.etl_modules.contracts.providers import MarketDataWriter, PriceDataProvider
from dags.etl_modules.orchestrators.shared_reporting import report_failed_symbols
from dags.etl_modules.transformers import price_transformers
from dags.etl_modules.transformers.shared_transformers import (
    chunk_assets as chunk_assets_shared,
)

AssetRecord: TypeAlias = dict[str, str]
ChunkPayload: TypeAlias = dict[str, Any]
ChunkSummary: TypeAlias = dict[str, Any]
FinalizeSummary: TypeAlias = dict[str, int | bool]


def list_price_assets(
    *,
    provider: PriceDataProvider | None = None,
) -> list[AssetRecord]:
    provider = provider or PriceProviderAdapter()
    assets = provider.list_assets()
    print(f"Fetched {len(assets)} active VN stock tickers for prices pipeline.")
    return assets


def chunk_assets(
    assets: list[dict[str, Any]],
    *,
    chunk_size: int,
) -> list[dict[str, Any]]:
    chunks = chunk_assets_shared(assets, chunk_size)
    print(f"Prepared {len(chunks)} price chunks (chunk_size={chunk_size}).")
    return chunks


def process_price_chunk(
    chunk_payload: ChunkPayload,
    *,
    db_url: str | None,
    batch_size: int,
    lookback_days: int,
    load_window_days: int,
    upsert_sql: str,
    price_columns: tuple[str, ...],
    provider: PriceDataProvider | None = None,
    repository: MarketDataWriter | None = None,
    as_of: datetime | None = None,
) -> ChunkSummary:
    provider = provider or PriceProviderAdapter()
    repository = repository or MarketDataRepository()

    chunk_index = int(chunk_payload.get("chunk_index") or 0)
    assets = chunk_payload.get("assets") or []
    failed_symbols: list[dict[str, str]] = []
    price_frames = []

    as_of = as_of or datetime.today()
    lookback_date = (as_of - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    end_date = as_of.strftime("%Y-%m-%d")
    filter_from = (as_of - timedelta(days=load_window_days)).strftime("%Y-%m-%d")

    print(
        f"process_price_chunk[{chunk_index}] fetching prices from {lookback_date} "
        f"to {end_date} (filtering to last {load_window_days} days)"
    )

    for asset in assets:
        symbol = str(asset.get("symbol") or "").strip().upper()
        asset_id = str(asset.get("asset_id") or "").strip()
        if not symbol or not asset_id:
            failed_symbols.append(
                {
                    "symbol": symbol or "unknown",
                    "error": "missing symbol or asset_id in chunk payload",
                }
            )
            continue

        try:
            frame = provider.fetch_prices(symbol, asset_id, lookback_date, end_date)
        except Exception as exc:
            failed_symbols.append({"symbol": symbol, "error": str(exc)})
            continue

        if frame is not None and not frame.empty:
            price_frames.append(frame)

    records = price_transformers.records_from_frames(price_frames, filter_from)
    rows, failed_rows = price_transformers.convert_records_to_rows(
        records, price_columns
    )

    failed_batches: list[dict[str, object]] = []
    fatal_error: str | None = None
    if rows:
        print(
            f"Upserting {len(rows)} price rows for chunk {chunk_index} into "
            "market_data.prices..."
        )
        failed_batches, fatal_error = repository.upsert_rows(
            db_url=db_url,
            query=upsert_sql,
            rows=rows,
            table_name="market_data.prices",
            batch_size=batch_size,
        )

    report_failed_symbols(f"process_price_chunk[{chunk_index}]", failed_symbols)
    report_failed_symbols(
        f"process_price_chunk[{chunk_index}] (row conversion)",
        failed_rows,
    )

    failed_batch_rows = 0
    for item in failed_batches:
        size = item.get("size")
        if isinstance(size, (int, float, str)):
            failed_batch_rows += int(size)
    loaded_rows = max(len(rows) - failed_batch_rows, 0)

    summary: ChunkSummary = {
        "chunk_index": chunk_index,
        "chunk_assets": len(assets),
        "records_extracted": len(records),
        "rows_prepared": len(rows),
        "rows_loaded": loaded_rows,
        "failed_symbols": failed_symbols,
        "failed_rows": failed_rows,
        "failed_batches": failed_batches,
        "fatal_error": fatal_error,
    }
    print(
        "process_price_chunk summary: "
        f"chunk={chunk_index}, assets={summary['chunk_assets']}, "
        f"extracted={summary['records_extracted']}, "
        f"loaded={summary['rows_loaded']}, "
        f"failed_symbols={len(failed_symbols)}, "
        f"failed_rows={len(failed_rows)}, "
        f"failed_batches={len(failed_batches)}, "
        f"fatal_error={bool(fatal_error)}"
    )
    return summary


def finalize_prices_load(chunk_results: list[ChunkSummary] | None) -> FinalizeSummary:
    results = [result for result in (chunk_results or []) if isinstance(result, dict)]
    if not results:
        raise RuntimeError("No chunk results were produced for prices pipeline")

    total_assets = sum(int(result.get("chunk_assets") or 0) for result in results)
    total_extracted = sum(
        int(result.get("records_extracted") or 0) for result in results
    )
    total_loaded = sum(int(result.get("rows_loaded") or 0) for result in results)

    failed_symbols: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    failed_batches: list[dict[str, object]] = []
    fatal_errors: list[dict[str, str]] = []

    for result in results:
        failed_symbols.extend(result.get("failed_symbols") or [])
        failed_rows.extend(result.get("failed_rows") or [])
        failed_batches.extend(result.get("failed_batches") or [])
        fatal_error = result.get("fatal_error")
        if fatal_error:
            fatal_errors.append(
                {
                    "symbol": f"chunk-{result.get('chunk_index')}",
                    "error": str(fatal_error),
                }
            )

    print(
        "prices pipeline summary: "
        f"chunks={len(results)}, assets={total_assets}, extracted={total_extracted}, "
        f"loaded={total_loaded}, failed_symbols={len(failed_symbols)}, "
        f"failed_rows={len(failed_rows)}, failed_batches={len(failed_batches)}, "
        f"fatal_errors={len(fatal_errors)}"
    )

    report_failed_symbols("finalize_prices_load (symbol failures)", failed_symbols)
    report_failed_symbols("finalize_prices_load (row failures)", failed_rows)
    report_failed_symbols("finalize_prices_load (fatal errors)", fatal_errors)

    if fatal_errors:
        raise RuntimeError(
            "Prices pipeline completed with fatal chunk errors: "
            f"fatal_errors={len(fatal_errors)}"
        )

    alert_mode = bool(failed_symbols or failed_rows or failed_batches)
    if alert_mode:
        print(
            "prices pipeline alert mode: partial failures detected but DAG will "
            "complete successfully to avoid replaying already-loaded chunks. "
            f"failed_symbols={len(failed_symbols)}, "
            f"failed_rows={len(failed_rows)}, "
            f"failed_batches={len(failed_batches)}"
        )

    return {
        "chunks": len(results),
        "assets": total_assets,
        "records_extracted": total_extracted,
        "rows_loaded": total_loaded,
        "alert_mode": alert_mode,
        "failed_symbols": len(failed_symbols),
        "failed_rows": len(failed_rows),
        "failed_batches": len(failed_batches),
    }
