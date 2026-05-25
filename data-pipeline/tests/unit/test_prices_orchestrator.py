from datetime import datetime
from typing import get_type_hints

import pandas as pd
import pytest

from dags.etl_modules.orchestrators import prices_orchestrator


class _PriceProviderStub:
    def list_assets(self):
        return [{"symbol": "AAA", "asset_id": "1"}]

    def fetch_prices(self, symbol, asset_id, start_date, end_date):
        if symbol == "ERR":
            raise RuntimeError("fetch failed")
        if symbol == "EMPTY":
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "trading_date": "2025-01-02",
                    "open": 10,
                    "high": 12,
                    "low": 9,
                    "close": 11,
                    "volume": 100,
                    "asset_id": asset_id,
                    "source": "vci",
                },
                {
                    "trading_date": "2025-01-10",
                    "open": 12,
                    "high": 13,
                    "low": 11,
                    "close": 12.5,
                    "volume": 120,
                    "asset_id": asset_id,
                    "source": "vci",
                },
            ]
        )


class _RepositoryStub:
    def __init__(self, failed_batches=None, fatal_error=None):
        self.failed_batches = failed_batches or []
        self.fatal_error = fatal_error
        self.captured_rows = None

    def upsert_rows(self, *, db_url, query, rows, table_name, batch_size):
        self.captured_rows = rows
        return self.failed_batches, self.fatal_error


@pytest.mark.unit
def test_chunk_assets_splits_assets_by_batch_size():
    assets = [
        {"symbol": "AAA", "asset_id": "1"},
        {"symbol": "BBB", "asset_id": "2"},
        {"symbol": "CCC", "asset_id": "3"},
    ]

    chunks = prices_orchestrator.chunk_assets(assets, chunk_size=2)

    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 1
    assert len(chunks[0]["assets"]) == 2
    assert chunks[1]["chunk_index"] == 2
    assert len(chunks[1]["assets"]) == 1


@pytest.mark.unit
def test_process_price_chunk_filters_window_and_collects_partial_failures():
    provider = _PriceProviderStub()
    repository = _RepositoryStub(
        failed_batches=[{"batch_index": 1, "size": 1, "error": "db"}]
    )

    summary = prices_orchestrator.process_price_chunk(
        {
            "chunk_index": 3,
            "assets": [
                {"symbol": "AAA", "asset_id": "1"},
                {"symbol": "ERR", "asset_id": "2"},
                {"symbol": "", "asset_id": "3"},
            ],
        },
        provider=provider,
        repository=repository,
        db_url="postgres://example.local/test",
        batch_size=100,
        lookback_days=250,
        load_window_days=2,
        upsert_sql="INSERT INTO market_data.prices VALUES %s",
        price_columns=(
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "asset_id",
            "source",
        ),
        as_of=datetime(2025, 1, 10),
    )

    assert summary["chunk_index"] == 3
    assert summary["chunk_assets"] == 3
    assert summary["records_extracted"] == 1
    assert summary["rows_prepared"] == 1
    assert summary["rows_loaded"] == 0
    assert len(summary["failed_symbols"]) == 2
    assert summary["failed_batches"] == [{"batch_index": 1, "size": 1, "error": "db"}]
    assert summary["fatal_error"] is None
    assert repository.captured_rows is not None
    assert len(repository.captured_rows) == 1


@pytest.mark.unit
def test_finalize_prices_load_raises_only_on_fatal_errors():
    partial_result = prices_orchestrator.finalize_prices_load(
        [
            {
                "chunk_index": 1,
                "chunk_assets": 2,
                "records_extracted": 10,
                "rows_prepared": 10,
                "rows_loaded": 9,
                "failed_symbols": [{"symbol": "AAA", "error": "timeout"}],
                "failed_rows": [],
                "failed_batches": [{"batch_index": 1, "size": 1, "error": "db"}],
                "fatal_error": None,
            }
        ]
    )

    assert partial_result["alert_mode"] is True
    assert partial_result["failed_batches"] == 1

    with pytest.raises(RuntimeError):
        prices_orchestrator.finalize_prices_load(
            [
                {
                    "chunk_index": 2,
                    "chunk_assets": 1,
                    "records_extracted": 1,
                    "rows_prepared": 1,
                    "rows_loaded": 0,
                    "failed_symbols": [],
                    "failed_rows": [],
                    "failed_batches": [],
                    "fatal_error": "connection error",
                }
            ]
        )


@pytest.mark.unit
def test_public_prices_orchestrator_interfaces_are_typed():
    list_hints = get_type_hints(prices_orchestrator.list_price_assets)
    chunk_hints = get_type_hints(prices_orchestrator.chunk_assets)
    process_hints = get_type_hints(prices_orchestrator.process_price_chunk)
    finalize_hints = get_type_hints(prices_orchestrator.finalize_prices_load)

    assert "return" in list_hints
    assert {"assets", "chunk_size", "return"} <= set(chunk_hints.keys())
    assert {
        "chunk_payload",
        "db_url",
        "batch_size",
        "lookback_days",
        "load_window_days",
        "upsert_sql",
        "price_columns",
        "return",
    } <= set(process_hints.keys())
    assert {"chunk_results", "return"} <= set(finalize_hints.keys())
