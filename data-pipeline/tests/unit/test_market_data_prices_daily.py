import pytest

from dags import market_data_prices_daily


@pytest.mark.unit
def test_prices_daily_dag_has_expected_identity_schedule_and_tasks():
    assert market_data_prices_daily.dag.dag_id == "market_data_prices_daily"
    assert market_data_prices_daily.dag.schedule == "0 18 * * 1-5"
    assert sorted(task.task_id for task in market_data_prices_daily.dag.tasks) == [
        "chunk_price_assets",
        "finalize_prices_load",
        "list_price_assets",
        "process_price_chunk",
    ]


@pytest.mark.unit
def test_chunk_price_assets_delegates_to_orchestrator(monkeypatch):
    assets = [{"symbol": "AAA", "asset_id": "1"}]
    captured = {}

    def _fake_chunk_assets(value, *, chunk_size):
        captured["value"] = value
        captured["chunk_size"] = chunk_size
        return [{"chunk_index": 1, "assets": value}]

    monkeypatch.setattr(
        market_data_prices_daily.prices_orchestrator,
        "chunk_assets",
        _fake_chunk_assets,
    )

    result = market_data_prices_daily.chunk_price_assets.function(assets)

    assert result == [{"chunk_index": 1, "assets": assets}]
    assert captured["value"] == assets
    assert captured["chunk_size"] == market_data_prices_daily.DB_UPSERT_BATCH_SIZE


@pytest.mark.unit
def test_process_price_chunk_delegates_to_orchestrator(monkeypatch):
    payload = {"chunk_index": 1, "assets": [{"symbol": "AAA", "asset_id": "1"}]}

    expected_summary = {
        "chunk_index": 1,
        "chunk_assets": 1,
        "records_extracted": 0,
        "rows_prepared": 0,
        "rows_loaded": 0,
        "failed_symbols": [],
        "failed_rows": [],
        "failed_batches": [],
        "fatal_error": None,
    }

    def _fake_process(chunk_payload, **kwargs):
        assert chunk_payload == payload
        assert kwargs["db_url"] == market_data_prices_daily.SUPABASE_DB_URL
        assert kwargs["batch_size"] == market_data_prices_daily.DB_UPSERT_BATCH_SIZE
        assert (
            kwargs["lookback_days"]
            == market_data_prices_daily.PRICE_INDICATOR_LOOKBACK_DAYS
        )
        assert (
            kwargs["load_window_days"]
            == market_data_prices_daily.PRICE_LOAD_WINDOW_DAYS
        )
        assert kwargs["upsert_sql"] == market_data_prices_daily.PRICES_UPSERT_SQL
        assert kwargs["price_columns"] == market_data_prices_daily.PRICE_COLUMNS
        return expected_summary

    monkeypatch.setattr(
        market_data_prices_daily.prices_orchestrator,
        "process_price_chunk",
        _fake_process,
    )

    result = market_data_prices_daily.process_price_chunk.function(payload)

    assert result == expected_summary


@pytest.mark.unit
def test_finalize_prices_load_delegates_to_orchestrator(monkeypatch):
    chunk_results = [{"chunk_index": 1, "rows_loaded": 1}]
    expected = {
        "chunks": 1,
        "assets": 1,
        "records_extracted": 1,
        "rows_loaded": 1,
        "alert_mode": False,
        "failed_symbols": 0,
        "failed_rows": 0,
        "failed_batches": 0,
    }

    def _fake_finalize(results):
        assert results == chunk_results
        return expected

    monkeypatch.setattr(
        market_data_prices_daily.prices_orchestrator,
        "finalize_prices_load",
        _fake_finalize,
    )

    assert (
        market_data_prices_daily.finalize_prices_load.function(chunk_results)
        == expected
    )
