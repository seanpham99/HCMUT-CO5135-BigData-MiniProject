from unittest.mock import MagicMock, patch

import pytest

from dags.etl_modules.adapters.market_data_repository import MarketDataRepository
from dags.etl_modules.errors import ConfigurationError


@pytest.mark.unit
@patch(
    "dags.etl_modules.adapters.market_data_repository.psycopg2.extras.execute_values"
)
def test_upsert_rows_in_batches_continues_on_failed_batch(mock_execute_values):
    repository = MarketDataRepository()
    conn = MagicMock()
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False

    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = False
    conn.cursor.return_value = cursor

    call_count = {"value": 0}

    def _execute(_cur, _query, batch_rows):
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise RuntimeError("temporary DB failure")
        return batch_rows

    mock_execute_values.side_effect = _execute
    rows = [(i,) for i in range(250)]

    failed_batches = repository.upsert_rows_in_batches(
        conn,
        "INSERT INTO any_table VALUES %s",
        rows,
        table_name="any_table",
        batch_size=100,
    )

    assert call_count["value"] == 3
    assert len(failed_batches) == 1
    assert failed_batches[0]["batch_index"] == 2
    assert conn.rollback.call_count == 1


@pytest.mark.unit
def test_upsert_rows_returns_fatal_error_when_db_url_missing():
    repository = MarketDataRepository()

    failed_batches, fatal_error = repository.upsert_rows(
        db_url=None,
        query="INSERT INTO any_table VALUES %s",
        rows=[("a",)],
        table_name="any_table",
        batch_size=10,
    )

    assert failed_batches == []
    assert fatal_error == "SUPABASE_DB_URL environment variable is not set"


@pytest.mark.unit
def test_upsert_rows_in_batches_rejects_non_positive_batch_size():
    repository = MarketDataRepository()
    conn = MagicMock()

    with pytest.raises(ConfigurationError):
        repository.upsert_rows_in_batches(
            conn,
            "INSERT INTO any_table VALUES %s",
            [(1,)],
            table_name="any_table",
            batch_size=0,
        )
