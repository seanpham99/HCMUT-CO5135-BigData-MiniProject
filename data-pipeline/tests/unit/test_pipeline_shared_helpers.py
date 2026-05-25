from datetime import date, datetime

import pytest

from dags.etl_modules.errors import ConfigurationError
from dags.etl_modules.orchestrators.shared_reporting import report_failed_symbols
from dags.etl_modules.transformers.shared_transformers import (
    chunk_assets,
    parse_date_value,
)


@pytest.mark.unit
def test_shared_chunk_assets_splits_batches():
    assets = [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}]

    chunks = chunk_assets(assets, chunk_size=2)

    assert chunks == [
        {"chunk_index": 1, "assets": [{"symbol": "AAA"}, {"symbol": "BBB"}]},
        {"chunk_index": 2, "assets": [{"symbol": "CCC"}]},
    ]


@pytest.mark.unit
def test_shared_parse_date_value_handles_common_inputs():
    assert parse_date_value("2025-01-10") == date(2025, 1, 10)
    assert parse_date_value(datetime(2025, 1, 10)) == date(2025, 1, 10)
    assert parse_date_value("NaT") is None


@pytest.mark.unit
def test_shared_report_failed_symbols_writes_output(capsys):
    report_failed_symbols(
        "stage-name",
        [{"symbol": "AAA", "error": "timeout"}],
    )

    output = capsys.readouterr().out
    assert "stage-name: failed symbols (1):" in output
    assert "- AAA: timeout" in output


@pytest.mark.unit
@pytest.mark.parametrize("chunk_size", [0, -1])
def test_shared_chunk_assets_raises_configuration_error_for_non_positive_size(
    chunk_size,
):
    with pytest.raises(ConfigurationError):
        chunk_assets([{"symbol": "AAA"}], chunk_size=chunk_size)
