import pytest

from dags.etl_modules.transformers import price_transformers, ratio_transformers


@pytest.mark.unit
def test_price_row_conversion_failure_reports_symbol_and_asset_id():
    _, failed_rows = price_transformers.convert_records_to_rows(
        records=[
            {
                "symbol": "AAA",
                "asset_id": "asset-1",
                "trading_date": "not-a-date",
                "open": 1.0,
            }
        ],
        price_columns=("trading_date", "open", "asset_id"),
    )

    assert len(failed_rows) == 1
    assert failed_rows[0]["symbol"] == "AAA"
    assert failed_rows[0]["asset_id"] == "asset-1"
    assert "price row conversion failed" in failed_rows[0]["error"]


@pytest.mark.unit
def test_price_row_conversion_failure_falls_back_to_ticker():
    _, failed_rows = price_transformers.convert_records_to_rows(
        records=[
            {
                "ticker": "VIC",
                "asset_id": "asset-3",
                "trading_date": "not-a-date",
                "open": 1.0,
            }
        ],
        price_columns=("trading_date", "open", "asset_id"),
    )

    assert len(failed_rows) == 1
    assert failed_rows[0]["symbol"] == "VIC"
    assert failed_rows[0]["asset_id"] == "asset-3"
    assert "price row conversion failed" in failed_rows[0]["error"]


@pytest.mark.unit
def test_ratio_row_conversion_failure_reports_symbol_and_asset_id():
    _, failed_rows, _ = ratio_transformers.convert_records_to_rows(
        records=[
            {
                "symbol": "BBB",
                "asset_id": "asset-2",
                "fiscal_date": "not-a-date",
                "pe_ratio": 1.2,
            }
        ],
        ratio_columns=("asset_id", "fiscal_date", "pe_ratio"),
        numeric_sanitize_columns=("pe_ratio",),
    )

    assert len(failed_rows) == 1
    assert failed_rows[0]["symbol"] == "BBB"
    assert failed_rows[0]["asset_id"] == "asset-2"
    assert "ratio row conversion failed" in failed_rows[0]["error"]


@pytest.mark.unit
def test_ratio_row_conversion_failure_falls_back_to_ticker():
    _, failed_rows, _ = ratio_transformers.convert_records_to_rows(
        records=[
            {
                "ticker": "HPG",
                "asset_id": "asset-4",
                "fiscal_date": "not-a-date",
                "pe_ratio": 1.2,
            }
        ],
        ratio_columns=("asset_id", "fiscal_date", "pe_ratio"),
        numeric_sanitize_columns=("pe_ratio",),
    )

    assert len(failed_rows) == 1
    assert failed_rows[0]["symbol"] == "HPG"
    assert failed_rows[0]["asset_id"] == "asset-4"
    assert "ratio row conversion failed" in failed_rows[0]["error"]
