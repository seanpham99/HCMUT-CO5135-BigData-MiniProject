"""
Unit tests for dags/assets_dimension_etl.py

Tests cover:
- Precious metals ingestion mapping (XAU/XAG)
- Commodity classification and metadata mapping
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.mark.unit
class TestFetchVnInstruments:
    """Unit tests for fetch_vn_instruments type-based classification."""

    @patch("dags.assets_dimension_etl.fetch_vn_industry_metadata")
    @patch("dags.assets_dimension_etl.fetch_vn_listing_symbols")
    def test_maps_provider_types_to_supported_asset_classes(
        self, mock_listing_symbols, mock_industry_metadata
    ):
        from dags import assets_dimension_etl as module

        mock_listing_symbols.return_value = pd.DataFrame(
            {
                "symbol": [
                    "HPG",
                    "41I1G4000",
                    "E1VFVN30",
                    "C4G12005",
                    "BONDVN",
                    "QOPEN",
                    "DBONDX",
                    "CLPB2503",
                ],
                "organ_name": [
                    "Hoa Phat Group",
                    "VN30 Index Futures 042026",
                    "VN30 ETF",
                    "Covered Warrant",
                    "Government Bond Futures",
                    "Open-end Unit Trust",
                    "Corporate Debenture",
                    "Chứng quyền LPB/12M/SSI/C/EU/Cash-20",
                ],
                "exchange": ["HOSE", "HNX", "HOSE", "HOSE", "HNX", "HSX", "HSX", "HSX"],
                "type": [
                    "STOCK",
                    "FU",
                    "ETF",
                    "CW",
                    "FU_BOND",
                    "UNIT_TRUST",
                    "DEBENTURE",
                    None,
                ],
                "product_grp_id": [
                    "STO",
                    "FIO",
                    "STO",
                    "STO",
                    "FIO",
                    "STO",
                    "HCX",
                    "STO",
                ],
            }
        )
        mock_industry_metadata.return_value = pd.DataFrame(
            {
                "symbol": ["HPG", "E1VFVN30"],
                "icb_name2": ["Materials", "Fund"],
                "icb_name3": ["Steel", "ETF"],
            }
        )

        captured_records = []

        def capture_upsert(records):
            captured_records.extend(records)
            return len(records)

        with (
            patch.object(module, "upsert_assets_records", side_effect=capture_upsert),
            patch.object(module, "deactivate_stale_vn_stock_rows", return_value=0),
        ):
            result = module.fetch_vn_instruments()

        assert result == 8
        assert len(captured_records) == 8

        by_symbol = {row["symbol"]: row for row in captured_records}
        assert by_symbol["HPG"]["asset_class"] == "STOCK"
        assert by_symbol["41I1G4000"]["asset_class"] == "DERIVATIVE"
        assert by_symbol["E1VFVN30"]["asset_class"] == "ETF"
        assert by_symbol["C4G12005"]["asset_class"] == "DERIVATIVE"
        assert by_symbol["BONDVN"]["asset_class"] == "DERIVATIVE"
        assert by_symbol["QOPEN"]["asset_class"] == "FUND"
        assert by_symbol["DBONDX"]["asset_class"] == "BOND"
        assert by_symbol["CLPB2503"]["asset_class"] == "DERIVATIVE"
        assert by_symbol["41I1G4000"]["external_api_metadata"]["symbol_type"] == "FU"
        assert by_symbol["C4G12005"]["external_api_metadata"]["symbol_type"] == "CW"
        assert by_symbol["BONDVN"]["external_api_metadata"]["symbol_type"] == "FU_BOND"

    @patch("dags.assets_dimension_etl.fetch_vn_industry_metadata")
    @patch("dags.assets_dimension_etl.fetch_vn_listing_symbols")
    def test_keeps_symbols_when_type_column_absent(
        self, mock_listing_symbols, mock_industry_metadata
    ):
        from dags import assets_dimension_etl as module

        mock_listing_symbols.return_value = pd.DataFrame(
            {
                "symbol": ["HPG", "VCB"],
                "organ_name": ["Hoa Phat Group", "Vietcombank"],
                "exchange": ["HOSE", "HOSE"],
            }
        )
        mock_industry_metadata.side_effect = Exception("industry unavailable")

        captured_records = []

        def capture_upsert(records):
            captured_records.extend(records)
            return len(records)

        with patch.object(module, "upsert_assets_records", side_effect=capture_upsert):
            result = module.fetch_vn_instruments()

        assert result == 2
        assert {row["asset_class"] for row in captured_records} == {"STOCK"}

    @patch("dags.assets_dimension_etl.fetch_vn_industry_metadata")
    @patch("dags.assets_dimension_etl.fetch_vn_listing_symbols")
    def test_merges_when_provider_returns_icb_code4_only(
        self, mock_listing_symbols, mock_industry_metadata
    ):
        from dags import assets_dimension_etl as module

        mock_listing_symbols.return_value = pd.DataFrame(
            {
                "symbol": ["HPG"],
                "organ_name": ["Hoa Phat Group"],
                "exchange": ["HOSE"],
                "type": ["STOCK"],
                "product_grp_id": ["STO"],
            }
        )
        mock_industry_metadata.return_value = pd.DataFrame(
            {
                "symbol": ["HPG"],
                "icb_name2": ["Materials"],
                "icb_name3": ["Steel"],
                "icb_code4": ["55101010"],
            }
        )

        captured_records = []

        def capture_upsert(records):
            captured_records.extend(records)
            return len(records)

        with patch.object(module, "upsert_assets_records", side_effect=capture_upsert):
            result = module.fetch_vn_instruments()

        assert result == 1
        assert captured_records[0]["sector"] == "Materials"
        assert captured_records[0]["industry"] == "Steel"
        assert captured_records[0]["industry_code"] == "55101010"

    @patch("dags.assets_dimension_etl.fetch_vn_industry_metadata")
    @patch("dags.assets_dimension_etl.fetch_vn_listing_symbols")
    def test_deactivates_stale_stock_rows_for_symbols_reclassified_non_stock(
        self, mock_listing_symbols, mock_industry_metadata
    ):
        from dags import assets_dimension_etl as module

        mock_listing_symbols.return_value = pd.DataFrame(
            {
                "symbol": ["HPG", "CLPB2503"],
                "organ_name": [
                    "Hoa Phat Group",
                    "Chứng quyền LPB/12M/SSI/C/EU/Cash-20",
                ],
                "exchange": ["HOSE", "HSX"],
                "type": ["STOCK", None],
                "product_grp_id": ["STO", "STO"],
            }
        )
        mock_industry_metadata.side_effect = Exception("industry unavailable")

        with (
            patch.object(module, "upsert_assets_records", return_value=2),
            patch.object(
                module,
                "deactivate_stale_vn_stock_rows",
                return_value=1,
                create=True,
            ) as mock_deactivate,
        ):
            module.fetch_vn_instruments()

        mock_deactivate.assert_called_once_with(["CLPB2503"])


@pytest.mark.unit
class TestFetchPreciousMetals:
    """Unit tests for fetch_precious_metals task."""

    @patch("yfinance.Ticker")
    def test_inserts_xau_xag_with_commodity_class(self, mock_yf_ticker):
        from dags import assets_dimension_etl as module

        # Return predictable metadata for both provider symbols.
        def make_ticker(symbol):
            ticker = MagicMock()
            ticker.info = {
                "longName": "Provider Name",
                "exchange": "CMX",
            }
            return ticker

        mock_yf_ticker.side_effect = make_ticker
        captured_records = []

        def capture_upsert(records):
            captured_records.extend(records)
            return len(records)

        with patch.object(module, "upsert_assets_records", side_effect=capture_upsert):
            result = module.fetch_precious_metals()

        assert result == 2
        assert len(captured_records) == 2

        symbols = {row["symbol"] for row in captured_records}
        assert symbols == {"XAU", "XAG"}
        assert {row["asset_class"] for row in captured_records} == {"COMMODITY"}
        assert {row["currency"] for row in captured_records} == {"USD"}
        assert {row["sector"] for row in captured_records} == {"Precious Metals"}

        xau_row = next(row for row in captured_records if row["symbol"] == "XAU")
        xag_row = next(row for row in captured_records if row["symbol"] == "XAG")

        assert xau_row["external_api_metadata"]["source_symbol"] == "GC=F"
        assert xag_row["external_api_metadata"]["source_symbol"] == "SI=F"
        assert xau_row["external_api_metadata"]["commodity_type"] == "precious_metal"
        assert xag_row["external_api_metadata"]["unit"] == "troy_ounce"

    @patch("yfinance.Ticker")
    def test_falls_back_when_yfinance_enrichment_fails(self, mock_yf_ticker):
        from dags import assets_dimension_etl as module

        mock_yf_ticker.side_effect = Exception("provider unavailable")
        captured_records = []

        def capture_upsert(records):
            captured_records.extend(records)
            return len(records)

        with patch.object(module, "upsert_assets_records", side_effect=capture_upsert):
            result = module.fetch_precious_metals()

        assert result == 2
        assert len(captured_records) == 2
        # Fallback names are explicitly set by mapping.
        assert "Gold (XAU/USD)" in [row["name_en"] for row in captured_records]
        assert "Silver (XAG/USD)" in [row["name_en"] for row in captured_records]
