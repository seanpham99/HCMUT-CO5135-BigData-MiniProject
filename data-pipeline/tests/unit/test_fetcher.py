"""
Unit tests for dags/etl_modules/fetcher.py

Tests cover:
- clean_decimal_cols: Data cleaning for ClickHouse Decimal types
- fetch_stock_price: Stock price fetching with technical indicators
- fetch_financial_ratios: Financial ratios extraction
- fetch_income_stmt: Income statement data
- fetch_dividends: Dividend history
- fetch_news: News articles fetching
"""

from datetime import date, datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dags.etl_modules.fetcher import (
    FetcherFundamentalsProvider,
    clean_decimal_cols,
    fetch_balance_sheet,
    fetch_corporate_events,
    fetch_dividends,
    fetch_financial_ratios,
    fetch_income_stmt,
    fetch_news,
    fetch_stock_price,
    get_active_vn_stock_tickers,
)

# ============================================================================
# Tests for clean_decimal_cols()
# ============================================================================


@pytest.mark.unit
class TestCleanDecimalCols:
    """Test suite for clean_decimal_cols helper function."""

    def test_replaces_nan_with_zero(self):
        """Test that NaN values are replaced with 0."""
        df = pd.DataFrame({"price": [100.5, np.nan, 200.0]})
        result = clean_decimal_cols(df, ["price"])

        assert result["price"].isna().sum() == 0
        assert result["price"][1] == 0.0

    def test_replaces_infinity_with_zero(self):
        """Test that Infinity values are replaced with 0."""
        df = pd.DataFrame({"value": [10.0, np.inf, -np.inf, 20.0]})
        result = clean_decimal_cols(df, ["value"])

        assert not np.isinf(result["value"]).any()
        assert result["value"][1] == 0.0
        assert result["value"][2] == 0.0

    def test_coerces_string_to_numeric(self):
        """Test that string values are coerced to numeric (becomes NaN then 0)."""
        df = pd.DataFrame({"amount": ["100", "invalid", "200"]})
        result = clean_decimal_cols(df, ["amount"])

        assert result["amount"][0] == 100.0
        assert result["amount"][1] == 0.0  # 'invalid' -> NaN -> 0
        assert result["amount"][2] == 200.0

    def test_handles_multiple_columns(self):
        """Test cleaning multiple columns simultaneously."""
        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, 3.0],
                "col2": [np.inf, 5.0, -np.inf],
                "col3": [7.0, 8.0, 9.0],  # This one stays untouched
            }
        )
        result = clean_decimal_cols(df, ["col1", "col2"])

        assert result["col1"][1] == 0.0
        assert result["col2"][0] == 0.0
        assert result["col2"][2] == 0.0
        assert result["col3"][0] == 7.0  # Unchanged

    def test_handles_missing_columns_gracefully(self):
        """Test that function doesn't fail if column doesn't exist."""
        df = pd.DataFrame({"price": [100.0, 200.0]})
        # Should not raise an error
        result = clean_decimal_cols(df, ["price", "nonexistent_col"])

        assert result["price"][0] == 100.0
        assert "nonexistent_col" not in result.columns

    def test_preserves_valid_values(self):
        """Test that valid numeric values are preserved."""
        df = pd.DataFrame({"value": [1.5, 2.7, 3.14159, 0.0, -5.2]})
        result = clean_decimal_cols(df, ["value"])

        assert result["value"].tolist() == [1.5, 2.7, 3.14159, 0.0, -5.2]

    def test_handles_none_values(self):
        """Test that None values are replaced with 0."""
        df = pd.DataFrame({"price": [100.0, None, 200.0]})
        result = clean_decimal_cols(df, ["price"])

        assert result["price"][1] == 0.0

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()
        result = clean_decimal_cols(df, ["price"])

        assert result.empty


# ============================================================================
# Tests for fetch_stock_price()
# ============================================================================


@pytest.mark.unit
class TestFetchStockPrice:
    """Test suite for fetch_stock_price function."""

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_successful_fetch_returns_dataframe(self, mock_price_frame):
        """Test successful stock price fetch returns properly formatted DataFrame."""
        # Setup mock
        mock_df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5] * 10,
                "volume": [1000000] * 10,
            }
        )
        mock_price_frame.return_value = mock_df

        # Execute
        result = fetch_stock_price("HPG", "dummy_asset_id", "2024-01-01", "2024-01-10")

        # Assert
        assert not result.empty
        assert "ticker" in result.columns
        assert "trading_date" in result.columns
        assert result["ticker"].iloc[0] == "HPG"
        assert len(result) == 10

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_empty_response_returns_empty_dataframe(self, mock_price_frame):
        """Test that empty API response returns empty DataFrame."""
        mock_price_frame.return_value = pd.DataFrame()

        result = fetch_stock_price(
            "INVALID", "dummy_asset_id", "2024-01-01", "2024-01-10"
        )

        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_api_exception_returns_empty_dataframe(self, mock_price_frame):
        """Test that API exceptions are caught and return empty DataFrame."""
        mock_price_frame.side_effect = Exception("API Error")

        result = fetch_stock_price("HPG", "dummy_asset_id", "2024-01-01", "2024-01-10")

        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_no_technical_indicators_in_output(self, mock_price_frame):
        """Test that TA indicator columns are not included in output."""
        # Setup mock with enough data
        dates = pd.date_range("2024-01-01", periods=250)
        prices = [100 + i * 0.1 for i in range(250)]

        mock_df = pd.DataFrame(
            {
                "time": dates,
                "close": prices,
                "volume": [1000000] * 250,
            }
        )
        mock_price_frame.return_value = mock_df

        result = fetch_stock_price("HPG", "dummy_asset_id", "2024-01-01", "2024-12-31")

        # TA columns should not be present (stripped in Phase 2 cleanup)
        for col in [
            "ma_50",
            "ma_200",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "daily_return",
        ]:
            assert col not in result.columns, (
                f"Stale TA column '{col}' should not be in output"
            )

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_nan_values_cleaned(self, mock_price_frame):
        """Test that NaN values in close prices are cleaned to 0."""
        mock_df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=10),
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.5, np.nan, 102.0] + [100.0] * 7,
                "volume": [1000000] * 10,
            }
        )
        mock_price_frame.return_value = mock_df

        result = fetch_stock_price("HPG", "dummy_asset_id", "2024-01-01", "2024-01-10")

        assert result["close"].isna().sum() == 0
        assert result["close"].iloc[1] == 0.0

    @patch("dags.etl_modules.fetcher.fetch_stock_price_frame")
    def test_trading_date_converted_to_date(self, mock_price_frame):
        """Test that trading_date is converted to date type (not datetime)."""
        mock_df = pd.DataFrame(
            {
                "trading_date": pd.date_range("2024-01-01", periods=5),
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.5] * 5,
                "volume": [1000000] * 5,
            }
        )
        mock_price_frame.return_value = mock_df

        result = fetch_stock_price("HPG", "dummy_asset_id", "2024-01-01", "2024-01-05")

        # Check that trading_date is date type, not datetime
        assert isinstance(result["trading_date"].iloc[0], date)
        assert not isinstance(result["trading_date"].iloc[0], datetime)


# ============================================================================
# Tests for fetch_financial_ratios()
# ============================================================================


@pytest.mark.unit
class TestFetchFinancialRatios:
    """Test suite for fetch_financial_ratios function."""

    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    def test_successful_fetch_returns_dataframe(self, mock_ratio_frame_vci):
        """Test successful ratio fetch returns properly formatted DataFrame."""
        mock_ratio_frame_vci.return_value = pd.DataFrame(
            {
                "yearReport": [2024, 2024],
                "lengthReport": [4, 3],
                "P/E": [15.5, 16.2],
                "P/B": [2.1, 2.3],
                "P/S": [1.5, 1.6],
                "ROE (%)": [18.5, 19.2],
                "EPS": [5000, 5200],
            }
        )

        result = fetch_financial_ratios("HPG", "dummy_asset_id")

        assert not result.empty
        assert "ticker" in result.columns
        assert "fiscal_date" in result.columns
        assert result["ticker"].iloc[0] == "HPG"

    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame")
    def test_empty_response_returns_empty_dataframe(
        self, mock_ratio_frame_kbs, mock_ratio_frame_vci
    ):
        """Test that empty response returns empty DataFrame."""
        mock_ratio_frame_vci.return_value = pd.DataFrame()
        mock_ratio_frame_kbs.return_value = pd.DataFrame()

        result = fetch_financial_ratios("INVALID", "dummy_asset_id")

        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    def test_column_mapping_applied(self, mock_ratio_frame_vci):
        """Test that Vietnamese column names are mapped to English."""
        mock_ratio_frame_vci.return_value = pd.DataFrame(
            {
                "yearReport": [2024],
                "lengthReport": [4],
                "P/E": [15.5],
                "ROE (%)": [18.5],
            }
        )

        result = fetch_financial_ratios("HPG", "dummy_asset_id")

        assert "pe_ratio" in result.columns
        assert "roe" in result.columns

    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    def test_alias_and_derived_metric_mapping_populates_extended_fields(
        self, mock_ratio_frame_vci
    ):
        mock_ratio_frame_vci.return_value = pd.DataFrame(
            {
                "yearReport": [2025],
                "lengthReport": [4],
                "Revenue (Bn. VND)": [1000.0],
                "Revenue YoY (%)": [0.12],
                "Operating Profit/Loss": [220.0],
                "Gross Profit": [500.0],
                "Accounts Receivables": [250.0],
                "Attribute to parent company YoY (%)": [0.08],
                "Net cash inflows/outflows from operating activities": [150.0],
            }
        )

        result = fetch_financial_ratios("HPG", "dummy_asset_id_v2")

        assert result["revenue_growth"].iloc[0] == pytest.approx(0.12)
        assert result["profit_growth"].iloc[0] == pytest.approx(0.08)
        assert result["operating_margin"].iloc[0] == pytest.approx(0.22)
        assert result["gross_margin"].iloc[0] == pytest.approx(0.5)
        assert result["receivable_turnover"].iloc[0] == pytest.approx(4.0)
        assert result["free_cash_flow"].iloc[0] == pytest.approx(150.0)

    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    def test_duplicate_ratio_columns_are_coalesced(self, mock_ratio_frame_vci):
        mock_ratio_frame_vci.return_value = pd.DataFrame(
            [[2025, 4, 12.5, 13.0, 1.3, 0.18]],
            columns=["yearReport", "lengthReport", "P/E", "P/E", "P/B", "ROE (%)"],
        )

        result = fetch_financial_ratios("HPG", "asset-dup")

        assert not result.empty
        assert result["pe_ratio"].iloc[0] == pytest.approx(12.5)
        assert result["pb_ratio"].iloc[0] == pytest.approx(1.3)

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame")
    @patch("dags.etl_modules.fetcher.fetch_financial_ratio_frame_vci")
    def test_ratios_fall_back_to_kbs_when_vci_empty(
        self,
        mock_ratio_vci,
        mock_ratio_kbs,
        _mock_get_redis_client,
    ):
        mock_ratio_vci.return_value = pd.DataFrame()
        mock_ratio_kbs.return_value = pd.DataFrame(
            {
                "yearReport": [2024],
                "lengthReport": [4],
                "P/E": [11.2],
            }
        )

        result = fetch_financial_ratios("HPG", "dummy_asset_id")

        assert not result.empty
        assert result["source_provider"].iloc[0] == "KBS"
        assert result["pe_ratio"].iloc[0] == pytest.approx(11.2)


# ============================================================================
# Placeholder tests for other functions (to be expanded)
# ============================================================================


@pytest.mark.unit
def test_fetch_income_stmt_placeholder():
    """Placeholder test for fetch_income_stmt - to be implemented."""
    # Placeholder removed in Phase 2
    assert True


@pytest.mark.unit
def test_fetch_dividends_placeholder():
    """Placeholder test for fetch_dividends - to be implemented."""
    # Placeholder removed in Phase 2
    assert True


@pytest.mark.unit
def test_fetch_news_placeholder():
    """Placeholder test for fetch_news - to be implemented."""
    # Placeholder removed in Phase 2
    assert True


# ============================================================================
# Phase 2: Tests for fetch_income_stmt(), fetch_dividends(), fetch_news()
# ============================================================================


@pytest.mark.unit
class TestFetchIncomeStmt:
    """Unit tests for fetch_income_stmt function."""

    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_success(self, mock_income_frame_vci):
        mock_income_frame_vci.return_value = pd.DataFrame(
            {
                "Net Sales": [5_000_000_000_000],
                "Cost of Sales": [3_500_000_000_000],
                "Gross Profit": [1_500_000_000_000],
                "Operating Profit/Loss": [800_000_000_000],
                "Net Profit For the Year": [600_000_000_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert not result.empty
        assert set(
            [
                "ticker",
                "fiscal_date",
                "year",
                "quarter",
                "source_provider",
                "revenue",
                "cost_of_goods_sold",
                "gross_profit",
                "operating_profit",
                "net_profit_post_tax",
            ]
        ).issubset(result.columns)
        assert result["ticker"].iloc[0] == "HPG"
        assert result["fiscal_date"].iloc[0] == "2024-12-31"
        assert result["source_provider"].iloc[0] == "VCI"
        # Values preserved and cleaned
        assert result["revenue"].iloc[0] == 5_000_000_000_000

    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_handles_missing_columns(self, mock_income_frame_vci):
        mock_income_frame_vci.return_value = pd.DataFrame(
            {
                "Net Sales": [5_000_000_000_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert not result.empty
        assert pd.isna(result["operating_profit"].iloc[0])
        assert pd.isna(result["net_profit_post_tax"].iloc[0])

    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame")
    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_empty_dataframe(
        self, mock_income_frame_vci, mock_income_frame
    ):
        mock_income_frame_vci.return_value = pd.DataFrame()
        mock_income_frame.return_value = pd.DataFrame()

        result = fetch_income_stmt("HPG", "dummy_asset_id")
        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_problematic_fields_remain_null_when_missing(
        self, mock_income_frame_vci
    ):
        mock_income_frame_vci.return_value = pd.DataFrame(
            {
                "Net Sales": [3_000_000_000],
                "Gross Profit": [800_000_000],
                "yearReport": [2025],
                "lengthReport": [4],
            }
        )

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert not result.empty
        for col in [
            "net_profit_post_tax",
            "admin_expenses",
            "financial_expenses",
            "other_income",
            "other_expenses",
            "ebitda",
        ]:
            assert pd.isna(result[col].iloc[0])

    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_coalesces_duplicate_mapped_columns(
        self, mock_income_frame_vci
    ):
        mock_income_frame_vci.return_value = pd.DataFrame(
            {
                "Net Sales": [1000],
                "Selling Expense": [None],
                "Selling Expenses": [125],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert not result.empty
        assert result["selling_expenses"].iloc[0] == 125.0

    @patch("dags.etl_modules.fetcher.logging.error")
    @patch("dags.etl_modules.fetcher.logging.warning")
    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame")
    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_transient_vci_failures_log_warning(
        self,
        mock_income_frame_vci,
        mock_income_frame,
        mock_warning,
        mock_error,
    ):
        mock_income_frame_vci.side_effect = RuntimeError(
            "Failed to reach https://trading.vietcap.com.vn/data-mt/graphql: "
            "<urlopen error [SSL: SSLV3_ALERT_CERTIFICATE_UNKNOWN] certificate unknown>"
        )
        mock_income_frame.side_effect = RuntimeError("KBS fallback also failed")

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert result.empty
        assert any(
            str(call.args[0]).startswith("VCI %s fetch failed")
            for call in mock_warning.call_args_list
        )
        mock_error.assert_not_called()

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame")
    @patch("dags.etl_modules.fetcher.fetch_income_statement_frame_vci")
    def test_income_stmt_falls_back_to_kbs_when_vci_empty(
        self,
        mock_income_vci,
        mock_income_kbs,
        _mock_get_redis_client,
    ):
        mock_income_vci.return_value = pd.DataFrame()
        mock_income_kbs.return_value = pd.DataFrame(
            {
                "Net Sales": [2_000_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_income_stmt("HPG", "dummy_asset_id")

        assert not result.empty
        assert result["source_provider"].iloc[0] == "KBS"
        assert result["revenue"].iloc[0] == 2_000_000


@pytest.mark.unit
class TestFetchBalanceSheet:
    """Unit tests for fetch_balance_sheet function."""

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame_vci")
    def test_balance_sheet_maps_vietnamese_labels(
        self, mock_balance_sheet_frame_vci, _mock_get_redis_client
    ):
        mock_balance_sheet_frame_vci.return_value = pd.DataFrame(
            {
                "Tổng tài sản": [1_000_000],
                "Tổng nợ phải trả": [400_000],
                "Vốn chủ sở hữu": [600_000],
                "Tiền và tương đương tiền": [120_000],
                "Tài sản ngắn hạn": [300_000],
                "Tài sản dài hạn": [700_000],
                "Nợ ngắn hạn": [200_000],
                "Nợ dài hạn": [200_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_balance_sheet("VNM", "asset-vnm")

        assert len(result) == 1
        assert result["fiscal_date"].iloc[0] == "2024-12-31"
        assert result["total_assets"].iloc[0] == 1_000_000
        assert result["total_liabilities"].iloc[0] == 400_000
        assert result["total_equity"].iloc[0] == 600_000
        assert result["cash_and_equivalents"].iloc[0] == 120_000
        assert result["short_term_assets"].iloc[0] == 300_000
        assert result["long_term_assets"].iloc[0] == 700_000
        assert result["short_term_liabilities"].iloc[0] == 200_000
        assert result["long_term_liabilities"].iloc[0] == 200_000

    @patch("dags.etl_modules.fetcher.logging.error")
    @patch("dags.etl_modules.fetcher.logging.warning")
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame")
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame_vci")
    def test_balance_sheet_transient_vci_failures_log_warning(
        self,
        mock_balance_sheet_frame_vci,
        mock_balance_sheet_frame,
        mock_warning,
        mock_error,
    ):
        mock_balance_sheet_frame_vci.side_effect = RuntimeError(
            "HTTP 502 from https://trading.vietcap.com.vn/data-mt/graphql: bad gateway"
        )
        mock_balance_sheet_frame.side_effect = RuntimeError("KBS fallback also failed")

        result = fetch_balance_sheet("HPG", "asset-hpg")

        assert result.empty
        assert any(
            str(call.args[0]).startswith("VCI %s fetch failed")
            for call in mock_warning.call_args_list
        )
        mock_error.assert_not_called()

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame_vci")
    def test_balance_sheet_maps_case_insensitive_labels(
        self, mock_balance_sheet_frame_vci, _mock_get_redis_client
    ):
        mock_balance_sheet_frame_vci.return_value = pd.DataFrame(
            {
                "total assets": [2_000_000],
                "total liabilities": [800_000],
                "owner's equity": [1_200_000],
                "cash and cash equivalents": [200_000],
                "short term assets": [900_000],
                "long term assets": [1_100_000],
                "short term liabilities": [400_000],
                "long term liabilities": [400_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_balance_sheet("AAA", "asset-aaa")

        assert len(result) == 1
        assert result["total_assets"].iloc[0] == 2_000_000
        assert result["total_liabilities"].iloc[0] == 800_000
        assert result["total_equity"].iloc[0] == 1_200_000
        assert result["cash_and_equivalents"].iloc[0] == 200_000

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame_vci")
    def test_balance_sheet_maps_vci_uppercase_and_derives_missing_totals(
        self, mock_balance_sheet_frame_vci, _mock_get_redis_client
    ):
        mock_balance_sheet_frame_vci.return_value = pd.DataFrame(
            {
                "TOTAL ASSETS": [1_000_000],
                "OWNER'S EQUITY": [600_000],
                "CURRENT ASSETS": [300_000],
                "NON-CURRENT ASSETS": [700_000],
                "Current liabilities": [200_000],
                "Long-term liabilities": [200_000],
                "Cash and cash equivalents": [120_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_balance_sheet("VCI", "asset-vci")

        assert len(result) == 1
        assert result["total_assets"].iloc[0] == 1_000_000
        assert result["short_term_assets"].iloc[0] == 300_000
        assert result["long_term_assets"].iloc[0] == 700_000
        assert result["short_term_liabilities"].iloc[0] == 200_000
        assert result["long_term_liabilities"].iloc[0] == 200_000
        # Derived from short + long liabilities (or total_assets - total_equity)
        assert result["total_liabilities"].iloc[0] == 400_000

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame")
    @patch("dags.etl_modules.fetcher.fetch_balance_sheet_frame_vci")
    def test_balance_sheet_falls_back_to_kbs_when_vci_empty(
        self,
        mock_balance_vci,
        mock_balance_kbs,
        _mock_get_redis_client,
    ):
        mock_balance_vci.return_value = pd.DataFrame()
        mock_balance_kbs.return_value = pd.DataFrame(
            {
                "Tổng tài sản": [800_000],
                "Tổng nợ phải trả": [300_000],
                "Vốn chủ sở hữu": [500_000],
                "yearReport": [2024],
                "lengthReport": [4],
            }
        )

        result = fetch_balance_sheet("VNM", "asset-vnm")

        assert not result.empty
        assert result["source_provider"].iloc[0] == "KBS"
        assert result["total_assets"].iloc[0] == 800_000


@pytest.mark.unit
class TestFetchDividends:
    """Unit tests for fetch_dividends function."""

    @patch("dags.etl_modules.fetcher.fetch_vietstock_dividends_frame")
    def test_dividends_success(self, mock_dividends_frame):
        mock_dividends_frame.return_value = pd.DataFrame(
            {
                "exercise_date": [date(2024, 6, 15)],
                "cash_year": [2024],
                "cash_dividend_percentage": [15.0],
                "stock_dividend_percentage": [0.0],
                "issue_method": ["Cash dividend"],
            }
        )

        result = fetch_dividends("HPG", "dummy_asset_id")

        assert not result.empty
        assert set(
            [
                "ticker",
                "exercise_date",
                "cash_year",
                "cash_dividend_percentage",
                "stock_dividend_percentage",
                "issue_method",
            ]
        ).issubset(result.columns)
        assert result["ticker"].iloc[0] == "HPG"
        assert str(result["exercise_date"].iloc[0]) == "2024-06-15"

    @patch("dags.etl_modules.fetcher.fetch_vietstock_dividends_frame")
    def test_dividends_missing_fields_filled(self, mock_dividends_frame):
        mock_dividends_frame.return_value = pd.DataFrame(
            {"exercise_date": ["2024-06-15"]}
        )

        result = fetch_dividends("HPG", "dummy_asset_id")
        assert not result.empty
        assert result["cash_year"].iloc[0] == 0
        assert result["cash_dividend_percentage"].iloc[0] == 0.0
        assert result["issue_method"].iloc[0] is None

    @patch("dags.etl_modules.fetcher.fetch_vietstock_dividends_frame")
    def test_dividends_empty_dataframe(self, mock_dividends_frame):
        mock_dividends_frame.return_value = pd.DataFrame()

        result = fetch_dividends("HPG", "dummy_asset_id")
        assert result.empty


@pytest.mark.unit
class TestFetchCorporateEvents:
    """Unit tests for fetch_corporate_events function."""

    @patch("dags.etl_modules.fetcher.fetch_vietstock_corporate_events_frame")
    def test_corporate_events_success(self, mock_events_frame):
        mock_events_frame.return_value = pd.DataFrame(
            {
                "event_id": [12345],
                "event_date": [date(2026, 5, 7)],
                "public_date": [date(2026, 4, 6)],
                "exright_date": [date(2026, 4, 5)],
                "event_title": ["VCI dividend notice"],
                "event_type": ["Cash dividend"],
                "event_description": ["Cash payout 5%"],
            }
        )

        result = fetch_corporate_events("VCI", "asset-vci")

        assert len(result) == 1
        assert result["asset_id"].iloc[0] == "asset-vci"
        assert result["event_id"].iloc[0] == "12345"
        assert str(result["exright_date"].iloc[0]) == "2026-04-05"

    @patch("dags.etl_modules.fetcher.fetch_vietstock_corporate_events_frame")
    def test_corporate_events_empty(self, mock_events_frame):
        mock_events_frame.return_value = pd.DataFrame()
        result = fetch_corporate_events("VCI", "asset-vci")
        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_vietstock_corporate_events_frame")
    def test_corporate_events_honors_explicit_date_window(self, mock_events_frame):
        mock_events_frame.return_value = pd.DataFrame(
            {
                "event_id": [12345],
                "event_date": [date(2026, 5, 7)],
                "public_date": [date(2026, 4, 6)],
                "exright_date": [date(2026, 4, 5)],
                "event_title": ["VCI dividend notice"],
                "event_type": ["Cash dividend"],
                "event_description": ["Cash payout 5%"],
            }
        )

        result = fetch_corporate_events(
            "VCI",
            "asset-vci",
            from_date="2026-01-01",
            to_date="2026-02-01",
        )

        assert not result.empty
        kwargs = mock_events_frame.call_args.kwargs
        assert kwargs["from_date"] == "2026-01-01"
        assert kwargs["to_date"] == "2026-02-01"


@pytest.mark.unit
class TestFetchNews:
    """Unit tests for fetch_news function."""

    @patch("dags.etl_modules.fetcher.fetch_company_news")
    def test_news_success(self, mock_news_frame):
        mock_news_frame.return_value = pd.DataFrame(
            {
                "publish_date": ["2024-12-20T10:30:00"],
                "title": ["HPG announces strong Q4 results"],
                "source": ["CafeF"],
                "price_at_publish": [25500],
                "price_change": [2.5],
                "price_change_ratio": [0.012],
                "rsi": [45.2],
                "rs": [0.5],
                "news_id": [12345],
            }
        )

        result = fetch_news("HPG", "dummy_asset_id")

        assert not result.empty
        assert set(
            [
                "ticker",
                "publish_date",
                "title",
                "source",
                "price_at_publish",
                "price_change",
                "price_change_ratio",
                "rsi",
                "rs",
                "news_id",
            ]
        ).issubset(result.columns)
        assert result["ticker"].iloc[0] == "HPG"
        assert (
            pd.to_datetime(result["publish_date"].iloc[0]).strftime("%Y-%m-%d")
            == "2024-12-20"
        )
        assert result["price_at_publish"].iloc[0] == 25500

    @patch("dags.etl_modules.fetcher.fetch_company_news")
    def test_news_missing_fields_filled(self, mock_news_frame):
        mock_news_frame.return_value = pd.DataFrame(
            {
                "publish_date": ["2024-12-20T10:30:00"],
                "title": ["HPG update"],
                "source": ["CafeF"],
            }
        )

        result = fetch_news("HPG", "dummy_asset_id")
        assert not result.empty
        assert result["price_at_publish"].iloc[0] == 0.0
        # Default for non-price/rs fields is None per implementation
        assert result["news_id"].iloc[0] is None

    @patch("dags.etl_modules.fetcher.fetch_company_news")
    def test_news_empty_dataframe(self, mock_news_frame):
        mock_news_frame.return_value = pd.DataFrame()

        result = fetch_news("HPG", "dummy_asset_id")
        assert result.empty

    @patch("dags.etl_modules.fetcher.fetch_company_news")
    def test_news_rate_limit_system_exit_returns_empty(self, mock_news_frame):
        """Provider errors should be handled gracefully."""
        mock_news_frame.side_effect = SystemExit("Rate limit exceeded")

        result = fetch_news("HPG", "dummy_asset_id")
        assert result.empty

    @patch("dags.etl_modules.cache.get_redis_client", return_value=None)
    @patch("dags.etl_modules.fetcher.fetch_company_news")
    def test_news_relative_source_url_does_not_drop_row(
        self, mock_news_frame, _mock_get_redis_client
    ):
        """Malformed/relative URLs should not crash source extraction."""
        mock_news_frame.return_value = pd.DataFrame(
            {
                "publish_date": ["2024-12-20T10:30:00"],
                "title": ["HPG update"],
                "news_source_link": ["relative/path"],
            }
        )

        result = fetch_news("HPG", "dummy_asset_id")

        assert not result.empty
        assert result["source"].iloc[0] is None


@pytest.mark.unit
class TestGetActiveVnStockTickers:
    """Unit tests for get_active_vn_stock_tickers ticker filtering."""

    @patch("dags.etl_modules.fetcher.list_active_vn_stock_tickers_frame")
    def test_normalizes_symbols_and_deduplicates(self, mock_ticker_frame):
        mock_ticker_frame.return_value = [
            {"symbol": "hpg"},
            {"symbol": "VCB"},
            {"symbol": " VCB "},
            {"symbol": None},
            {"symbol": ""},
            {"symbol": "   "},
            {"symbol": "FPT"},
        ]

        result = get_active_vn_stock_tickers()

        assert result == [
            {"symbol": "HPG", "asset_id": "fallback"},
            {"symbol": "VCB", "asset_id": "fallback"},
            {"symbol": "FPT", "asset_id": "fallback"},
        ]

    @patch("dags.etl_modules.fetcher.list_active_vn_stock_tickers_frame")
    def test_ignores_only_explicit_non_stock_symbol_type(self, mock_ticker_frame):
        mock_ticker_frame.return_value = [
            {"id": "asset-1", "symbol": "HPG", "metadata": {"symbol_type": "STOCK"}},
            {"id": "asset-2", "symbol": "CLPB2503", "metadata": {"symbol_type": "CW"}},
            {"id": "asset-3", "symbol": "41I1G4000", "metadata": {}},
            {"id": "asset-4", "symbol": "TV2", "metadata": {}},
        ]

        result = get_active_vn_stock_tickers()

        assert result == [
            {"symbol": "HPG", "asset_id": "asset-1"},
            {"symbol": "41I1G4000", "asset_id": "asset-3"},
            {"symbol": "TV2", "asset_id": "asset-4"},
        ]

    @patch("dags.etl_modules.fetcher.list_active_vn_stock_tickers_frame")
    def test_prefers_asset_id_key_from_provider_rows(self, mock_ticker_frame):
        mock_ticker_frame.return_value = [
            {"asset_id": "asset-hpg", "symbol": "HPG"},
            {"id": "asset-vcb", "symbol": "VCB"},
        ]

        result = get_active_vn_stock_tickers()

        assert result == [
            {"symbol": "HPG", "asset_id": "asset-hpg"},
            {"symbol": "VCB", "asset_id": "asset-vcb"},
        ]


@pytest.mark.unit
class TestFetcherFundamentalsProvider:
    def test_list_assets_delegates_with_raise_on_fallback_true(self, monkeypatch):
        expected_assets = [{"symbol": "HPG", "asset_id": "asset-hpg"}]
        captured = {}

        def _fake_get_active_vn_stock_tickers(*, raise_on_fallback=False):
            captured["raise_on_fallback"] = raise_on_fallback
            return expected_assets

        monkeypatch.setattr(
            "dags.etl_modules.fetcher.get_active_vn_stock_tickers",
            _fake_get_active_vn_stock_tickers,
        )

        provider = FetcherFundamentalsProvider()
        result = provider.list_assets()

        assert result == expected_assets
        assert captured["raise_on_fallback"] is True

    def test_fetch_income_statement_delegates_to_fetcher_function(self, monkeypatch):
        expected_frame = pd.DataFrame([{"asset_id": "asset-hpg", "revenue": 100.0}])
        captured = {}

        def _fake_fetch_income_stmt(symbol, asset_id):
            captured["symbol"] = symbol
            captured["asset_id"] = asset_id
            return expected_frame

        monkeypatch.setattr(
            "dags.etl_modules.fetcher.fetch_income_stmt",
            _fake_fetch_income_stmt,
        )

        provider = FetcherFundamentalsProvider()
        result = provider.fetch_income_statement("HPG", "asset-hpg")

        assert result is expected_frame
        assert captured == {"symbol": "HPG", "asset_id": "asset-hpg"}

    def test_fetch_balance_sheet_delegates_to_fetcher_function(self, monkeypatch):
        expected_frame = pd.DataFrame(
            [{"asset_id": "asset-vcb", "total_assets": 200.0}]
        )
        captured = {}

        def _fake_fetch_balance_sheet(symbol, asset_id):
            captured["symbol"] = symbol
            captured["asset_id"] = asset_id
            return expected_frame

        monkeypatch.setattr(
            "dags.etl_modules.fetcher.fetch_balance_sheet",
            _fake_fetch_balance_sheet,
        )

        provider = FetcherFundamentalsProvider()
        result = provider.fetch_balance_sheet("VCB", "asset-vcb")

        assert result is expected_frame
        assert captured == {"symbol": "VCB", "asset_id": "asset-vcb"}
