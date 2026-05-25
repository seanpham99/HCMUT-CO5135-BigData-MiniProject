from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from dags.etl_modules import kbs_provider


@pytest.mark.unit
class TestKbsProviderIncomeStatement:
    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_income_statement_maps_metrics_and_periods(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [
                {"YearPeriod": 2025, "TermName": "Quý 4"},
                {"YearPeriod": 2025, "TermName": "Quý 3"},
            ],
            "Content": {
                "Kết quả kinh doanh": [
                    {
                        "Name": "Doanh thu thuan",
                        "NameEn": "Net Sales",
                        "Value1": 1000.0,
                        "Value2": 900.0,
                    },
                    {
                        "Name": "Gia von hang ban",
                        "NameEn": "Cost of Sales",
                        "Value1": -700.0,
                        "Value2": -620.0,
                    },
                    {
                        "Name": "Loi nhuan sau thue",
                        "NameEn": "Profit after tax",
                        "Value1": 120.0,
                        "Value2": 95.0,
                    },
                ]
            },
        }

        result = kbs_provider.fetch_income_statement("VCI", period="Q")

        assert not result.empty
        assert set(["yearReport", "lengthReport", "fiscal_date", "revenue"]).issubset(
            result.columns
        )
        assert result.iloc[0]["yearReport"] == 2025
        assert result.iloc[0]["lengthReport"] == 4
        assert result.iloc[0]["fiscal_date"] == "2025-12-31"
        assert result.iloc[0]["revenue"] == 1000.0
        assert result.iloc[0]["cost_of_goods_sold"] == -700.0
        assert result.iloc[0]["net_profit_post_tax"] == 120.0
        assert pd.isna(result.iloc[0]["ebitda"])

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_income_statement_year_period_defaults_to_q4(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [{"YearPeriod": 2024, "TermName": "Năm"}],
            "Content": {
                "Kết quả kinh doanh": [
                    {"Name": "Doanh thu", "NameEn": "Revenue", "Value1": 5000.0}
                ]
            },
        }

        result = kbs_provider.fetch_income_statement("VCI", period="Y")

        assert len(result) == 1
        assert result.iloc[0]["yearReport"] == 2024
        assert result.iloc[0]["lengthReport"] == 4
        assert result.iloc[0]["fiscal_date"] == "2024-12-31"

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_income_statement_returns_empty_on_kbs_404(self, mock_get):
        response = MagicMock()
        response.status_code = 404

        error = requests.HTTPError("404 Client Error: Not Found")
        error.response = response

        mock_get.return_value.raise_for_status.side_effect = error

        result = kbs_provider.fetch_income_statement("A32", period="Q")

        assert result.empty


@pytest.mark.unit
class TestKbsProviderBalanceSheet:
    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_balance_sheet_maps_core_metrics(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Cân đối kế toán": [
                    {
                        "Name": "Tong tai san",
                        "NameEn": "Total Assets",
                        "Value1": 10000.0,
                    },
                    {
                        "Name": "Tong no phai tra",
                        "NameEn": "Total Liabilities",
                        "Value1": 3500.0,
                    },
                    {
                        "Name": "Von chu so huu",
                        "NameEn": "Owner's Equity",
                        "Value1": 6500.0,
                    },
                ]
            },
        }

        result = kbs_provider.fetch_balance_sheet("VCI", period="Q")

        assert not result.empty
        assert result.iloc[0]["total_assets"] == 10000.0
        assert result.iloc[0]["total_liabilities"] == 3500.0
        assert result.iloc[0]["total_equity"] == 6500.0

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_balance_sheet_returns_empty_on_kbs_404(self, mock_get):
        response = MagicMock()
        response.status_code = 404

        error = requests.HTTPError("404 Client Error: Not Found")
        error.response = response

        mock_get.return_value.raise_for_status.side_effect = error

        result = kbs_provider.fetch_balance_sheet("VPW", period="Q")

        assert result.empty


@pytest.mark.unit
class TestKbsProviderRatios:
    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_ratio_maps_common_metrics(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Nhóm chỉ số Định giá": [
                    {"Name": "PE", "NameEn": "PE", "Value1": 12.0}
                ],
                "Nhóm chỉ số Sinh lợi": [
                    {"Name": "ROE", "NameEn": "ROE", "Value1": 0.18}
                ],
                "Nhóm chỉ số Thanh khoản": [
                    {"Name": "Current Ratio", "NameEn": "Current Ratio", "Value1": 1.6}
                ],
            },
        }

        result = kbs_provider.fetch_financial_ratios("VCI", period="Q")

        assert not result.empty
        assert result.iloc[0]["pe_ratio"] == 12.0
        assert result.iloc[0]["roe"] == 0.18
        assert result.iloc[0]["current_ratio"] == 1.6

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_ratio_maps_extended_canonical_metrics(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Nhóm chỉ số Định giá": [
                    {"Name": "P/CF", "NameEn": "P/Cash Flow", "Value1": 4.2}
                ],
                "Nhóm chỉ số Sinh lợi": [
                    {"Name": "ROIC", "NameEn": "ROIC", "Value1": 0.14},
                    {
                        "Name": "Financial Leverage",
                        "NameEn": "Financial Leverage",
                        "Value1": 1.9,
                    },
                    {
                        "Name": "Dividend Yield",
                        "NameEn": "Dividend Yield",
                        "Value1": 0.03,
                    },
                    {
                        "Name": "Net Profit Margin",
                        "NameEn": "Net Profit Margin",
                        "Value1": 0.11,
                    },
                ],
                "Nhóm chỉ số Thanh khoản": [
                    {
                        "Name": "Interest Coverage",
                        "NameEn": "Interest Coverage",
                        "Value1": 7.5,
                    }
                ],
                "Nhóm chỉ số Chất lượng tài sản": [
                    {
                        "Name": "Asset Turnover",
                        "NameEn": "Asset Turnover",
                        "Value1": 0.95,
                    },
                    {
                        "Name": "Inventory Turnover",
                        "NameEn": "Inventory Turnover",
                        "Value1": 5.1,
                    },
                    {
                        "Name": "Accounts Receivable Turnover",
                        "NameEn": "Accounts Receivable Turnover",
                        "Value1": 6.2,
                    },
                ],
                "Nhóm chỉ số Tăng trưởng": [
                    {
                        "Name": "Revenue Growth",
                        "NameEn": "Revenue Growth",
                        "Value1": 0.07,
                    },
                    {
                        "Name": "Profit Growth",
                        "NameEn": "Profit Growth",
                        "Value1": 0.08,
                    },
                    {
                        "Name": "Operating Margin",
                        "NameEn": "Operating Margin",
                        "Value1": 0.12,
                    },
                    {"Name": "Gross Margin", "NameEn": "Gross Margin", "Value1": 0.28},
                ],
                "Khác": [
                    {
                        "Name": "Free Cash Flow",
                        "NameEn": "Free Cash Flow",
                        "Value1": 1234.0,
                    },
                    {
                        "Name": "Market Capital",
                        "NameEn": "Market Capital",
                        "Value1": 1500000.0,
                    },
                ],
            },
        }

        result = kbs_provider.fetch_financial_ratios("VCI", period="Q")

        assert not result.empty
        assert result.iloc[0]["p_cashflow_ratio"] == 4.2
        assert result.iloc[0]["roic"] == 0.14
        assert result.iloc[0]["financial_leverage"] == 1.9
        assert result.iloc[0]["dividend_yield"] == 0.03
        assert result.iloc[0]["net_profit_margin"] == 0.11
        assert result.iloc[0]["interest_coverage"] == 7.5
        assert result.iloc[0]["asset_turnover"] == 0.95
        assert result.iloc[0]["inventory_turnover"] == 5.1
        assert result.iloc[0]["receivable_turnover"] == 6.2
        assert result.iloc[0]["revenue_growth"] == 0.07
        assert result.iloc[0]["profit_growth"] == 0.08
        assert result.iloc[0]["operating_margin"] == 0.12
        assert result.iloc[0]["gross_margin"] == 0.28
        assert result.iloc[0]["free_cash_flow"] == 1234.0
        assert result.iloc[0]["market_cap"] == 1500000.0

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_ratio_reads_multiple_pages_when_first_page_hits_limit(
        self, mock_get, monkeypatch
    ):
        monkeypatch.setattr(kbs_provider, "KBS_FINANCE_PAGE_SIZE", 2, raising=False)
        monkeypatch.setattr(kbs_provider, "KBS_FINANCE_MAX_PAGES", 5, raising=False)

        first = MagicMock()
        first.raise_for_status.return_value = None
        first.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Nhóm chỉ số Định giá": [
                    {"Name": "PE", "NameEn": "PE", "Value1": 12.0},
                    {"Name": "PB", "NameEn": "PB", "Value1": 1.8},
                ]
            },
        }

        second = MagicMock()
        second.raise_for_status.return_value = None
        second.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Nhóm chỉ số Sinh lợi": [
                    {"Name": "ROE", "NameEn": "ROE", "Value1": 0.2}
                ]
            },
        }

        mock_get.side_effect = [first, second]

        result = kbs_provider.fetch_financial_ratios("VCI", period="Q")

        assert not result.empty
        assert result.iloc[0]["pe_ratio"] == 12.0
        assert result.iloc[0]["roe"] == 0.2
        assert mock_get.call_count == 2

    @patch("dags.etl_modules.kbs_provider.requests.get")
    def test_fetch_uses_governed_call_wrapper(self, mock_get, monkeypatch):
        called = {"value": False}

        def _fake_governed_call(_source, request_fn, **_kwargs):
            called["value"] = True
            return request_fn()

        monkeypatch.setattr(
            kbs_provider,
            "governed_call",
            _fake_governed_call,
            raising=False,
        )
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "Head": [{"YearPeriod": 2025, "TermName": "Quý 4"}],
            "Content": {
                "Nhóm chỉ số Định giá": [{"Name": "PE", "NameEn": "PE", "Value1": 10.0}]
            },
        }

        result = kbs_provider.fetch_financial_ratios("VCI", period="Q")

        assert not result.empty
        assert called["value"] is True
