from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dags.etl_modules import vci_provider


@pytest.mark.unit
class TestFinancialStatementSelection:
    @patch("dags.etl_modules.vci_provider._graphql")
    @patch("dags.etl_modules.vci_provider._company_type_code")
    @patch("dags.etl_modules.vci_provider._financial_ratio_metadata")
    def test_income_statement_selects_income_fields_only(
        self, mock_metadata, mock_company_type, mock_graphql
    ):
        mock_company_type.return_value = "CT"
        mock_metadata.return_value = pd.DataFrame(
            [
                {
                    "field_name": "is_net_sales",
                    "en_name": "Net Sales",
                    "name": "Doanh thu thuần",
                    "type": "Chỉ tiêu kết quả kinh doanh",
                    "com_type_code": "CT",
                    "order": 1,
                },
                {
                    "field_name": "bs_total_asset",
                    "en_name": "Total Asset",
                    "name": "Tổng tài sản",
                    "type": "Chỉ tiêu cân đối kế toán",
                    "com_type_code": "CT",
                    "order": 2,
                },
            ]
        )
        mock_graphql.return_value = {
            "CompanyFinancialRatio": {
                "ratio": [
                    {
                        "ticker": "HPG",
                        "yearReport": 2024,
                        "lengthReport": 4,
                        "updateDate": "2024-12-31",
                        "is_net_sales": 1000,
                        "bs_total_asset": 5000,
                    }
                ]
            }
        }

        df = vci_provider.fetch_income_statement("HPG", period="Q")

        assert "Net Sales" in df.columns
        assert "Total Asset" not in df.columns

    @patch("dags.etl_modules.vci_provider._graphql")
    @patch("dags.etl_modules.vci_provider._company_type_code")
    @patch("dags.etl_modules.vci_provider._financial_ratio_metadata")
    def test_balance_sheet_selects_balance_fields_only(
        self, mock_metadata, mock_company_type, mock_graphql
    ):
        mock_company_type.return_value = "CT"
        mock_metadata.return_value = pd.DataFrame(
            [
                {
                    "field_name": "is_net_sales",
                    "en_name": "Net Sales",
                    "name": "Doanh thu thuần",
                    "type": "Chỉ tiêu kết quả kinh doanh",
                    "com_type_code": "CT",
                    "order": 1,
                },
                {
                    "field_name": "bs_total_asset",
                    "en_name": "Total Asset",
                    "name": "Tổng tài sản",
                    "type": "Chỉ tiêu cân đối kế toán",
                    "com_type_code": "CT",
                    "order": 2,
                },
            ]
        )
        mock_graphql.return_value = {
            "CompanyFinancialRatio": {
                "ratio": [
                    {
                        "ticker": "HPG",
                        "yearReport": 2024,
                        "lengthReport": 4,
                        "updateDate": "2024-12-31",
                        "is_net_sales": 1000,
                        "bs_total_asset": 5000,
                    }
                ]
            }
        }

        df = vci_provider.fetch_balance_sheet("HPG", period="Q")

        assert "Total Asset" in df.columns
        assert "Net Sales" not in df.columns


@pytest.mark.unit
class TestIndustryMetadataNormalization:
    @patch("dags.etl_modules.vci_provider._graphql")
    def test_industry_metadata_derives_icb_code_from_icb_code4(self, mock_graphql):
        mock_graphql.return_value = {
            "CompaniesListingInfo": [
                {
                    "ticker": "HPG",
                    "icbName2": "Materials",
                    "icbName3": "Steel",
                    "icbCode4": "55101010",
                }
            ]
        }

        df = vci_provider.fetch_vn_industry_metadata()

        assert "icb_code" in df.columns
        assert df.loc[0, "icb_code"] == "55101010"


@pytest.mark.unit
class TestRequestGovernanceAndDateParsing:
    @patch("dags.etl_modules.vci_provider.urlopen")
    @patch("dags.etl_modules.vci_provider.governed_call")
    def test_request_json_routes_calls_through_governor(
        self, mock_governed_call, mock_urlopen
    ):
        response = MagicMock()
        response.read.return_value = b'{"ok": true}'
        mock_urlopen.return_value.__enter__.return_value = response
        mock_governed_call.side_effect = (
            lambda source, request_fn, **kwargs: request_fn()
        )

        payload = vci_provider._request_json("GET", "https://example.com/test")

        assert payload == {"ok": True}
        assert mock_governed_call.call_count == 1
        assert mock_governed_call.call_args.args[0] == "vci"

    @patch("dags.etl_modules.vci_provider._graphql")
    def test_fetch_company_events_parses_mixed_date_formats(self, mock_graphql):
        mock_graphql.return_value = {
            "OrganizationEvents": [
                {
                    "id": 10,
                    "eventTitle": "Dividend update",
                    "publicDate": "05/04/2026",
                    "issueDate": "/Date(1778173200000)/",
                    "exrightDate": 1775494800000,
                    "eventListCode": "CASH",
                    "eventListName": "Cash dividend",
                }
            ]
        }

        df = vci_provider.fetch_company_events("VCI")

        assert len(df) == 1
        assert str(df.loc[0, "public_date"]) == "2026-04-05"
        assert str(df.loc[0, "event_date"]) == "2026-05-07"
        assert str(df.loc[0, "exright_date"]) == "2026-04-06"


@pytest.mark.unit
class TestCompanyTypeFallback:
    @patch("dags.etl_modules.vci_provider._graphql")
    def test_company_type_defaults_to_ct_when_lookup_fails(self, mock_graphql):
        mock_graphql.side_effect = RuntimeError("upstream timeout")

        result = vci_provider._company_type_code("DPR")

        assert result == "CT"
