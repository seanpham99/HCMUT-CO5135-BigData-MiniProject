import pytest

from dags.etl_modules.contracts import providers


class _FinanceProvider:
    def list_assets(self):
        return [{"symbol": "AAA", "asset_id": "1"}]

    def fetch_income_statement(self, symbol, asset_id):
        return None

    def fetch_balance_sheet(self, symbol, asset_id):
        return None


class _NotificationSender:
    def is_configured(self):
        return True

    def send_message(self, *, text, parse_mode="Markdown"):
        return True


class _MarketDataWriter:
    def upsert_rows(
        self,
        *,
        db_url,
        query,
        rows,
        table_name,
        batch_size,
    ):
        return [], None


class _NewsSummarizer:
    def summarize(self, news_data):
        return "summary"


@pytest.mark.unit
def test_provider_protocols_are_runtime_checkable():
    assert isinstance(_FinanceProvider(), providers.FundamentalsFinanceProvider)
    assert isinstance(_MarketDataWriter(), providers.MarketDataWriter)
    assert isinstance(_NotificationSender(), providers.NotificationSender)
    assert isinstance(_NewsSummarizer(), providers.NewsSummaryProvider)
