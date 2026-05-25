from __future__ import annotations

from typing import Any, Protocol, TypeAlias, TypedDict, runtime_checkable

import pandas as pd

AssetRecord: TypeAlias = dict[str, str]
Record: TypeAlias = dict[str, Any]


class FailureRecord(TypedDict):
    symbol: str
    error: str


class ExtractPayload(TypedDict):
    records: list[Record]
    failed_symbols: list[FailureRecord]


class LoadSummary(TypedDict):
    records_input: int
    rows_prepared: int
    rows_loaded: int
    failed_symbols: list[FailureRecord]
    failed_rows: list[FailureRecord]
    failed_batches: list[dict[str, object]]


@runtime_checkable
class FundamentalsFinanceProvider(Protocol):
    def list_assets(self) -> list[AssetRecord]: ...

    def fetch_income_statement(self, symbol: str, asset_id: str) -> pd.DataFrame: ...

    def fetch_balance_sheet(self, symbol: str, asset_id: str) -> pd.DataFrame: ...


@runtime_checkable
class PriceDataProvider(Protocol):
    def list_assets(self) -> list[AssetRecord]: ...

    def fetch_prices(
        self,
        symbol: str,
        asset_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame: ...


@runtime_checkable
class RatioDataProvider(Protocol):
    def list_assets(self) -> list[AssetRecord]: ...

    def fetch_ratios(self, symbol: str, asset_id: str) -> pd.DataFrame: ...


@runtime_checkable
class MarketDataWriter(Protocol):
    def upsert_rows(
        self,
        *,
        db_url: str | None,
        query: str,
        rows: list[tuple[object, ...]],
        table_name: str,
        batch_size: int,
    ) -> tuple[list[dict[str, object]], str | None]: ...


@runtime_checkable
class NotificationSender(Protocol):
    def is_configured(self) -> bool: ...

    def send_message(self, *, text: str, parse_mode: str = "Markdown") -> bool: ...


@runtime_checkable
class NewsSummaryProvider(Protocol):
    def summarize(self, news_data: list[Record]) -> str | None: ...
