"""Direct VCI data access helpers used by the data-pipeline DAGs.

This module replaces the vnstock runtime dependency for the split EOD market
data DAGs by calling the Vietcap endpoints directly and shaping the payloads
into the same tabular contracts the loaders already expect.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, timedelta, timezone
from io import StringIO
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

from dags.etl_modules.cache import get_redis_client
from dags.etl_modules.request_governor import governed_call

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://trading.vietcap.com.vn/data-mt/graphql"
TRADING_URL = "https://trading.vietcap.com.vn/api/"
OHLC_PATH = "chart/OHLCChart/gap-chart"
NEWS_LANG = "vi"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "content-type": "application/json",
    "origin": "https://trading.vietcap.com.vn",
    "referer": "https://trading.vietcap.com.vn/",
    "user-agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
}

_FALLBACK_VN_TICKERS = ["HPG", "VCB", "VNM", "FPT", "MWG", "VIC"]
_DOTNET_DATE_PATTERN = re.compile(r"/Date\((\d+)(?:[+-]\d+)?\)/")
_METADATA_REDIS_KEY = "vci:financial_ratio_metadata"
_METADATA_TTL_SECONDS = 3600


def _camel_to_snake(name: str) -> str:
    value = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", str(name))
    value = re.sub("([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.replace("__", "_").lower()


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: int = 30,
    operation: str | None = None,
) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=DEFAULT_HEADERS, method=method)

    def _execute() -> Any:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return governed_call(
            "vci",
            _execute,
            retry_profile="balanced",
            operation=operation or f"{method} {url}",
        )
    except HTTPError as exc:
        raise RuntimeError(
            f"HTTP {exc.code} from {url}: {exc.read().decode('utf-8', 'ignore')}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc}") from exc


def _graphql(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    operation: str | None = None,
) -> dict[str, Any]:
    response = _request_json(
        "POST",
        GRAPHQL_URL,
        payload={"query": query, "variables": variables or {}},
        operation=operation,
    )
    if isinstance(response, dict) and response.get("errors"):
        raise RuntimeError(f"VCI GraphQL error: {response['errors']}")
    return response.get("data", {}) if isinstance(response, dict) else {}


def _company_type_code(symbol: str) -> str:
    query = """
    query Query($ticker: String!) {
      CompanyListingInfo(ticker: $ticker) {
        icbName4
      }
    }
    """
    try:
        data = _graphql(
            query,
            {"ticker": symbol},
            operation=f"POST {GRAPHQL_URL} CompanyListingInfo ticker={symbol}",
        )
    except RuntimeError as exc:
        logger.warning(
            "Falling back to company type CT for %s after lookup failure: %s",
            symbol,
            exc,
        )
        return "CT"
    listing_info = data.get("CompanyListingInfo") or {}
    icb_name = str(listing_info.get("icbName4") or "").strip().lower()

    if "ngân hàng" in icb_name:
        return "NH"
    if "môi giới chứng khoán" in icb_name or "chứng khoán" in icb_name:
        return "CK"
    if "bảo hiểm" in icb_name:
        return "BH"
    return "CT"


def _financial_ratio_metadata() -> pd.DataFrame:
    redis_client: Any = get_redis_client()
    if redis_client is not None:
        try:
            cached = redis_client.get(_METADATA_REDIS_KEY)
            if cached:
                cached_text = (
                    cached.decode("utf-8")
                    if isinstance(cached, (bytes, bytearray))
                    else str(cached)
                )
                return pd.read_json(StringIO(cached_text), orient="records")
        except Exception as exc:
            logger.warning("Failed to load cached financial ratio metadata: %s", exc)

    query = """
    query Query {
      ListFinancialRatio {
        id
        type
        name
        unit
        isDefault
        fieldName
        en_Type
        en_Name
        tagName
        comTypeCode
        order
      }
    }
    """
    data = _graphql(query, operation=f"POST {GRAPHQL_URL} ListFinancialRatio")
    rows = data.get("ListFinancialRatio") or []
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.columns = [_camel_to_snake(column) for column in frame.columns]

    if redis_client is not None:
        try:
            redis_client.set(
                _METADATA_REDIS_KEY,
                frame.to_json(orient="records"),
                ex=_METADATA_TTL_SECONDS,
            )
        except Exception as exc:
            logger.warning("Failed to cache financial ratio metadata: %s", exc)

    return frame


def _build_financial_frame(
    symbol: str,
    period: str,
    *,
    report_types: set[str] | None = None,
) -> pd.DataFrame:
    metadata = _financial_ratio_metadata()
    if metadata.empty:
        return pd.DataFrame()

    if report_types:
        normalized_types = {
            str(report_type).strip().lower()
            for report_type in report_types
            if str(report_type).strip()
        }
        if "type" not in metadata.columns:
            return pd.DataFrame()
        metadata = metadata[
            metadata["type"].astype(str).str.strip().str.lower().isin(normalized_types)
        ].copy()
        if metadata.empty:
            return pd.DataFrame()

    company_type = _company_type_code(symbol)
    if company_type != "CT":
        metadata = metadata[metadata["com_type_code"].isin(["CT", company_type])].copy()
    else:
        metadata = metadata[metadata["com_type_code"] == "CT"].copy()

    metadata = metadata.drop_duplicates(subset=["field_name"], keep="first")
    field_names = [str(field) for field in metadata["field_name"].tolist() if field]
    if not field_names:
        return pd.DataFrame()

    selection = "\n          ".join(
        ["ticker", "yearReport", "lengthReport", "updateDate", *field_names]
    )
    query = f"""
    query Query($ticker: String!, $period: String!) {{
      CompanyFinancialRatio(ticker: $ticker, period: $period) {{
        ratio {{
          {selection}
        }}
        period
      }}
    }}
    """
    data = _graphql(
        query,
        {"ticker": symbol, "period": period},
        operation=(
            f"POST {GRAPHQL_URL} CompanyFinancialRatio ticker={symbol} period={period}"
        ),
    )
    ratio_rows = ((data.get("CompanyFinancialRatio") or {}).get("ratio")) or []
    frame = pd.DataFrame(ratio_rows)
    if frame.empty:
        return frame

    translation = metadata.set_index("field_name")[["en_name", "name"]].fillna("")
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        if column in translation.index:
            label = (
                translation.loc[column, "en_name"]
                or translation.loc[column, "name"]
                or column
            )
            rename_map[column] = str(label)
    frame = frame.rename(columns=rename_map)

    ordered_columns: list[str] = [
        column
        for column in ["ticker", "yearReport", "lengthReport", "updateDate"]
        if column in frame.columns
    ]
    ordered_columns.extend(
        [
            str(rename_map.get(field_name, field_name))
            for field_name in metadata.sort_values(["order", "field_name"])[
                "field_name"
            ].tolist()
            if str(rename_map.get(field_name, field_name)) in frame.columns
        ]
    )
    ordered_columns = list(dict.fromkeys(ordered_columns))
    return frame[ordered_columns]


def _find_columns(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = list(columns)
    normalized = [str(column).lower() for column in column_list]
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for index, column in enumerate(normalized):
            if candidate_lower in column:
                return column_list[index]
    return None


def _extract_source_host(url: object) -> str | None:
    if not isinstance(url, str) or not url.strip():
        return None
    host = urlparse(url.strip()).netloc
    return host or None


def _parse_date_value(value: Any, *, numeric_unit: str = "ms") -> date | None:
    if value in (None, "", "null", "NaT", "nan"):
        return None

    if isinstance(value, (int, float)) and not pd.isna(value):
        ts = int(value)
        if numeric_unit == "ms":
            return datetime.fromtimestamp(ts / 1000, tz=UTC).date()
        return datetime.fromtimestamp(ts, tz=UTC).date()

    text = str(value).strip()
    if not text:
        return None

    dotnet_match = _DOTNET_DATE_PATTERN.search(text)
    if dotnet_match:
        ts_ms = int(dotnet_match.group(1))
        return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).date()

    if text.isdigit():
        ts = int(text)
        if numeric_unit == "ms":
            return datetime.fromtimestamp(ts / 1000, tz=UTC).date()
        return datetime.fromtimestamp(ts, tz=UTC).date()

    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_date_series(values: Iterable[Any], *, numeric_unit: str = "ms") -> pd.Series:
    return pd.Series(
        [_parse_date_value(value, numeric_unit=numeric_unit) for value in values],
        index=getattr(values, "index", None),
    )


def list_active_vn_stock_tickers(
    db_url: str | None, *, raise_on_fallback: bool = False
) -> list[dict[str, str]]:
    if not db_url:
        if raise_on_fallback:
            raise RuntimeError("SUPABASE_DB_URL is not set")
        return [
            {"symbol": ticker, "asset_id": "fallback"}
            for ticker in _FALLBACK_VN_TICKERS
        ]

    try:
        with psycopg2.connect(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, symbol, metadata
                    FROM market_data.assets
                    WHERE asset_class = 'STOCK'
                      AND market = 'VN'
                      AND status = 'active'
                    ORDER BY symbol
                    """
                )
                rows = cursor.fetchall()
    except Exception as exc:
        if raise_on_fallback:
            raise RuntimeError(f"Could not query market_data.assets: {exc}") from exc
        logger.warning("Falling back to seed VN tickers after query failure: %s", exc)
        return [
            {"symbol": ticker, "asset_id": "fallback"}
            for ticker in _FALLBACK_VN_TICKERS
        ]

    tickers: list[dict[str, str]] = []
    seen_symbols: set[str] = set()
    for row in rows:
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen_symbols:
            continue

        metadata = row.get("metadata") or {}
        symbol_type = str(metadata.get("symbol_type") or "").strip().upper()
        if symbol_type and symbol_type != "STOCK":
            continue

        tickers.append({"symbol": symbol, "asset_id": row.get("id") or "fallback"})
        seen_symbols.add(symbol)

    if tickers:
        return tickers

    if raise_on_fallback:
        raise RuntimeError("market_data.assets returned zero active VN stock tickers")
    return [
        {"symbol": ticker, "asset_id": "fallback"} for ticker in _FALLBACK_VN_TICKERS
    ]


def fetch_stock_price(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    end_stamp = int(
        (end_dt + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp()
    )
    business_days = pd.bdate_range(start=start_date, end=end_date)
    count_back = int(len(business_days) + 1)

    payload = {
        "timeFrame": "ONE_DAY",
        "symbols": [symbol],
        "to": end_stamp,
        "countBack": count_back,
    }
    data = _request_json("POST", f"{TRADING_URL}{OHLC_PATH}", payload=payload)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if isinstance(data, list) and data:
        first_item = data[0]
        if isinstance(first_item, dict) and isinstance(first_item.get("o"), list):
            data = pd.DataFrame(
                {
                    "t": first_item.get("t"),
                    "o": first_item.get("o"),
                    "h": first_item.get("h"),
                    "l": first_item.get("l"),
                    "c": first_item.get("c"),
                    "v": first_item.get("v"),
                }
            ).to_dict("records")

    frame = pd.DataFrame(data or [])
    if frame.empty:
        return frame

    rename_map = {
        "t": "trading_date",
        "time": "trading_date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    frame = frame.rename(
        columns={
            column: rename_map[column]
            for column in frame.columns
            if column in rename_map
        }
    )

    required_cols = ["trading_date", "open", "high", "low", "close", "volume"]
    frame = frame[[column for column in required_cols if column in frame.columns]]
    if "trading_date" in frame.columns:
        frame["trading_date"] = _parse_date_series(
            frame["trading_date"], numeric_unit="s"
        )

    frame["ticker"] = symbol
    frame["source"] = "vci"
    for column in ["open", "high", "low", "close"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if "volume" in frame.columns:
        frame["volume"] = (
            pd.to_numeric(frame["volume"], errors="coerce").fillna(0).astype(int)
        )
    return frame


_INTRADAY_TIMEFRAME = "ONE_MINUTE"
# vnstock uses 255 minutes per VN cash session for countBack sizing (quote.py).
_VN_MINUTES_PER_SESSION = 255
_VN_INDEX_SYMBOLS = frozenset(
    {
        "VNINDEX",
        "HNXINDEX",
        "UPCOMINDEX",
        "HNX30",
        "VN30",
        "VN100",
        "VNALL",
        "VNMID",
        "VNSML",
    }
)


def _infer_vci_asset_type(symbol: str) -> str:
    """Mirror vnstock ``get_asset_type`` for gap-chart price scaling."""
    normalized = symbol.strip().upper()
    if normalized in _VN_INDEX_SYMBOLS:
        return "index"
    if len(normalized) == 3:
        return "stock"
    if re.match(r"^VN30F\d{1,2}[MQ]$", normalized) or re.match(
        r"^VN100F\d{1,2}[MQ]$", normalized
    ):
        return "derivative"
    if re.match(r"^4[12][A-Z0-9]{2}[0-9A-HJ-NP-TV-W][1-9A-C]\d{3}$", normalized):
        return "derivative"
    return "stock"


def _intraday_count_back(
    start_dt: date,
    end_dt: date,
    *,
    time_frame: str = _INTRADAY_TIMEFRAME,
) -> int:
    """Compute gap-chart countBack the same way vnstock VCI Quote.history does."""
    start_time = datetime.combine(start_dt, datetime.min.time())
    end_time = datetime.combine(end_dt, datetime.min.time()) + timedelta(days=1)
    business_days = pd.bdate_range(start=start_time, end=end_time)

    if time_frame == "ONE_DAY":
        return int(len(business_days) + 1)
    if time_frame == "ONE_HOUR":
        return int(len(business_days) * 5 + 1)
    if time_frame == "ONE_MINUTE":
        return int(len(business_days) * _VN_MINUTES_PER_SESSION + 1)
    return int(len(business_days) * _VN_MINUTES_PER_SESSION + 1)


def _vector_gap_chart_to_frame(symbol_data: dict[str, Any]) -> pd.DataFrame:
    """Convert VCI vector OHLC payload (t,o,h,l,c,v arrays) to row records."""
    return pd.DataFrame(
        {
            "time": symbol_data.get("t"),
            "open": symbol_data.get("o"),
            "high": symbol_data.get("h"),
            "low": symbol_data.get("l"),
            "close": symbol_data.get("c"),
            "volume": symbol_data.get("v"),
        }
    )


def _transform_vci_intraday_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    asset_type: str,
) -> pd.DataFrame:
    """Apply vnstock VCI post-processing: UTC epoch -> VN tz, stock /1000 prices."""
    if frame.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    result = frame.copy()
    result["time"] = (
        pd.to_datetime(result["time"].astype(int), unit="s", utc=True)
        .dt.tz_convert("Asia/Ho_Chi_Minh")
    )

    for column in ["open", "high", "low", "close"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["volume"] = (
        pd.to_numeric(result["volume"], errors="coerce").fillna(0).astype(int)
    )

    # vnstock ohlc_to_df: equities are stored in VND x1000 in the API payload.
    if asset_type not in ("index", "derivative"):
        result[["open", "high", "low", "close"]] = result[
            ["open", "high", "low", "close"]
        ].div(1000)

    return result.sort_values("time").reset_index(drop=True)


def fetch_vci_intraday_ohlc(
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    time_frame: str = _INTRADAY_TIMEFRAME,
) -> pd.DataFrame:
    """Fetch intraday OHLCV from VCI gap-chart (vnstock-compatible contract).

    Mirrors ``vnstock.explorer.vci.Quote.history`` for the gap-chart path:
    - POST ``chart/OHLCChart/gap-chart`` with ``timeFrame``, ``symbols``, ``to``, ``countBack``
    - ``countBack`` = business_days * 255 + 1 for ``ONE_MINUTE`` (no pagination loop)
    - Vector columns ``t/o/h/l/c/v`` expanded to rows, timestamps UTC -> Asia/Ho_Chi_Minh
    - Stock prices divided by 1000; derivatives/indices unchanged

    See ``project_intraday/.venv/.../vnstock/explorer/vci/quote.py`` for reference.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if start_dt > end_dt:
        raise ValueError(f"start_date {start_date} must be <= end_date {end_date}")

    end_time = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    end_stamp = int(end_time.replace(tzinfo=timezone.utc).timestamp())
    count_back = _intraday_count_back(start_dt, end_dt, time_frame=time_frame)
    asset_type = _infer_vci_asset_type(symbol)

    payload = {
        "timeFrame": time_frame,
        "symbols": [symbol],
        "to": end_stamp,
        "countBack": count_back,
    }
    raw = _request_json(
        "POST",
        f"{TRADING_URL}{OHLC_PATH}",
        payload=payload,
        operation=f"vci_intraday_{time_frame}_{symbol}",
    )

    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list) or not raw:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    symbol_data = raw[0]
    if not isinstance(symbol_data, dict) or not isinstance(symbol_data.get("o"), list):
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    frame = _vector_gap_chart_to_frame(symbol_data)
    frame = _transform_vci_intraday_frame(frame, symbol=symbol, asset_type=asset_type)

    mask = (frame["time"].dt.date >= start_dt) & (frame["time"].dt.date <= end_dt)
    return frame.loc[mask].reset_index(drop=True)


def fetch_financial_ratios(symbol: str, period: str = "Q") -> pd.DataFrame:
    return _build_financial_frame(symbol, period)


def fetch_income_statement(symbol: str, period: str = "Q") -> pd.DataFrame:
    return _build_financial_frame(
        symbol,
        period,
        report_types={"Chỉ tiêu kết quả kinh doanh", "income statement"},
    )


def fetch_balance_sheet(symbol: str, period: str = "Q") -> pd.DataFrame:
    return _build_financial_frame(
        symbol,
        period,
        report_types={"Chỉ tiêu cân đối kế toán", "balance sheet"},
    )


def fetch_company_events(symbol: str) -> pd.DataFrame:
    query = """
    query Query($ticker: String!, $lang: String!) {
      OrganizationEvents(ticker: $ticker) {
        id
        organCode
        ticker
        eventTitle
        en_EventTitle
        publicDate
        issueDate
        sourceUrl
        eventListCode
        ratio
        value
        recordDate
        exrightDate
        eventListName
        en_EventListName
      }
    }
    """
    data = _graphql(query, {"ticker": symbol, "lang": NEWS_LANG})
    rows = data.get("OrganizationEvents") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.rename(
        columns={
            "id": "event_id",
            "eventTitle": "event_title",
            "en_EventTitle": "event_title_en",
            "publicDate": "public_date",
            "issueDate": "event_date",
            "sourceUrl": "source_url",
            "eventListCode": "event_type",
            "eventListName": "event_description",
            "en_EventListName": "event_description_en",
            "recordDate": "record_date",
            "exrightDate": "exright_date",
        }
    )

    if "event_date" not in frame.columns:
        frame["event_date"] = frame.get("record_date")

    for column in ["event_date", "public_date", "exright_date"]:
        if column in frame.columns:
            frame[column] = _parse_date_series(frame[column], numeric_unit="ms")

    if "event_id" in frame.columns:
        frame["event_id"] = frame["event_id"].astype(str)
    else:
        frame["event_id"] = None
    if "event_description_en" in frame.columns:
        fallback_description = (
            frame["event_description"]
            if "event_description" in frame.columns
            else pd.Series([None] * len(frame), index=frame.index)
        )
        frame["event_description"] = frame["event_description_en"].fillna(
            fallback_description
        )
    if "event_type" in frame.columns:
        fallback_event_type = (
            frame["organCode"]
            if "organCode" in frame.columns
            else pd.Series([None] * len(frame), index=frame.index)
        )
        frame["event_type"] = frame["event_type"].fillna(fallback_event_type)
    else:
        frame["event_type"] = (
            frame["organCode"] if "organCode" in frame.columns else None
        )
    required_cols = [
        "event_id",
        "event_date",
        "public_date",
        "exright_date",
        "event_title",
        "event_type",
        "event_description",
    ]
    for column in required_cols:
        if column not in frame.columns:
            frame[column] = None
    return frame[required_cols]


def fetch_company_news(symbol: str) -> pd.DataFrame:
    query = """
    query Query($ticker: String!, $lang: String!) {
      News(ticker: $ticker, langCode: $lang) {
        id
        organCode
        ticker
        newsTitle
        newsSourceLink
        createdAt
        publicDate
        updatedAt
        newsId
        closePrice
        referencePrice
        floorPrice
        ceilingPrice
        percentPriceChange
      }
    }
    """
    data = _graphql(query, {"ticker": symbol, "lang": NEWS_LANG})
    rows = data.get("News") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.rename(
        columns={
            "id": "news_id",
            "newsTitle": "title",
            "newsSourceLink": "news_source_link",
            "publicDate": "publish_date",
            "closePrice": "price_at_publish",
            "referencePrice": "reference_price",
            "percentPriceChange": "price_change_ratio",
        }
    )
    frame["ticker"] = symbol
    if "news_source_link" in frame.columns:
        frame["source"] = frame["news_source_link"].apply(_extract_source_host)
    else:
        frame["source"] = None
    if "price_at_publish" in frame.columns and "reference_price" in frame.columns:
        frame["price_change"] = pd.to_numeric(
            frame["price_at_publish"], errors="coerce"
        ) - pd.to_numeric(frame["reference_price"], errors="coerce")
    frame["publish_date"] = _parse_date_series(frame["publish_date"], numeric_unit="ms")
    if "price_change_ratio" in frame.columns:
        frame["price_change_ratio"] = pd.to_numeric(
            frame["price_change_ratio"], errors="coerce"
        )
    for column in [
        "price_at_publish",
        "price_change",
        "price_change_ratio",
        "rsi",
        "rs",
    ]:
        if column not in frame.columns:
            frame[column] = 0.0
    required_cols = [
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
    return frame[required_cols]


def fetch_company_overview(symbol: str) -> pd.DataFrame:
    query = """
    query Query($ticker: String!) {
      CompanyListingInfo(ticker: $ticker) {
        ticker
        companyProfile
        en_CompanyProfile
        icbName2
        enIcbName2
        icbName3
        enIcbName3
        icbName4
        enIcbName4
        __typename
      }
    }
    """
    try:
        data = _graphql(query, {"ticker": symbol})
    except RuntimeError as exc:
        logger.warning(
            "Falling back to default company type CT for %s after lookup failure: %s",
            symbol,
            exc,
        )
        return pd.DataFrame()
    listing_info = data.get("CompanyListingInfo") or {}
    if not listing_info:
        return pd.DataFrame()

    frame = pd.DataFrame([listing_info])
    frame = frame.rename(
        columns={
            "ticker": "symbol",
            "companyProfile": "company_profile",
            "en_CompanyProfile": "company_profile_en",
            "icbName2": "icb_name2",
            "enIcbName2": "icb_name2_en",
            "icbName3": "icb_name3",
            "enIcbName3": "icb_name3_en",
            "icbName4": "icb_name4",
            "enIcbName4": "icb_name4_en",
        }
    )
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    return frame


def fetch_vn_listing_symbols() -> pd.DataFrame:
    url = "https://trading.vietcap.com.vn/api/price/symbols/getAll"
    data = _request_json("GET", url)
    frame = pd.DataFrame(data or [])
    if frame.empty:
        return frame

    frame.columns = [_camel_to_snake(column) for column in frame.columns]
    frame = frame.rename(columns={"board": "exchange"})
    return frame


def fetch_vn_industry_metadata() -> pd.DataFrame:
    query = """
    query Query {
      CompaniesListingInfo {
        ticker
        organName
        enOrganName
        icbName3
        enIcbName3
        icbName2
        enIcbName2
        icbName4
        enIcbName4
        comTypeCode
        icbCode1
        icbCode2
        icbCode3
        icbCode4
      }
    }
    """
    data = _graphql(query)
    rows = data.get("CompaniesListingInfo") or []
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame.columns = [_camel_to_snake(column) for column in frame.columns]
    frame = frame.rename(columns={"ticker": "symbol"})
    if "icb_code" not in frame.columns:
        icb_code_candidates = [
            column
            for column in ["icb_code4", "icb_code3", "icb_code2", "icb_code1"]
            if column in frame.columns
        ]
        if icb_code_candidates:
            frame["icb_code"] = frame[icb_code_candidates].bfill(axis=1).iloc[:, 0]
        else:
            frame["icb_code"] = None
    return frame
