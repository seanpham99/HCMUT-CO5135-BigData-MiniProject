from __future__ import annotations

import json
import math
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Iterable

import pandas as pd
import requests

from dags.etl_modules.request_governor import governed_call

BASE_URL = "https://finance.vietstock.vn"
EVENTS_PAGE_PATH = "/lich-su-kien.htm"
EVENT_TYPE_PATH = "/data/eventtypebyid"
EVENTS_TYPEDATA_PATH = "/data/eventstypedata"
ARTICLE_PATH = "/data/GetArticle"

DEFAULT_EVENT_TYPE_ID = 1
DEFAULT_CHANNEL_IDS = (13, 14, 15, 16)

_TOKEN_PATTERN = re.compile(
    r'name=(?:"|)?__RequestVerificationToken(?:"|)?[^>]*value=(?:"|)?([^\s">]+)'
)
_DOTNET_DATE_PATTERN = re.compile(r"/Date\((\d+)(?:[+-]\d+)?\)/")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}


def _extract_token(html: str) -> str:
    match = _TOKEN_PATTERN.search(html or "")
    if not match:
        raise RuntimeError(
            "Could not extract __RequestVerificationToken from page HTML"
        )
    return match.group(1)


def _parse_dotnet_date(value: Any) -> date | None:
    if value in (None, "", "null"):
        return None
    text = str(value)
    match = _DOTNET_DATE_PATTERN.search(text)
    if not match:
        return None
    timestamp_ms = int(match.group(1))
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).date()


def _strip_html(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = _HTML_TAG_PATTERN.sub(" ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _safe_json_loads(raw_text: str) -> Any:
    text = (raw_text or "").lstrip("\ufeff").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unexpected response payload: {text[:200]}") from exc


def _normalize_rows(
    rows: list[dict[str, Any]],
    *,
    channel_id: int,
    channel_name: str | None,
    event_type_id: int,
    article_details: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        article_id = row.get("ArticleID")
        detail = article_details.get(int(article_id), {}) if article_id else {}
        normalized.append(
            {
                "event_id": row.get("EventID"),
                "article_id": article_id,
                "code": row.get("Code"),
                "company_name": row.get("CompanyName"),
                "cat_id": row.get("CatID"),
                "exchange": row.get("Exchange"),
                "gdkhq_date": _parse_dotnet_date(row.get("GDKHQDate")),
                "ndkcc_date": _parse_dotnet_date(row.get("NDKCCDate")),
                "event_date": _parse_dotnet_date(row.get("Time")),
                "date_order": _parse_dotnet_date(row.get("DateOrder")),
                "note": row.get("Note"),
                "title": row.get("Title"),
                "content": _strip_html(row.get("Content")),
                "file_url": row.get("FileUrl"),
                "rate_type_id": row.get("RateTypeID"),
                "rate": row.get("Rate"),
                "volume_publishing": row.get("VolumePublishing"),
                "row_num": row.get("Row"),
                "channel_id": channel_id,
                "channel_name": channel_name,
                "event_type_id": event_type_id,
                "article_title": detail.get("Title"),
                "article_head": detail.get("Head"),
                "article_content": _strip_html(detail.get("Content")),
                "article_publish_time": detail.get("PublishTime"),
                "article_time_string": detail.get("TimeString"),
            }
        )
    return normalized


def _parse_channel_name_map(payload: Any) -> dict[int, str]:
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[1], list)
    ):
        return {}
    mapping: dict[int, str] = {}
    for item in payload[1]:
        if not isinstance(item, dict):
            continue
        channel_id = item.get("ChannelID")
        if channel_id is None:
            continue
        mapping[int(channel_id)] = (
            item.get("NameEn") or item.get("Name") or str(channel_id)
        )
    return mapping


def _request_headers(*, page_url: str) -> dict[str, str]:
    return {
        **_BROWSER_HEADERS,
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Origin": BASE_URL,
        "Referer": page_url,
        "X-Requested-With": "XMLHttpRequest",
    }


def _extract_total_pages(payload: Any, page_size: int) -> int | None:
    if (
        not isinstance(payload, list)
        or len(payload) < 2
        or not isinstance(payload[1], list)
    ):
        return None
    if not payload[1]:
        return None
    raw_total = payload[1][0]
    if isinstance(raw_total, list):
        raw_total = raw_total[0] if raw_total else 0
    if raw_total in (None, ""):
        return None
    total = int(raw_total)
    if total <= 0:
        return 1
    return max(1, math.ceil(total / page_size))


def _governed_request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: dict[str, Any] | None = None,
    timeout: int = 30,
) -> requests.Response:
    def _execute() -> requests.Response:
        if method.upper() == "GET":
            response = session.get(url, headers=headers, timeout=timeout)
        else:
            response = session.post(
                url, data=data or {}, headers=headers, timeout=timeout
            )
        response.raise_for_status()
        return response

    return governed_call(
        "vietstock",
        _execute,
        retry_profile="balanced",
        operation=f"{method.upper()} {url}",
    )


def _event_calendar_page_url(*, channel_id: int, from_date: str, to_date: str) -> str:
    return (
        f"{BASE_URL}{EVENTS_PAGE_PATH}"
        f"?languageid=2&page=1&from={from_date}&to={to_date}&tab={DEFAULT_EVENT_TYPE_ID}"
        f"&group={channel_id}&exchange=1"
    )


def _fetch_article_detail(
    session: requests.Session,
    *,
    headers: dict[str, str],
    article_id: int,
) -> dict[str, Any]:
    response = _governed_request(
        session,
        "POST",
        f"{BASE_URL}{ARTICLE_PATH}",
        headers=headers,
        data={"id": str(article_id)},
    )
    payload = _safe_json_loads(response.text)
    if not isinstance(payload, dict):
        return {}
    return payload


def _parse_rate_to_percentage(rate: Any) -> float:
    if rate in (None, "", "null"):
        return 0.0
    text = str(rate).strip().replace(",", "")
    if ":" in text:
        left, right = text.split(":", 1)
        left_value = float(left)
        right_value = float(right)
        if left_value == 0:
            return 0.0
        return (right_value / left_value) * 100
    return float(text)


def _build_dividends_frame(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()

    dividend_actions = actions[actions["channel_id"].isin([13, 14, 15])].copy()
    if dividend_actions.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, row in dividend_actions.iterrows():
        percent = _parse_rate_to_percentage(row.get("rate"))
        raw_channel_id = row.get("channel_id")
        if raw_channel_id is None:
            continue
        channel_id = int(raw_channel_id)
        exercise_date = row.get("gdkhq_date") or row.get("ndkcc_date")
        cash_percentage = percent if channel_id == 13 else 0.0
        stock_percentage = percent if channel_id in (14, 15) else 0.0
        records.append(
            {
                "exercise_date": exercise_date,
                "cash_year": exercise_date.year
                if isinstance(exercise_date, date)
                else 0,
                "cash_dividend_percentage": cash_percentage,
                "stock_dividend_percentage": stock_percentage,
                "issue_method": row.get("channel_name"),
            }
        )

    return pd.DataFrame(records)


def _build_corporate_events_frame(actions: pd.DataFrame) -> pd.DataFrame:
    if actions.empty:
        return pd.DataFrame()

    frame = pd.DataFrame(
        {
            "event_id": actions["event_id"].astype(str),
            "event_date": actions["event_date"],
            "public_date": actions["ndkcc_date"],
            "exright_date": actions["gdkhq_date"],
            "event_title": actions["title"],
            "event_type": actions["channel_name"].fillna(
                actions["channel_id"].astype(str)
            ),
            "event_description": actions["note"].fillna(actions["content"]),
        }
    )
    return frame


def fetch_vietstock_corporate_actions(
    symbol: str,
    *,
    from_date: str,
    to_date: str,
    channel_ids: Iterable[int] = DEFAULT_CHANNEL_IDS,
    event_type_id: int = DEFAULT_EVENT_TYPE_ID,
    page_size: int = 100,
) -> pd.DataFrame:
    session = requests.Session()
    channel_ids = tuple(int(channel_id) for channel_id in channel_ids)
    if not channel_ids:
        return pd.DataFrame()

    page_url = _event_calendar_page_url(
        channel_id=channel_ids[0],
        from_date=from_date,
        to_date=to_date,
    )
    page_response = _governed_request(
        session,
        "GET",
        page_url,
        headers={
            **_BROWSER_HEADERS,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    token = _extract_token(page_response.text)
    headers = _request_headers(page_url=page_url)

    event_type_response = _governed_request(
        session,
        "POST",
        f"{BASE_URL}{EVENT_TYPE_PATH}",
        headers=headers,
        data={"id": str(event_type_id)},
    )
    channel_name_map = _parse_channel_name_map(
        _safe_json_loads(event_type_response.text)
    )

    all_rows: list[dict[str, Any]] = []
    article_cache: dict[int, dict[str, Any]] = {}
    for channel_id in channel_ids:
        page = 1
        total_pages: int | None = None
        while True:
            payload = {
                "eventTypeID": str(event_type_id),
                "channelID": str(channel_id),
                "code": symbol,
                "catID": "",
                "fDate": from_date,
                "tDate": to_date,
                "page": str(page),
                "pageSize": str(page_size),
                "orderBy": "Date1",
                "orderDir": "DESC",
                "__RequestVerificationToken": token,
            }
            response = _governed_request(
                session,
                "POST",
                f"{BASE_URL}{EVENTS_TYPEDATA_PATH}",
                headers=headers,
                data=payload,
            )
            parsed = _safe_json_loads(response.text)
            if not parsed:
                break
            if (
                not isinstance(parsed, list)
                or not parsed
                or not isinstance(parsed[0], list)
            ):
                raise RuntimeError("Unexpected eventstypedata payload structure")

            page_rows: list[dict[str, Any]] = parsed[0]
            if not page_rows:
                break

            for row in page_rows:
                article_id = row.get("ArticleID")
                if not article_id:
                    continue
                article_id_int = int(article_id)
                if article_id_int in article_cache:
                    continue
                article_cache[article_id_int] = _fetch_article_detail(
                    session,
                    headers=headers,
                    article_id=article_id_int,
                )

            all_rows.extend(
                _normalize_rows(
                    page_rows,
                    channel_id=channel_id,
                    channel_name=channel_name_map.get(channel_id),
                    event_type_id=event_type_id,
                    article_details=article_cache,
                )
            )

            if total_pages is None:
                total_pages = _extract_total_pages(parsed, page_size)
            if total_pages is not None and page >= total_pages:
                break
            if len(page_rows) < page_size:
                break
            page += 1

    if not all_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(all_rows)
    if "code" in frame.columns:
        frame = frame[frame["code"].astype(str).str.upper() == symbol.upper()]
    return frame.reset_index(drop=True)


def fetch_vietstock_corporate_events(
    symbol: str,
    *,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    actions = fetch_vietstock_corporate_actions(
        symbol,
        from_date=from_date,
        to_date=to_date,
        channel_ids=DEFAULT_CHANNEL_IDS,
    )
    return _build_corporate_events_frame(actions)


def fetch_vietstock_dividends(
    symbol: str,
    *,
    from_date: str,
    to_date: str,
) -> pd.DataFrame:
    actions = fetch_vietstock_corporate_actions(
        symbol,
        from_date=from_date,
        to_date=to_date,
        channel_ids=(13, 14, 15),
    )
    return _build_dividends_frame(actions)


def default_date_window() -> tuple[str, str]:
    start = "2000-01-01"
    end = (datetime.now(tz=UTC).date() + timedelta(days=365)).isoformat()
    return start, end
