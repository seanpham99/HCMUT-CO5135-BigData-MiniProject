import logging
import re
import unicodedata

import numpy as np
import pandas as pd
import requests

from dags.etl_modules.request_governor import governed_call

KBS_FINANCE_INFO_URL = (
    "https://kbbuddywts.kbsec.com.vn/sas/kbsv-stock-data-store/stock/finance-info"
)
DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "User-Agent": "Mozilla/5.0",
}
KBS_FINANCE_PAGE_SIZE = 200
KBS_FINANCE_MAX_PAGES = 20

logger = logging.getLogger(__name__)


def _normalize_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    normalized = normalized.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(cleaned.split())


def _contains_keyword(haystack: str, keyword: str) -> bool:
    keyword = _normalize_text(keyword)
    if not keyword:
        return False
    if len(keyword) <= 3:
        return keyword in haystack.split()
    return keyword in haystack


def _period_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if isinstance(col, str) and re.match(r"^\d{4}(?:-Q[1-4])?$", col)
    ]


def _period_to_year_quarter(period: str) -> tuple[int | None, int | None]:
    if not isinstance(period, str) or not period:
        return None, None
    match = re.match(r"^(\d{4})-Q([1-4])$", period)
    if match:
        return int(match.group(1)), int(match.group(2))
    if re.match(r"^\d{4}$", period):
        return int(period), 4
    return None, None


def _quarter_end_date(year: int | None, quarter: int | None) -> str | None:
    if year is None or quarter is None:
        return None
    if quarter == 1:
        return f"{year}-03-31"
    if quarter == 2:
        return f"{year}-06-30"
    if quarter == 3:
        return f"{year}-09-30"
    if quarter == 4:
        return f"{year}-12-31"
    return None


def _build_period_frame(periods: list[str]) -> pd.DataFrame:
    rows = []
    for period in periods:
        year, quarter = _period_to_year_quarter(period)
        rows.append(
            {
                "period_label": period,
                "year": year,
                "quarter": quarter,
                "fiscal_date": _quarter_end_date(year, quarter),
            }
        )
    return pd.DataFrame(rows)


def _build_search_text(row: pd.Series) -> str:
    parts = [
        _normalize_text(row.get("item_id")),
        _normalize_text(row.get("item")),
        _normalize_text(row.get("item_en")),
    ]
    return " ".join(part for part in parts if part)


def _pick_metric_series(
    df: pd.DataFrame,
    period_cols: list[str],
    keyword_groups: list[list[str]],
) -> pd.Series:
    if df.empty:
        return pd.Series(np.nan, index=period_cols)

    best_row = None
    best_score = (-1, -1)

    for _, row in df.iterrows():
        haystack = _build_search_text(row)
        if not haystack:
            continue
        matched_group_idx = None
        for group_idx, keywords in enumerate(keyword_groups):
            if any(_contains_keyword(haystack, keyword) for keyword in keywords):
                matched_group_idx = group_idx
                break
        if matched_group_idx is None:
            continue

        numeric_values = pd.to_numeric(row[period_cols], errors="coerce")
        non_null_count = int(numeric_values.notna().sum())
        score = (len(keyword_groups) - matched_group_idx, non_null_count)
        if score > best_score:
            best_score = score
            best_row = numeric_values

    if best_row is None:
        return pd.Series(np.nan, index=period_cols)

    return best_row


def _period_from_input(period: str) -> str:
    return "year" if str(period or "").upper() == "Y" else "quarter"


def _to_item_id(item: object, item_en: object) -> str:
    candidate = str(item_en or "").strip() or str(item or "").strip()
    normalized = _normalize_text(candidate)
    return normalized.replace(" ", "_") if normalized else ""


def _parse_period_labels(head_list: list[dict]) -> list[str]:
    periods: list[str] = []
    for head in head_list:
        if not isinstance(head, dict):
            continue
        year = str(head.get("YearPeriod") or "").strip()
        term_name = str(head.get("TermName") or "").strip()
        if not year:
            continue
        if "Quý" in term_name:
            quarter_num = term_name.replace("Quý", "").strip()
            periods.append(f"{year}-Q{quarter_num}")
        else:
            periods.append(year)
    return periods


def _records_to_frame(records: list[dict], periods: list[str]) -> pd.DataFrame:
    rows = []
    for record in records:
        if not isinstance(record, dict):
            continue
        row = {
            "item": record.get("Name", ""),
            "item_en": record.get("NameEn", ""),
            "item_id": _to_item_id(record.get("Name", ""), record.get("NameEn", "")),
        }
        for i, period_label in enumerate(periods, 1):
            value = record.get(f"Value{i}")
            row[period_label] = pd.to_numeric(value, errors="coerce")
        rows.append(row)

    return pd.DataFrame(rows)


def _payload_data(payload: object) -> dict:
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        return payload["data"]
    return payload if isinstance(payload, dict) else {}


def _content_record_count(content: object) -> int:
    if not isinstance(content, dict):
        return 0
    total = 0
    for value in content.values():
        if isinstance(value, list):
            total += len(value)
    return total


def _content_signature(content: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(content, dict):
        return ()
    signature_rows: list[tuple[str, int]] = []
    for key in sorted(content.keys()):
        value = content.get(key)
        if isinstance(value, list):
            signature_rows.append((str(key), len(value)))
    return tuple(signature_rows)


def _merge_content(merged: dict[str, list], incoming: object) -> dict[str, list]:
    if not isinstance(incoming, dict):
        return merged
    for key, value in incoming.items():
        if not isinstance(value, list):
            continue
        merged.setdefault(str(key), []).extend(value)
    return merged


def _contains_any_keyword(haystack: str, keyword_groups: list[list[str]]) -> bool:
    for group in keyword_groups:
        if any(_contains_keyword(haystack, keyword) for keyword in group):
            return True
    return False


def _collect_unmapped_metric_labels(
    df: pd.DataFrame,
    metric_map: dict[str, list[list[str]]],
) -> list[str]:
    if df.empty:
        return []

    keyword_groups: list[list[str]] = []
    for groups in metric_map.values():
        keyword_groups.extend(groups)

    unmapped_labels: list[str] = []
    for _, row in df.iterrows():
        haystack = _build_search_text(row)
        if not haystack:
            continue
        if _contains_any_keyword(haystack, keyword_groups):
            continue
        label = str(row.get("item_en") or row.get("item") or "").strip()
        if label:
            unmapped_labels.append(label)

    return unmapped_labels


def _fetch_kbs_financial_data(symbol: str, report_type: str, period: str) -> dict:
    period_type = 1 if _period_from_input(period) == "year" else 2
    merged_head: list[dict] = []
    merged_content: dict[str, list] = {}
    last_signature: tuple[tuple[str, int], ...] | None = None

    for page in range(1, KBS_FINANCE_MAX_PAGES + 1):
        params = {
            "page": page,
            "pageSize": KBS_FINANCE_PAGE_SIZE,
            "type": report_type,
            "unit": 1000,
            "termtype": period_type,
        }
        if report_type != "LCTT":
            params["languageid"] = 1
        else:
            params["code"] = symbol.upper()
            params["termType"] = period_type

        def _request_page():
            response = requests.get(
                f"{KBS_FINANCE_INFO_URL}/{symbol.upper()}",
                headers=DEFAULT_HEADERS,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        try:
            payload = governed_call(
                "kbs",
                _request_page,
                operation=f"kbs_finance_{report_type.lower()}_page_{page}",
            )
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code is None and "404" in str(exc):
                status_code = 404
            if status_code == 404:
                logger.info(
                    "KBS finance-info returned 404 for %s (%s); treating as unavailable",
                    symbol.upper(),
                    report_type,
                )
                break
            raise
        payload_data = _payload_data(payload)
        if not merged_head:
            raw_head = payload_data.get("Head") or []
            if isinstance(raw_head, list):
                merged_head = raw_head

        content = payload_data.get("Content") or {}
        current_count = _content_record_count(content)
        if current_count <= 0:
            break

        current_signature = _content_signature(content)
        if last_signature is not None and current_signature == last_signature:
            break

        _merge_content(merged_content, content)
        last_signature = current_signature

        if current_count < KBS_FINANCE_PAGE_SIZE:
            break
    else:
        logger.warning(
            "KBS pagination reached max pages without a short page for %s (%s)",
            symbol.upper(),
            report_type,
        )

    return {
        "Head": merged_head,
        "Content": merged_content,
    }


def _extract_finance_frame(symbol: str, report: str, period: str) -> pd.DataFrame:
    if report == "income_statement":
        payload = _fetch_kbs_financial_data(symbol, "KQKD", period)
        report_records = (payload.get("Content") or {}).get("Kết quả kinh doanh") or []
    elif report == "balance_sheet":
        payload = _fetch_kbs_financial_data(symbol, "CDKT", period)
        report_records = (payload.get("Content") or {}).get("Cân đối kế toán") or []
    elif report == "ratio":
        payload = _fetch_kbs_financial_data(symbol, "CSTC", period)
        ratio_content = payload.get("Content") or {}
        report_records = []
        for group_records in ratio_content.values():
            if isinstance(group_records, list):
                report_records.extend(group_records)
    else:
        raise ValueError(f"Unsupported KBS finance report type: {report}")

    head_list = payload.get("Head") or []
    periods = _parse_period_labels(head_list)
    if not report_records or not periods:
        return pd.DataFrame()

    df = _records_to_frame(report_records, periods)
    if df.empty:
        return pd.DataFrame()

    return df


def fetch_income_statement(symbol: str, period: str = "Q") -> pd.DataFrame:
    df = _extract_finance_frame(symbol, "income_statement", period)
    if df.empty:
        return pd.DataFrame()

    period_cols = _period_columns(df)
    period_frame = _build_period_frame(period_cols)
    metric_map = {
        "revenue": [["net sales"], ["revenue"], ["doanh thu"]],
        "cost_of_goods_sold": [["cost of sales"], ["cost of goods sold"], ["gia von"]],
        "gross_profit": [["gross profit"], ["loi nhuan gop"]],
        "operating_profit": [["operating profit"], ["loi nhuan hoat dong"]],
        "net_profit_post_tax": [
            ["net profit for the year"],
            ["profit after tax"],
            ["loi nhuan sau thue"],
        ],
        "selling_expenses": [
            ["selling expenses"],
            ["selling expense"],
            ["chi phi ban hang"],
        ],
        "admin_expenses": [["general admin"], ["admin expense"], ["chi phi quan ly"]],
        "financial_income": [["financial income"], ["thu nhap tai chinh"]],
        "financial_expenses": [
            ["financial expense"],
            ["finance expenses"],
            ["chi phi tai chinh"],
        ],
        "other_income": [["other income"], ["thu nhap khac"]],
        "other_expenses": [["other expense"], ["chi phi khac"]],
        "ebitda": [["ebitda"]],
    }

    for metric, keyword_groups in metric_map.items():
        values = _pick_metric_series(df, period_cols, keyword_groups)
        period_frame[metric] = pd.to_numeric(values.values, errors="coerce")

    period_frame = period_frame.rename(
        columns={"year": "yearReport", "quarter": "lengthReport"}
    )
    return period_frame.drop(columns=["period_label"])


def fetch_balance_sheet(symbol: str, period: str = "Q") -> pd.DataFrame:
    df = _extract_finance_frame(symbol, "balance_sheet", period)
    if df.empty:
        return pd.DataFrame()

    period_cols = _period_columns(df)
    period_frame = _build_period_frame(period_cols)
    metric_map = {
        "total_assets": [["total assets"], ["tong tai san"]],
        "total_liabilities": [
            ["total liabilities"],
            ["tong no phai tra"],
            ["no phai tra"],
        ],
        "total_equity": [["owner s equity"], ["equity"], ["von chu so huu"]],
        "cash_and_equivalents": [
            ["cash and cash equivalents"],
            ["cash and equivalents"],
            ["tien va cac khoan tuong duong tien"],
        ],
        "short_term_assets": [
            ["current assets"],
            ["short term assets"],
            ["tai san ngan han"],
        ],
        "long_term_assets": [
            ["non current assets"],
            ["long term assets"],
            ["tai san dai han"],
        ],
        "short_term_liabilities": [
            ["current liabilities"],
            ["short term liabilities"],
            ["no ngan han"],
        ],
        "long_term_liabilities": [["long term liabilities"], ["no dai han"]],
    }

    for metric, keyword_groups in metric_map.items():
        values = _pick_metric_series(df, period_cols, keyword_groups)
        period_frame[metric] = pd.to_numeric(values.values, errors="coerce")

    period_frame = period_frame.rename(
        columns={"year": "yearReport", "quarter": "lengthReport"}
    )
    return period_frame.drop(columns=["period_label"])


def fetch_financial_ratios(symbol: str, period: str = "Q") -> pd.DataFrame:
    df = _extract_finance_frame(symbol, "ratio", period)
    if df.empty:
        return pd.DataFrame()

    period_cols = _period_columns(df)
    period_frame = _build_period_frame(period_cols)
    metric_map = {
        "pe_ratio": [["pe"], ["price earnings"]],
        "pb_ratio": [["pb"], ["price book"]],
        "ps_ratio": [["ps"], ["price sales"]],
        "p_cashflow_ratio": [["p cash flow"], ["p cf"], ["pcf"]],
        "eps": [["eps"], ["earnings per share"]],
        "bvps": [["book value per share"], ["bvps"]],
        "market_cap": [
            ["market cap"],
            ["market capital"],
            ["market capitalization"],
            ["von hoa"],
        ],
        "roe": [["roe"], ["return on equity"]],
        "roa": [["roa"], ["return on assets"]],
        "roic": [["roic"], ["return on invested capital"]],
        "financial_leverage": [["financial leverage"], ["don bay tai chinh"]],
        "dividend_yield": [["dividend yield"], ["ty suat co tuc"]],
        "gross_margin": [["gross margin"], ["bien loi nhuan gop"]],
        "operating_margin": [["operating margin"], ["bien loi nhuan hoat dong"]],
        "net_profit_margin": [
            ["net profit margin"],
            ["net margin"],
            ["bien loi nhuan rong"],
        ],
        "interest_coverage": [["interest coverage"], ["kha nang tra lai"]],
        "asset_turnover": [["asset turnover"], ["vong quay tai san"]],
        "inventory_turnover": [["inventory turnover"], ["vong quay hang ton kho"]],
        "receivable_turnover": [
            ["receivable turnover"],
            ["accounts receivable turnover"],
            ["vong quay khoan phai thu"],
        ],
        "revenue_growth": [["revenue growth"], ["tang truong doanh thu"]],
        "profit_growth": [
            ["profit growth"],
            ["tang truong loi nhuan"],
            ["net profit growth"],
        ],
        "current_ratio": [["current ratio"], ["ty le thanh toan hien hanh"]],
        "quick_ratio": [["quick ratio"], ["ty le thanh toan nhanh"]],
        "debt_to_equity": [["debt to equity"], ["ty le no von"]],
        "free_cash_flow": [["free cash flow"], ["dong tien tu do"]],
    }

    unmapped_metric_labels = _collect_unmapped_metric_labels(df, metric_map)
    if unmapped_metric_labels:
        preview = ", ".join(unmapped_metric_labels[:5])
        logger.warning(
            "KBS ratio mapping has %s unmapped metrics for %s (sample: %s)",
            len(unmapped_metric_labels),
            symbol.upper(),
            preview,
        )

    for metric, keyword_groups in metric_map.items():
        values = _pick_metric_series(df, period_cols, keyword_groups)
        period_frame[metric] = pd.to_numeric(values.values, errors="coerce")

    period_frame = period_frame.rename(
        columns={"year": "yearReport", "quarter": "lengthReport"}
    )
    return period_frame.drop(columns=["period_label"])
