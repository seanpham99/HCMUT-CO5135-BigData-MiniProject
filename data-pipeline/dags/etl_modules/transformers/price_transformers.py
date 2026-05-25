from typing import Any

import pandas as pd

from dags.etl_modules.transformers.shared_transformers import parse_date_value


def records_from_frames(
    price_frames: list[pd.DataFrame],
    filter_from: str,
) -> list[dict[str, Any]]:
    filtered_frames: list[pd.DataFrame] = []
    for frame in price_frames:
        if frame is None or frame.empty:
            continue
        if "trading_date" in frame.columns:
            frame = frame[frame["trading_date"].astype(str) >= filter_from]
        if not frame.empty:
            filtered_frames.append(frame)

    if not filtered_frames:
        return []

    return pd.concat(filtered_frames, ignore_index=True).to_dict("records")


def convert_records_to_rows(
    records: list[dict[str, Any]],
    price_columns: tuple[str, ...],
) -> tuple[list[tuple[Any, ...]], list[dict[str, str]]]:
    rows: list[tuple[Any, ...]] = []
    failed_rows: list[dict[str, str]] = []

    for row in records:
        asset_id = row.get("asset_id")
        symbol = row.get("symbol") or row.get("ticker")
        try:
            row_values = dict(row)
            row_values["trading_date"] = parse_date_value(row.get("trading_date"))
            rows.append(tuple(row_values.get(col) for col in price_columns))
        except Exception as exc:
            failed_rows.append(
                {
                    "symbol": str(symbol or "unknown"),
                    "asset_id": str(asset_id or "unknown"),
                    "error": f"price row conversion failed: {exc}",
                }
            )
    return rows, failed_rows
