from datetime import datetime
from itertools import islice
from typing import Any

import pandas as pd

from dags.etl_modules.errors import ConfigurationError


def chunk_assets(
    assets: list[dict[str, Any]],
    chunk_size: int,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ConfigurationError(f"chunk_size must be > 0, got {chunk_size}")

    chunks: list[dict[str, Any]] = []
    iterator = iter(assets)
    chunk_index = 1
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return chunks
        chunks.append({"chunk_index": chunk_index, "assets": chunk})
        chunk_index += 1


def parse_date_value(value: Any):
    if value in (None, "", "NaT", "nan"):
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, datetime):
        return value.date()
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()
