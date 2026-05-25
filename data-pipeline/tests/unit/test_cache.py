"""Unit tests for dags.etl_modules.cache."""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


def load_cache_module(module_name: str = "cache_test_module"):
    cache_path = (
        Path(__file__).resolve().parents[2] / "dags" / "etl_modules" / "cache.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, cache_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_import_does_not_connect_to_redis():
    with patch("redis.from_url") as mock_from_url, patch("redis.Redis") as mock_redis:
        module = load_cache_module()

    assert not mock_from_url.called
    assert not mock_redis.called
    assert module.redis_client is module._REDIS_CLIENT_UNINITIALIZED


def test_get_cache_key_is_deterministic_for_kwargs_order():
    module = load_cache_module("cache_key_module")

    key_one = module.get_cache_key(
        "fetch_stock_price", ("HPG",), {"end": "2024-01-10", "start": "2024-01-01"}
    )
    key_two = module.get_cache_key(
        "fetch_stock_price", ("HPG",), {"start": "2024-01-01", "end": "2024-01-10"}
    )

    assert key_one == key_two


def test_cached_data_lazily_initializes_and_round_trips_dataframe():
    module = load_cache_module("cache_round_trip_module")
    module.redis_client = module._REDIS_CLIENT_UNINITIALIZED

    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.setex.return_value = True
    mock_client.ping.return_value = True

    calls = {"count": 0}

    def create_client():
        return mock_client

    @module.cached_data(ttl_seconds=60)
    def build_frame(symbol):
        calls["count"] += 1
        return pd.DataFrame(
            {
                "trading_date": [date(2024, 1, 1)],
                "ticker": [symbol],
                "close": [123.45],
            }
        )

    module._create_redis_client = create_client

    first_result = build_frame("HPG")
    assert calls["count"] == 1
    assert mock_client.get.called
    assert mock_client.setex.called

    cached_payload = mock_client.setex.call_args.args[2]
    decoded_payload = json.loads(cached_payload.decode("utf-8"))
    assert decoded_payload["__cache_kind__"] == "dataframe"
    assert decoded_payload["records"][0]["trading_date"]["__cache_kind__"] == "date"

    mock_client.get.return_value = cached_payload
    second_result = build_frame("HPG")

    assert calls["count"] == 1
    assert isinstance(first_result, pd.DataFrame)
    assert isinstance(second_result, pd.DataFrame)
    assert second_result.equals(first_result)
    assert isinstance(second_result["trading_date"].iloc[0], date)


def test_cached_data_key_fn_overrides_default_argument_keying():
    module = load_cache_module("cache_key_fn_module")
    module.redis_client = module._REDIS_CLIENT_UNINITIALIZED

    mock_client = Mock()
    mock_client.get.return_value = None
    mock_client.setex.return_value = True
    mock_client.ping.return_value = True
    module._create_redis_client = lambda: mock_client

    normalize_inputs = []
    original_normalize = module._normalize_for_key

    def tracking_normalize(value):
        normalize_inputs.append(value)
        return original_normalize(value)

    module._normalize_for_key = tracking_normalize

    @module.cached_data(
        ttl_seconds=60, key_fn=lambda *args, **kwargs: "explicit-cache-key"
    )
    def echo(value):
        return value

    result = echo("HPG")

    assert result == "HPG"
    assert "HPG" not in normalize_inputs
    assert "explicit-cache-key" in normalize_inputs
