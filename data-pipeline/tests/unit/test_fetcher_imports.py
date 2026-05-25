"""Regression tests for fetcher module imports."""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_fetcher_package_import_resolves_cache_decorator():
    """Package import resolves cached_data from package cache module."""
    sys.modules.pop("dags.etl_modules.fetcher", None)
    module = importlib.import_module("dags.etl_modules.fetcher")
    cache_module = importlib.import_module("dags.etl_modules.cache")

    assert module.cached_data is cache_module.cached_data


@pytest.mark.unit
def test_fetcher_file_import_resolves_cache_decorator():
    """File import resolves cached_data via absolute package imports."""
    fetcher_path = (
        Path(__file__).resolve().parents[2] / "dags" / "etl_modules" / "fetcher.py"
    )

    sys.modules.pop("fetcher_runtime_test", None)
    spec = importlib.util.spec_from_file_location("fetcher_runtime_test", fetcher_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    cache_module = importlib.import_module("dags.etl_modules.cache")
    assert module.cached_data is cache_module.cached_data
