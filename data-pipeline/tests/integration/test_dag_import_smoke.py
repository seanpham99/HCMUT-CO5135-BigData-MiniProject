"""Smoke checks that all Airflow DAG modules parse/import cleanly."""

import importlib
import pkgutil
from pathlib import Path

import pytest
from airflow.models.dagbag import DagBag

REQUIRED_ENV = {
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SECRET_OR_SERVICE_ROLE_KEY": "test-service-role-key",
    "SUPABASE_DB_URL": "postgresql://test_user:test_password@localhost:5432/postgres",
    "TELEGRAM_BOT_TOKEN": "test_token_123456",
    "TELEGRAM_CHAT_ID": "test_chat_id_789",
    "GEMINI_API_KEY": "test_gemini_key_abc",
    "DATA_PIPELINE_API_KEY": "test_data_pipeline_key",
    "PORTFOLIO_API_BASE_URL": "http://localhost:3000",
}

EXPECTED_DAG_IDS = {
    "assets_dimension_etl",
    "market_data_prices_daily",
    "market_news_morning",
    "refresh_historical_prices",
}


@pytest.fixture(autouse=True)
def _set_required_env(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)


@pytest.mark.integration
def test_all_dag_modules_import_without_runtime_errors():
    import dags

    dag_module_names = sorted(
        module.name
        for module in pkgutil.iter_modules(dags.__path__)
        if not module.ispkg
    )

    for module_name in dag_module_names:
        importlib.import_module(f"dags.{module_name}")


@pytest.mark.integration
def test_dagbag_parses_without_import_errors():
    dags_folder = Path(__file__).resolve().parents[2] / "dags"
    dagbag = DagBag(
        dag_folder=str(dags_folder),
        include_examples=False,
        safe_mode=False,
    )

    assert dagbag.import_errors == {}
    assert EXPECTED_DAG_IDS.issubset(set(dagbag.dag_ids))
