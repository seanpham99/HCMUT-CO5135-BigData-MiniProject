#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

run_step() {
    printf '\n==> %s\n' "$1"
    shift
    "$@"
}

scope="${VALIDATE_SCOPE:-phased}"

if [[ "$scope" == "full" ]]; then
    lint_paths=(dags tests)
    mypy_paths=(dags tests)
    pytest_cmd=(uv run pytest)
elif [[ "$scope" == "phased" ]]; then
    lint_paths=(
        dags/etl_modules/adapters
        dags/etl_modules/contracts
        dags/etl_modules/orchestrators
        dags/etl_modules/transformers
        dags/etl_modules/errors.py
        dags/etl_modules/settings.py
        dags/etl_modules/fetcher.py
        dags/etl_modules/notifications.py
        dags/market_data_prices_daily.py
        dags/market_data_ratios_weekly.py
        dags/market_data_fundamentals_weekly.py
        tests/integration/test_dag_import_smoke.py
        tests/integration/test_ingest_company_intelligence_integration.py
        tests/unit/test_fetcher.py
        tests/unit/test_fetcher_imports.py
        tests/unit/test_fundamentals_orchestrator.py
        tests/unit/test_market_data_fundamentals_weekly.py
        tests/unit/test_market_data_prices_daily.py
        tests/unit/test_market_data_ratios_weekly.py
        tests/unit/test_market_data_repository.py
        tests/unit/test_notification_adapter.py
        tests/unit/test_notifications.py
        tests/unit/test_notifications_orchestrator.py
        tests/unit/test_pipeline_shared_helpers.py
        tests/unit/test_price_ratio_transformers.py
        tests/unit/test_prices_orchestrator.py
        tests/unit/test_provider_contracts.py
        tests/unit/test_ratios_orchestrator.py
    )
    mypy_paths=(
        dags/etl_modules/contracts
        dags/etl_modules/orchestrators
        dags/etl_modules/transformers
    )
    pytest_cmd=(
        uv run pytest -q -o addopts=''
        tests/unit/test_fetcher.py
        tests/unit/test_fundamentals_orchestrator.py
        tests/unit/test_market_data_fundamentals_weekly.py
        tests/unit/test_market_data_prices_daily.py
        tests/unit/test_market_data_ratios_weekly.py
        tests/unit/test_market_data_repository.py
        tests/unit/test_notification_adapter.py
        tests/unit/test_notifications.py
        tests/unit/test_notifications_orchestrator.py
        tests/unit/test_pipeline_shared_helpers.py
        tests/unit/test_price_ratio_transformers.py
        tests/unit/test_prices_orchestrator.py
        tests/unit/test_provider_contracts.py
        tests/unit/test_ratios_orchestrator.py
        tests/integration/test_dag_import_smoke.py
        tests/integration/test_ingest_company_intelligence_integration.py
    )
else
    printf 'Unsupported VALIDATE_SCOPE value: %s\n' "$scope" >&2
    exit 1
fi

run_step "ruff format check ($scope)" uv run ruff format --check "${lint_paths[@]}"
run_step "ruff lint check ($scope)" uv run ruff check "${lint_paths[@]}"
run_step "mypy type check ($scope)" \
    uv run mypy --follow-imports=skip "${mypy_paths[@]}"
run_step "pytest suite ($scope)" "${pytest_cmd[@]}"
