#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[pool-bootstrap] %s\n' "$1"
}

AIRFLOW_BIN="${AIRFLOW_BIN:-/home/airflow/.local/bin/airflow}"
if [[ ! -x "$AIRFLOW_BIN" ]]; then
  if command -v airflow >/dev/null 2>&1; then
    AIRFLOW_BIN="$(command -v airflow)"
  else
    log "Airflow CLI not found. Set AIRFLOW_BIN explicitly."
    exit 1
  fi
fi

require_integer() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    log "Invalid integer for ${name}: '${value}'"
    exit 1
  fi
}

set_pool() {
  local name="$1"
  local slots="$2"
  local description="$3"
  require_integer "slots for ${name}" "$slots"
  "$AIRFLOW_BIN" pools set "$name" "$slots" "$description"
  log "Ensured pool '${name}' with ${slots} slots"
}

VCI_POOL_NAME="${AIRFLOW_POOL_VCI_GRAPHQL_NAME:-vci_graphql}"
VCI_POOL_SLOTS="${AIRFLOW_POOL_VCI_GRAPHQL_SLOTS:-8}"
VCI_POOL_DESCRIPTION="${AIRFLOW_POOL_VCI_GRAPHQL_DESCRIPTION:-Legacy VCI GraphQL ingestion pool}"

KBS_POOL_NAME="${AIRFLOW_POOL_KBS_FINANCE_NAME:-kbs_finance}"
KBS_POOL_SLOTS="${AIRFLOW_POOL_KBS_FINANCE_SLOTS:-8}"
KBS_POOL_DESCRIPTION="${AIRFLOW_POOL_KBS_FINANCE_DESCRIPTION:-KBS finance ingestion pool}"

set_pool "$VCI_POOL_NAME" "$VCI_POOL_SLOTS" "$VCI_POOL_DESCRIPTION"
set_pool "$KBS_POOL_NAME" "$KBS_POOL_SLOTS" "$KBS_POOL_DESCRIPTION"
