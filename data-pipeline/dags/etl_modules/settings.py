"""Centralized environment/configuration access for data-pipeline modules."""

from __future__ import annotations

import os

from dags.etl_modules.errors import ConfigurationError


def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    return value


def get_required_env(name: str) -> str:
    value = get_env(name)
    if value is None or value == "":
        raise ConfigurationError(f"Required environment variable {name} is not set")
    return value


def get_env_int(name: str, default: int) -> int:
    raw = get_env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"Environment variable {name} must be an integer, got {raw!r}"
        ) from exc
