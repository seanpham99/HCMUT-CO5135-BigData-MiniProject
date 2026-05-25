from __future__ import annotations

import logging
import random
import re
import ssl
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, TypeVar
from urllib.error import HTTPError, URLError

import redis
import requests

from dags.etl_modules.cache import get_redis_client

logger = logging.getLogger(__name__)

T = TypeVar("T")

_STATUS_PATTERN = re.compile(r"\bHTTP\s+(\d{3})\b")
_BUCKET_PREFIX = "rate_governor:bucket"


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float
    jitter_ratio: float
    min_interval_seconds: float


_RETRY_PROFILES: dict[str, RetryPolicy] = {
    "conservative": RetryPolicy(
        max_attempts=4,
        base_delay_seconds=1.0,
        max_delay_seconds=20.0,
        jitter_ratio=0.30,
        min_interval_seconds=0.40,
    ),
    "balanced": RetryPolicy(
        max_attempts=5,
        base_delay_seconds=0.8,
        max_delay_seconds=12.0,
        jitter_ratio=0.25,
        min_interval_seconds=0.25,
    ),
    "aggressive": RetryPolicy(
        max_attempts=6,
        base_delay_seconds=0.5,
        max_delay_seconds=8.0,
        jitter_ratio=0.20,
        min_interval_seconds=0.15,
    ),
}

_SOURCE_MIN_INTERVAL_OVERRIDES = {
    "vci": 0.30,
    "vietstock": 0.35,
}

_local_last_request_at: dict[str, float] = {}
_local_locks: defaultdict[str, Lock] = defaultdict(Lock)


def _resolve_policy(source: str, retry_profile: str) -> RetryPolicy:
    profile = _RETRY_PROFILES.get(retry_profile, _RETRY_PROFILES["balanced"])
    source_interval = _SOURCE_MIN_INTERVAL_OVERRIDES.get(
        str(source or "").strip().lower()
    )
    if source_interval is None:
        return profile
    return RetryPolicy(
        max_attempts=profile.max_attempts,
        base_delay_seconds=profile.base_delay_seconds,
        max_delay_seconds=profile.max_delay_seconds,
        jitter_ratio=profile.jitter_ratio,
        min_interval_seconds=max(profile.min_interval_seconds, source_interval),
    )


def _extract_status_code(exc: Exception) -> int | None:
    if isinstance(exc, HTTPError):
        return int(exc.code)
    if isinstance(exc, requests.exceptions.HTTPError):
        if exc.response is not None and exc.response.status_code is not None:
            return int(exc.response.status_code)
    if hasattr(exc, "code"):
        code = getattr(exc, "code")
        if isinstance(code, int):
            return code
    match = _STATUS_PATTERN.search(str(exc))
    if match:
        return int(match.group(1))
    return None


def _extract_retry_after_seconds(exc: Exception) -> float | None:
    headers: Any = None
    if isinstance(exc, HTTPError):
        headers = exc.headers
    elif isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        headers = exc.response.headers

    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception, status_code: int | None) -> bool:
    if status_code in {429, 500, 502, 503, 504}:
        return True
    if isinstance(
        exc,
        (
            URLError,
            TimeoutError,
            ssl.SSLError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ),
    ):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        return status_code is not None and 500 <= status_code < 600
    return False


def _compute_backoff_seconds(
    policy: RetryPolicy,
    attempt: int,
    *,
    retry_after_seconds: float | None = None,
) -> float:
    base = min(
        policy.max_delay_seconds,
        policy.base_delay_seconds * (2 ** max(0, attempt - 1)),
    )
    if retry_after_seconds is not None:
        base = max(base, min(retry_after_seconds, policy.max_delay_seconds))
    jitter = random.uniform(0.0, base * policy.jitter_ratio)
    return min(policy.max_delay_seconds, base + jitter)


def _acquire_local_slot(source: str, min_interval_seconds: float) -> None:
    key = str(source or "default").strip().lower()
    lock = _local_locks[key]
    with lock:
        now = time.monotonic()
        last = _local_last_request_at.get(key, 0.0)
        wait_seconds = min_interval_seconds - (now - last)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        _local_last_request_at[key] = time.monotonic()


def _acquire_source_slot(source: str, min_interval_seconds: float) -> None:
    key = str(source or "default").strip().lower()
    redis_client = get_redis_client()
    if redis_client is None:
        _acquire_local_slot(key, min_interval_seconds)
        return

    bucket_key = f"{_BUCKET_PREFIX}:{key}"
    if min_interval_seconds >= 1.0:
        max_tokens = 1
        window_ms = max(1, int(min_interval_seconds * 1000))
    else:
        max_tokens = max(1, int(1.0 / min_interval_seconds))
        window_ms = 1000

    max_wait_seconds = max(5.0, min_interval_seconds * 30)
    deadline = time.monotonic() + max_wait_seconds

    while time.monotonic() < deadline:
        now_ms = int(time.time() * 1000)
        window_start = now_ms - window_ms
        try:
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(bucket_key, 0, window_start)
            pipe.zcard(bucket_key)
            _, current_count = pipe.execute()

            if current_count < max_tokens:
                token = f"{now_ms}-{uuid.uuid4().hex[:8]}"
                pipe = redis_client.pipeline()
                pipe.zadd(bucket_key, {token: now_ms})
                pipe.pexpire(bucket_key, window_ms * 2)
                pipe.execute()
                return
        except (
            redis.exceptions.RedisError,
            AttributeError,
            TypeError,
            ValueError,
        ):
            _acquire_local_slot(key, min_interval_seconds)
            return

        time.sleep(min(0.05, max(0.01, min_interval_seconds * 0.5)))

    raise RuntimeError(
        f"Rate-limit slot timed out for source={key} after {max_wait_seconds:.1f}s"
    )


def governed_call(
    source: str,
    request_fn: Callable[[], T],
    *,
    retry_profile: str = "balanced",
    operation: str = "request",
) -> T:
    policy = _resolve_policy(source, retry_profile)
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        _acquire_source_slot(source, policy.min_interval_seconds)
        try:
            return request_fn()
        except Exception as exc:
            last_error = exc
            status_code = _extract_status_code(exc)
            retryable = _is_retryable(exc, status_code)
            if not retryable or attempt >= policy.max_attempts:
                raise

            delay = _compute_backoff_seconds(
                policy,
                attempt,
                retry_after_seconds=_extract_retry_after_seconds(exc),
            )
            logger.warning(
                "Retrying %s for source=%s after attempt %s/%s (status=%s, wait=%.2fs): %s",
                operation,
                source,
                attempt,
                policy.max_attempts,
                status_code,
                delay,
                exc,
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"governed_call exhausted without result for source={source}")
