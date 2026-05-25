import functools
import hashlib
import json
import logging
from datetime import date, datetime, timedelta

import pandas as pd
import redis

from dags.etl_modules.settings import get_env, get_env_int

# Configuration
REDIS_URL = get_env("REDIS_URL")
REDIS_HOST = get_env("REDIS_HOST", "redis")
REDIS_PORT = get_env_int("REDIS_PORT", 6379)
REDIS_PASSWORD = get_env("REDIS_PASSWORD")
REDIS_DB = get_env_int("REDIS_DB", 0)

_REDIS_CLIENT_UNINITIALIZED = object()
redis_client = _REDIS_CLIENT_UNINITIALIZED


def _create_redis_client():
    if REDIS_URL:
        logging.info("Connecting to Redis using REDIS_URL")
        return redis.from_url(REDIS_URL, decode_responses=False)

    logging.info(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}")
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=REDIS_DB,
        decode_responses=False,
    )


def get_redis_client():
    global redis_client

    if redis_client is _REDIS_CLIENT_UNINITIALIZED:
        try:
            client = _create_redis_client()
            client.ping()
            redis_client = client
            logging.info("Connected to Redis")
        except (redis.exceptions.RedisError, OSError, ValueError) as e:
            logging.warning(
                f"Failed to connect to Redis: {e}. Caching will be disabled."
            )
            redis_client = None

    return None if redis_client is _REDIS_CLIENT_UNINITIALIZED else redis_client


def _hash_pandas_object(value):
    hashed = pd.util.hash_pandas_object(value, index=True)
    return hashlib.md5(hashed.values.tobytes()).hexdigest()


def _normalize_for_key(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return {"__cache_kind__": "timestamp", "value": value.isoformat()}

    if isinstance(value, datetime):
        return {"__cache_kind__": "datetime", "value": value.isoformat()}

    if isinstance(value, date):
        return {"__cache_kind__": "date", "value": value.isoformat()}

    if isinstance(value, pd.DataFrame):
        return {"__cache_kind__": "dataframe", "value": _hash_pandas_object(value)}

    if isinstance(value, pd.Series):
        return {"__cache_kind__": "series", "value": _hash_pandas_object(value)}

    if isinstance(value, dict):
        return {
            str(key): _normalize_for_key(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_for_key(item) for item in value]

    if isinstance(value, set):
        normalized_items = [_normalize_for_key(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        )

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, AttributeError):
            pass

    return repr(value)


def _serialize_value(value):
    if isinstance(value, pd.DataFrame):
        return {
            "__cache_kind__": "dataframe",
            "columns": list(value.columns),
            "records": [
                _serialize_value(record) for record in value.to_dict("records")
            ],
        }

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return {"__cache_kind__": "timestamp", "value": value.isoformat()}

    if isinstance(value, datetime):
        return {"__cache_kind__": "datetime", "value": value.isoformat()}

    if isinstance(value, date):
        return {"__cache_kind__": "date", "value": value.isoformat()}

    if isinstance(value, dict):
        return {str(key): _serialize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]

    if isinstance(value, set):
        return [_serialize_value(item) for item in sorted(value, key=repr)]

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, AttributeError):
            pass

    if pd.isna(value):
        return None

    return value


def _deserialize_value(value):
    if isinstance(value, dict):
        cache_kind = value.get("__cache_kind__")

        if cache_kind == "dataframe":
            records = [
                _deserialize_value(record) for record in value.get("records", [])
            ]
            columns = value.get("columns")
            return pd.DataFrame(records, columns=columns)

        if cache_kind in {"datetime", "timestamp"}:
            return pd.Timestamp(value["value"]).to_pydatetime()

        if cache_kind == "date":
            return date.fromisoformat(value["value"])

        return {key: _deserialize_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_deserialize_value(item) for item in value]

    return value


def get_cache_key(func_name, args, kwargs, key_fn=None):
    """Generate a unique cache key based on function name and arguments."""
    if key_fn is not None:
        key_source = key_fn(*args, **kwargs)
    else:
        key_source = {"args": args, "kwargs": kwargs}

    normalized_key_source = _normalize_for_key(key_source)
    arg_str = json.dumps(
        normalized_key_source,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    arg_hash = hashlib.md5(arg_str.encode("utf-8")).hexdigest()
    return f"cache:{func_name}:{arg_hash}"


def cached_data(ttl_seconds=3600, key_fn=None):
    """
    Decorator to cache function results in Redis.
    Suitable for functions returning pandas DataFrames or serializable objects.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            client = get_redis_client()
            if not client:
                return func(*args, **kwargs)

            key = get_cache_key(func.__name__, args, kwargs, key_fn=key_fn)

            try:
                cached_bytes = client.get(key)
                if cached_bytes is not None:
                    logging.debug(f"Cache HIT for {func.__name__}")
                    cached_value = json.loads(cached_bytes.decode("utf-8"))
                    return _deserialize_value(cached_value)
            except (
                redis.exceptions.RedisError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as e:
                logging.warning(f"Error reading from cache for {func.__name__}: {e}")

            result = func(*args, **kwargs)

            try:
                if isinstance(result, pd.DataFrame) and result.empty:
                    pass
                elif result is None:
                    pass
                else:
                    payload = json.dumps(
                        _serialize_value(result),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    client.setex(key, timedelta(seconds=ttl_seconds), payload)
                    logging.debug(f"Cache SET for {func.__name__}")
            except (
                redis.exceptions.RedisError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                logging.warning(f"Error writing to cache for {func.__name__}: {e}")

            return result

        return wrapper

    return decorator
