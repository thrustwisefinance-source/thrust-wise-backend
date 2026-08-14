"""
Redis JSON cache with graceful degradation: if Redis is unavailable the
API still works, it just recomputes on every request.

Includes per-key locking to prevent stampede on cache miss.
"""

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable

import redis.asyncio as aioredis

from app.config import settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None

# Circuit breaker: after a connection failure, skip Redis entirely for a
# cooldown window instead of paying the connect timeout on every request.
_COOLDOWN_SECONDS = 60
_unavailable_until = 0.0

# Per-key locks to prevent stampede when multiple requests hit a cold key
_key_locks: dict[str, asyncio.Lock] = {}
_locks_lock = asyncio.Lock()


def _client() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=2,
        )
    return _redis


def _available() -> bool:
    return time.monotonic() >= _unavailable_until


def _mark_unavailable(op: str, key: str, exc: Exception) -> None:
    global _unavailable_until
    _unavailable_until = time.monotonic() + _COOLDOWN_SECONDS
    logger.warning(
        "Redis %s failed for %s (%s) — cache disabled for %ss",
        op,
        key,
        exc,
        _COOLDOWN_SECONDS,
    )


async def get_json(key: str) -> Any | None:
    if not _available():
        return None
    try:
        raw = await _client().get(key)
        return json.loads(raw) if raw else None
    except Exception as exc:  # noqa: BLE001 — cache must never break requests
        _mark_unavailable("GET", key, exc)
        return None


async def set_json(key: str, value: Any, ttl: int | None = None) -> None:
    if not _available():
        return
    try:
        await _client().set(
            key, json.dumps(value), ex=ttl or settings.cache_ttl_seconds
        )
    except Exception as exc:  # noqa: BLE001
        _mark_unavailable("SET", key, exc)


async def get_or_compute(
    key: str,
    compute_fn: Callable[[], Awaitable[Any]],
    ttl: int | None = None,
) -> Any:
    """Get cached value or compute it, with per-key lock to prevent stampede.

    Only one concurrent caller per key will run compute_fn; others wait
    for the winner's result.
    """
    # Fast path: cache hit
    cached = await get_json(key)
    if cached is not None:
        return cached

    # Acquire per-key lock (create if needed)
    async with _locks_lock:
        if key not in _key_locks:
            _key_locks[key] = asyncio.Lock()
        lock = _key_locks[key]

    async with lock:
        # Double-check after acquiring lock (winner may have populated cache)
        cached = await get_json(key)
        if cached is not None:
            return cached

        # Compute the value
        value = await compute_fn()
        await set_json(key, value, ttl=ttl)
        return value


async def delete_prefix(prefix: str) -> None:
    """Delete all keys starting with prefix (used after nightly ingestion)."""
    if not _available():
        return
    try:
        client = _client()
        async for key in client.scan_iter(match=f"{prefix}*"):
            await client.delete(key)
    except Exception as exc:  # noqa: BLE001
        _mark_unavailable("DEL", prefix, exc)


async def close() -> None:
    global _redis
    if _redis is not None:
        try:
            await _redis.aclose()
        except Exception:  # noqa: BLE001
            pass
        _redis = None
