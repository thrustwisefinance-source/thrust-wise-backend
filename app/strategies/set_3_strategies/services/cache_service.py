"""
Set 3 cache integration for the additive quantitative strategies.

Keys use the Set 3 namespace

    tw:set3:{category}:{symbol}:{newest_price_date}:{fingerprint}

built with the existing Set 3 helper (``common.build_cache_key``), so nothing
here can ever address a ``tw:v*`` (Set 1 / Set 2) key. This module only
READS and WRITES keys it built itself - it never deletes or scans keys and
does not touch any invalidation logic.

The fingerprint covers the strategy id, EVERY request parameter and the
calculation version, so different parameter sets (or a new calculation
version) never reuse each other's entries. The Redis client is the existing
``app.services.cache`` module (graceful degradation / circuit breaker); every
call is additionally wrapped here so a cache failure can never break an
endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

from app.services import cache
from app.strategies.set_3_strategies.common import (
    CACHE_PREFIX,
    build_cache_key,
    params_fingerprint,
)

logger = logging.getLogger(__name__)

CACHE_NAMESPACE_PREFIX = CACHE_PREFIX  # "tw:set3"


def strategy_fingerprint(
    strategy_id: str,
    params: Any,
    calculation_version: str,
    extra: dict[str, Any] | None = None,
) -> str:
    """Stable fingerprint of (strategy, params, version[, extra inputs])."""
    payload: dict[str, Any] = {
        "strategy_id": strategy_id,
        "version": calculation_version,
        "params": params.model_dump(mode="json") if hasattr(params, "model_dump") else params,
    }
    if extra:
        payload["extra"] = extra
    return params_fingerprint(payload)


def strategy_cache_key(
    category: str, symbol: str, newest_price_date: str, fingerprint: str
) -> str:
    """``tw:set3:{category}:{symbol}:{newest_price_date}:{fingerprint}``"""
    key = build_cache_key(
        category, symbol, data_version=newest_price_date, fingerprint=fingerprint
    )
    assert key.startswith(f"{CACHE_NAMESPACE_PREFIX}:")  # namespace guard
    return key


async def get_cached(key: str) -> dict | None:
    try:
        value = await cache.get_json(key)
    except Exception:  # noqa: BLE001 - cache must never break the endpoint
        logger.warning("Set 3 cache read failed for %s", key, exc_info=True)
        return None
    return value if isinstance(value, dict) else None


async def set_cached(key: str, value: dict, ttl: int | None = None) -> None:
    try:
        await cache.set_json(key, value, ttl=ttl)
    except Exception:  # noqa: BLE001
        logger.warning("Set 3 cache write failed for %s", key, exc_info=True)
