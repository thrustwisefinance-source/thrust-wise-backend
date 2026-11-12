"""
Cache namespace for the Set 3 stock scanners (separate from the ETF strategy
namespace and from every ``tw:v*`` key):

    tw:set3:scanner:{category}:{SYMBOL}:{as_of_date}:{fingerprint}

One entry per symbol; ``as_of_date`` is the request date (UTC) and
``fingerprint`` hashes the scanner id, every scoring parameter and the
calculation version, so different thresholds / periods / versions never share
entries. Only reads and writes keys built here - no deletes, no scans.
Redis failures degrade to "no cache" and never break the endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import hashlib
import json

from app.services import cache

logger = logging.getLogger(__name__)

SCANNER_CACHE_PREFIX = "tw:set3:scanner"
TTL_COMPLETE_SECONDS = 900
TTL_PARTIAL_SECONDS = 300


def scanner_fingerprint(scanner_id: str, params: Any, calculation_version: str) -> str:
    payload = {
        "scanner_id": scanner_id,
        "version": calculation_version,
        "params": params.model_dump(mode="json") if hasattr(params, "model_dump") else params,
    }
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def scanner_cache_key(category: str, symbol: str, as_of: str, fingerprint: str) -> str:
    key = f"{SCANNER_CACHE_PREFIX}:{category}:{symbol.upper()}:{as_of}:{fingerprint}"
    assert key.startswith("tw:set3:scanner:")
    return key


async def get_cached(key: str) -> dict | None:
    try:
        value = await cache.get_json(key)
    except Exception:  # noqa: BLE001
        logger.warning("Set 3 scanner cache read failed for %s", key, exc_info=True)
        return None
    return value if isinstance(value, dict) else None


async def set_cached(key: str, value: dict, ttl: int) -> None:
    try:
        await cache.set_json(key, value, ttl=ttl)
    except Exception:  # noqa: BLE001
        logger.warning("Set 3 scanner cache write failed for %s", key, exc_info=True)
