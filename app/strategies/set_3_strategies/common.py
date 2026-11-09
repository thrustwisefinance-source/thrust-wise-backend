"""
Set 3 shared utilities.

Everything in here is used by the Set 3 strategy modules / router ONLY. It
deliberately imports nothing from app.services.strategies (the existing
strategy engine) so the two areas stay independent; it is also free of any
database / settings import so the calculation modules are unit-testable in
isolation.

Contents
--------
* Cache-key helpers  -> independent ``tw:set3:{category}:{symbol}`` namespaces
* Strategy registry primitives (``Set3Strategy`` / ``Set3Context``)
* Price-frame helpers, JSON sanitising, drawdown / downsampling helpers
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SET_NAME = "set3"
CACHE_PREFIX = "tw:set3"
CATEGORIES: tuple[str, ...] = (
    "technical",
    "quantitative",
    "regime",
    "calendar",
    "portfolio",
)
DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Cache keys  (independent namespace per category - never touches tw:v*:)
# ---------------------------------------------------------------------------


def cache_namespace(category: str, symbol: str) -> str:
    """``tw:set3:{category}:{symbol}`` - the per-category namespace."""
    return f"{CACHE_PREFIX}:{category}:{symbol.upper()}"


def params_fingerprint(payload: Any) -> str:
    """Stable short hash of any JSON-able / pydantic payload."""
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    category: str, symbol: str, *, data_version: str, fingerprint: str
) -> str:
    """
    ``tw:set3:{category}:{symbol}:{data_version}:{fingerprint}``

    ``data_version`` is derived from the newest ingested price date, so a
    nightly ingestion automatically produces a fresh key (the existing
    ingestion job only purges ``tw:v*`` prefixes and must not be edited).
    ``fingerprint`` covers selected strategies + every request parameter
    (including the random seed).
    """
    return f"{cache_namespace(category, symbol)}:{data_version}:{fingerprint}"


# ---------------------------------------------------------------------------
# Registry primitives
# ---------------------------------------------------------------------------


class Set3StrategyUnavailable(Exception):
    """Raised by a strategy when it cannot be computed for deterministic
    reasons (e.g. not enough history). The message is shown to the client
    and the outcome is cacheable."""


@dataclass
class Set3Context:
    """Everything a strategy's ``compute`` needs - no DB / network access."""

    symbol: str
    prices: pd.DataFrame
    extra_prices: dict[str, pd.DataFrame] = field(default_factory=dict)
    params: Any = None


def _no_symbols(_params: Any) -> tuple[str, ...]:
    return ()


def _always_cacheable(_params: Any) -> bool:
    return True


@dataclass(frozen=True)
class Set3Strategy:
    """
    One registered Set 3 strategy.

    key               snake_case; MUST equal the field name on the category's
                      ``strategies`` response model in schemas/strategy_set3.py
    compute           sync, pure: ``Set3Context -> dict``; raise
                      ``Set3StrategyUnavailable`` when it cannot be computed
    required_symbols  extra symbols whose prices the router must load
    cacheable         False for non-deterministic requests (e.g. no seed)
    """

    key: str
    title: str
    category: str
    compute: Callable[[Set3Context], dict]
    required_symbols: Callable[[Any], tuple[str, ...]] = _no_symbols
    cacheable: Callable[[Any], bool] = _always_cacheable


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------


def prices_to_frame(rows: list[Any]) -> pd.DataFrame:
    """
    Convert DailyPrice-like rows into a clean, date-indexed frame.

    Column ``price`` is the adjusted close (falls back to close), so strategy
    and Buy & Hold are both measured on the same total-return basis.
    Non-positive / NaN prices are dropped; duplicate dates keep the last row.
    """
    if not rows:
        return pd.DataFrame(columns=["price", "close", "volume"])
    records = []
    for r in rows:
        adj = getattr(r, "adjusted_close", None)
        close = getattr(r, "close", None)
        price = adj if adj not in (None, 0) and not _is_nan(adj) else close
        records.append(
            {
                "date": pd.Timestamp(r.date),
                "price": price,
                "close": close,
                "volume": getattr(r, "volume", 0) or 0,
            }
        )
    frame = pd.DataFrame.from_records(records).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame[frame["price"].notna() & (frame["price"] > 0)]
    return frame


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):
        return False


def slice_dates(
    frame: pd.DataFrame, start: datetime.date | None, end: datetime.date | None
) -> pd.DataFrame:
    """Inclusive date window (no-op for ``None`` bounds)."""
    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end)]
    return frame


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def years_between(start: datetime.date, end: datetime.date) -> float:
    return (end - start).days / DAYS_PER_YEAR


def annualized_return_pct(
    total_return: np.ndarray | float, years: float
) -> np.ndarray | float | None:
    """CAGR in percent from a total return given as a fraction (0.12 = 12 %)."""
    if years <= 0:
        return None
    growth = np.maximum(np.asarray(total_return, dtype=float) + 1.0, 0.0)
    out = (np.power(growth, 1.0 / years) - 1.0) * 100.0
    return out if isinstance(total_return, np.ndarray) else float(out)


def max_drawdown_pct(equity: np.ndarray) -> np.ndarray:
    """
    Max drawdown in percent (<= 0) along the LAST axis; works for a single
    curve (1-D) or one curve per row (2-D).
    """
    equity = np.asarray(equity, dtype=float)
    running_peak = np.maximum.accumulate(equity, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(running_peak > 0, equity / running_peak - 1.0, 0.0)
    return dd.min(axis=-1) * 100.0


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    """Evenly spaced indices that always include the first and last point."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def rnd(value: Any, digits: int = 2) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / date types and NaN/inf into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime.date, datetime.datetime, pd.Timestamp)):
        return obj.isoformat()[:10] if not isinstance(obj, datetime.datetime) else obj.isoformat()
    return obj
