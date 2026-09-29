"""
Stock universe abstraction for scanners.

`get_stock_universe("sp500")` returns the currently-ingested constituent
list for that universe from Postgres (app.models.StockUniverseMember) —
NOT a hard-coded ticker list baked into scanner code. The list itself is
populated/refreshed by `refresh_universe` below, which pulls EODHD's
actual S&P 500 index-fundamentals "Components" section (see
app.services.eodhd_client.fetch_index_fundamentals) — the same real
data source approach used for every other symbol in this codebase
(EODHD), instead of the source article's own hard-coded 503-ticker
Python list, which goes stale the moment index membership changes.

If EODHD returns 403 for the Components lookup (Index Constituents
data is a separate entitlement from EODHD's base Fundamentals package;
see app.scanners.sp500_fallback), sp500 falls back to scraping
Wikipedia's actively-maintained constituent table instead of leaving
the universe permanently empty. EODHD is always tried first.

Adding another universe later (NASDAQ 100, Russell 1000, a custom list)
means adding one entry to UNIVERSE_REGISTRY plus (if it also comes from
an EODHD index) one EODHD index symbol — no changes to
app/scanners/canslim.py, app/scanners/data.py, or the router.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import StockUniverseMember
from app.scanners import sp500_fallback
from app.services import eodhd_client

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseSource:
    key: str  # e.g. "sp500" — used in the API and cache keys
    label: str  # e.g. "S&P 500"
    eodhd_index_symbol: str | None  # e.g. "GSPC.INDX"; None for a universe with no EODHD index backing


# Registry of universes the scanner architecture knows how to refresh.
# Only "sp500" is wired up to real EODHD data today (per the task's
# initial-implementation requirement); nasdaq100/russell1000 are listed
# as placeholders showing how the architecture extends, and report
# themselves as not-yet-available rather than silently falling back to
# sp500 or returning a fabricated list.
UNIVERSE_REGISTRY: dict[str, UniverseSource] = {
    "sp500": UniverseSource(key="sp500", label="S&P 500", eodhd_index_symbol="GSPC.INDX"),
    "nasdaq100": UniverseSource(
        key="nasdaq100", label="NASDAQ 100", eodhd_index_symbol="NDX.INDX"
    ),
    "russell1000": UniverseSource(
        key="russell1000", label="Russell 1000", eodhd_index_symbol=None
    ),
}


def list_universes() -> list[UniverseSource]:
    return list(UNIVERSE_REGISTRY.values())


def get_universe_source(universe_key: str) -> UniverseSource | None:
    return UNIVERSE_REGISTRY.get(universe_key.lower())


async def get_stock_universe(
    db: AsyncSession, universe_key: str
) -> list[StockUniverseMember]:
    """Currently-ingested, active members of `universe_key`, symbol-ascending.

    Returns an empty list (never a fabricated ticker list) if the
    universe hasn't been ingested yet — the router surfaces this as a
    clear "universe not yet ingested" message rather than 500ing or
    silently scanning zero stocks.
    """
    stmt = (
        select(StockUniverseMember)
        .where(
            StockUniverseMember.universe == universe_key.lower(),
            StockUniverseMember.is_current.is_(True),
        )
        .order_by(StockUniverseMember.symbol.asc())
    )
    rows = (await db.scalars(stmt)).all()
    return list(rows)


async def refresh_universe(db: AsyncSession, universe_key: str) -> int:
    """Refresh `universe_key`'s constituent list from its EODHD index.

    Marks every existing row for the universe `is_current=False`, then
    upserts the fresh constituent list back to `is_current=True` — see
    app.models.stock_universe module docstring for why (crash-safety:
    a failed refresh never leaves half-old/half-new rows indistinguishable).

    Returns the number of constituents ingested. Raises if the universe
    has no EODHD index backing (see UNIVERSE_REGISTRY) or EODHD returns
    no components — callers should not silently treat that as "zero
    stocks currently qualify".
    """
    source = get_universe_source(universe_key)
    if source is None:
        raise ValueError(f"Unknown universe: {universe_key}")
    if source.eodhd_index_symbol is None:
        raise ValueError(
            f"Universe '{universe_key}' has no EODHD index backing configured yet."
        )

    components_list: list[dict]
    try:
        payload = await eodhd_client.fetch_index_fundamentals(source.eodhd_index_symbol)
        components = (payload or {}).get("Components") or {}
        if not components:
            raise RuntimeError(
                f"EODHD returned no index components for {source.eodhd_index_symbol}; "
                "refusing to wipe the existing universe."
            )
        components_list = list(components.values())
    except httpx.HTTPStatusError as exc:
        # A 403 here specifically means EODHD's Index Components data isn't
        # included in this account's plan/add-ons (see this module's and
        # app.scanners.sp500_fallback's docstrings) — it is not a symbol,
        # auth, or logic problem, and retrying won't help. Only sp500 has a
        # free fallback source wired up today; other index-backed universes
        # still surface the failure as-is.
        if exc.response.status_code == 403 and universe_key.lower() == "sp500":
            logger.warning(
                "EODHD returned 403 for %s Index Components — this means "
                "Index Constituents data isn't included in the current EODHD "
                "plan/add-ons (contact support@eodhistoricaldata.com or check "
                "https://eodhd.com/pricing), not a bug in this code. Falling "
                "back to Wikipedia's S&P 500 constituent table for this refresh.",
                source.eodhd_index_symbol,
            )
            components_list = await sp500_fallback.fetch_sp500_constituents_from_wikipedia()
        else:
            raise

    await db.execute(
        update(StockUniverseMember)
        .where(StockUniverseMember.universe == universe_key.lower())
        .values(is_current=False)
    )

    count = 0
    for entry in components_list:
        code = (entry.get("Code") or "").strip().upper()
        if not code:
            continue

        existing = await db.scalar(
            select(StockUniverseMember).where(
                StockUniverseMember.universe == universe_key.lower(),
                StockUniverseMember.symbol == code,
            )
        )
        if existing:
            existing.name = entry.get("Name")
            existing.sector = entry.get("Sector")
            existing.industry = entry.get("Industry")
            existing.exchange = entry.get("Exchange") or "US"
            existing.is_current = True
        else:
            db.add(
                StockUniverseMember(
                    universe=universe_key.lower(),
                    symbol=code,
                    name=entry.get("Name"),
                    sector=entry.get("Sector"),
                    industry=entry.get("Industry"),
                    exchange=entry.get("Exchange") or "US",
                    is_current=True,
                )
            )
        count += 1

    await db.commit()
    return count