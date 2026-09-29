"""
Stock universe abstraction for scanners.

`get_stock_universe("sp500")` returns the currently-ingested constituent
list for that universe from Postgres (app.models.StockUniverseMember) —
NOT a hard-coded ticker list baked into scanner code.

THIS MODULE NO LONGER USES EODHD. S&P 500 constituents are sourced from
Wikipedia's actively-maintained "List of S&P 500 companies" table (see
app.scanners.sp500_fallback) — never from EODHD's index-fundamentals/
Components endpoint (GSPC.INDX), and there is no hidden EODHD fallback
anywhere in this path. EODHD is used only by the OLD ETF/index ingestion
(app.services.ingestion, app.services.eodhd_client), which this file
does not import and does not touch. See app.scanners.data /
app.scanners.yfinance_client for why per-stock fundamentals/prices also
no longer use EODHD.

Adding another universe later (NASDAQ 100, Russell 1000, a custom list)
means adding one entry to UNIVERSE_REGISTRY plus a constituent-source
function for it in refresh_universe below — no changes to
app/scanners/canslim.py, app/scanners/data.py, or the router.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import StockUniverseMember
from app.scanners import sp500_fallback

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniverseSource:
    key: str  # e.g. "sp500" — used in the API and cache keys
    label: str  # e.g. "S&P 500"
    # Kept ONLY for GET /api/scanners/universes' response-shape stability
    # (see app.schemas.scanners.StockUniverseInfo.eodhd_index_symbol).
    # No universe ingests from EODHD any more, so this is always None —
    # it is not read anywhere in this module's actual ingestion logic.
    eodhd_index_symbol: str | None = None


# Registry of universes the scanner architecture knows how to refresh.
# Only "sp500" is wired up to a real constituent source today (Wikipedia
# — see refresh_universe); nasdaq100/russell1000 are listed as
# placeholders showing how the architecture extends, and report
# themselves as not-yet-available rather than silently falling back to
# sp500 or returning a fabricated list.
UNIVERSE_REGISTRY: dict[str, UniverseSource] = {
    "sp500": UniverseSource(key="sp500", label="S&P 500"),
    "nasdaq100": UniverseSource(key="nasdaq100", label="NASDAQ 100"),
    "russell1000": UniverseSource(key="russell1000", label="Russell 1000"),
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
    """Refresh `universe_key`'s constituent list.

    sp500: sourced from Wikipedia's constituent table (see
    app.scanners.sp500_fallback) — no EODHD call is made for this
    universe, hidden or otherwise.
    nasdaq100 / russell1000: no free constituent source is wired up yet;
    they report themselves as not-yet-available (raise ValueError)
    rather than silently reusing sp500's list or fabricating one.

    Marks every existing row for the universe `is_current=False`, then
    upserts the fresh constituent list back to `is_current=True` (see
    app.models.stock_universe module docstring for why — crash safety: a
    failed refresh never leaves half-old/half-new rows indistinguishable).

    Returns the number of constituents ingested. Raises if the universe
    has no constituent source configured, or that source returns nothing
    — callers should not silently treat that as "zero stocks currently
    qualify".
    """
    source = get_universe_source(universe_key)
    if source is None:
        raise ValueError(f"Unknown universe: {universe_key}")

    key = universe_key.lower()
    if key == "sp500":
        components_list = await sp500_fallback.fetch_sp500_constituents_from_wikipedia()
        row_source_label = "wikipedia_sp500_table"
    else:
        raise ValueError(
            f"Universe '{universe_key}' has no constituent source configured yet."
        )

    if not components_list:
        raise RuntimeError(
            f"No constituents returned for '{universe_key}'; refusing to wipe "
            "the existing universe."
        )

    await db.execute(
        update(StockUniverseMember)
        .where(StockUniverseMember.universe == key)
        .values(is_current=False)
    )

    count = 0
    for entry in components_list:
        code = (entry.get("Code") or "").strip().upper()
        if not code:
            continue

        existing = await db.scalar(
            select(StockUniverseMember).where(
                StockUniverseMember.universe == key,
                StockUniverseMember.symbol == code,
            )
        )
        if existing:
            existing.name = entry.get("Name")
            existing.sector = entry.get("Sector")
            existing.industry = entry.get("Industry")
            existing.exchange = entry.get("Exchange") or "US"
            existing.is_current = True
            existing.source = row_source_label
        else:
            db.add(
                StockUniverseMember(
                    universe=key,
                    symbol=code,
                    name=entry.get("Name"),
                    sector=entry.get("Sector"),
                    industry=entry.get("Industry"),
                    exchange=entry.get("Exchange") or "US",
                    is_current=True,
                    source=row_source_label,
                )
            )
        count += 1

    await db.commit()
    return count
