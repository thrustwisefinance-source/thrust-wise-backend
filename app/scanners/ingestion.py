"""
Scanner ingestion: populates/refreshes the data app/scanners/canslim.py
needs — universe membership, fundamentals snapshots, and stock price
history — kept separate from app.services.ingestion (which only ever
knew about ThrustWise's six ETFs + indices) rather than folding stock
ingestion into that module.

Bounded concurrency (`PRICE_INGESTION_CONCURRENCY`) is used for the
per-symbol price backfill so a ~500-stock universe doesn't serialize
500 sequential HTTP round-trips, while still keeping well under any
reasonable provider rate limit — this is the "avoid N×M request
explosions" performance requirement applied to the one leg (price
history) that EODHD has no bulk endpoint for.
"""

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal
from app.scanners import data as scanner_data
from app.scanners import universe as scanner_universe
from app.services import cache

logger = logging.getLogger(__name__)

PRICE_INGESTION_CONCURRENCY = 8

SCANNER_CACHE_PREFIX = "tw:v1:scanner:"


async def refresh_universe_and_data(db: AsyncSession, universe_key: str) -> dict:
    """Full refresh for one universe: constituent list -> fundamentals ->
    price history -> bust the scanner's own cache namespace.

    Each stage is independent and logged; a failure in one stage does
    not necessarily block the others (fundamentals/prices are still
    worth refreshing for whatever constituents are already on file even
    if, say, the universe-membership refresh itself fails because EODHD
    is down).
    """
    result = {
        "universe": universe_key,
        "constituents_ingested": 0,
        "fundamentals_updated": 0,
        "price_symbols_updated": 0,
        "price_rows_inserted": 0,
        "errors": [],
    }

    try:
        result["constituents_ingested"] = await scanner_universe.refresh_universe(
            db, universe_key
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Universe refresh failed for %s", universe_key)
        result["errors"].append(f"universe: {exc}")

    members = await scanner_universe.get_stock_universe(db, universe_key)
    symbols = [m.symbol for m in members]
    if not symbols:
        result["errors"].append(
            "No universe members on file — skipping fundamentals/price refresh."
        )
        return result

    try:
        result["fundamentals_updated"] = await scanner_data.refresh_fundamentals_snapshots(
            db, symbols
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fundamentals refresh failed for %s", universe_key)
        result["errors"].append(f"fundamentals: {exc}")

    price_rows, price_symbols, price_errors = await _refresh_prices_bounded(db, symbols)
    result["price_rows_inserted"] = price_rows
    result["price_symbols_updated"] = price_symbols
    result["errors"].extend(price_errors)

    await cache.delete_prefix(f"{SCANNER_CACHE_PREFIX}canslim:{universe_key}")

    logger.info(
        "Scanner refresh complete for %s: %d constituents, %d fundamentals, "
        "%d symbols priced (%d rows), %d errors",
        universe_key,
        result["constituents_ingested"],
        result["fundamentals_updated"],
        result["price_symbols_updated"],
        result["price_rows_inserted"],
        len(result["errors"]),
    )
    return result


async def _refresh_prices_bounded(
    db: AsyncSession, symbols: list[str]
) -> tuple[int, int, list[str]]:
    """Bounded-concurrency price backfill.

    A SQLAlchemy AsyncSession is not safe for concurrent use from
    multiple coroutines, so each concurrent task opens its OWN short-
    lived session for its DB write (same pattern as
    app.services.ingestion.ingest_all's per-symbol sessions) rather than
    sharing the caller's `db` — only the outer refresh_universe_and_data
    orchestration (universe read/write) uses the passed-in session.
    """
    semaphore = asyncio.Semaphore(PRICE_INGESTION_CONCURRENCY)
    errors: list[str] = []
    total_rows = 0
    symbols_ok = 0

    async def _one(symbol: str) -> None:
        nonlocal total_rows, symbols_ok
        async with semaphore:
            try:
                async with AsyncSessionLocal() as session:
                    inserted = await scanner_data.ingest_stock_prices(
                        session, symbol, history_years=settings.ingestion_history_years
                    )
                total_rows += inserted
                symbols_ok += 1
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the rest
                logger.warning("Price ingestion failed for %s: %s", symbol, exc)
                errors.append(f"prices:{symbol}: {exc}")

    await asyncio.gather(*(_one(s) for s in symbols))
    return total_rows, symbols_ok, errors


async def _scheduled_refresh_all_configured_universes() -> None:
    for universe_key in settings.scanner_universes_to_ingest_list:
        try:
            async with AsyncSessionLocal() as session:
                await refresh_universe_and_data(session, universe_key)
        except Exception:  # noqa: BLE001 — one bad universe must not stop the rest
            logger.exception("Scheduled scanner refresh failed for %s", universe_key)


def create_scanner_scheduler() -> AsyncIOScheduler:
    """Nightly refresh for every universe listed in
    settings.scanner_universes_to_ingest, offset from the ETF/index
    nightly ingestion (22:30 UTC, see app.services.ingestion) so the two
    don't compete for the same window. Opt-in — see
    settings.run_scanner_scheduler / settings.scanner_universes_to_ingest.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        _scheduled_refresh_all_configured_universes,
        CronTrigger(day_of_week="mon-fri", hour=23, minute=15, timezone="UTC"),
        id="nightly_scanner_refresh",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler
