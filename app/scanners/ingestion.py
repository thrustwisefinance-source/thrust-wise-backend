"""
Scanner ingestion: populates/refreshes everything app/scanners/canslim.py
needs — universe membership, the S&P 500 INDEX's own price history (for
the M/market-direction criterion), fundamentals snapshots, and stock
price history — entirely via yfinance (see app.scanners.yfinance_client,
app.scanners.data, app.scanners.universe).

THIS MODULE NEVER CALLS EODHD, hidden or otherwise. It is kept separate
from app.services.ingestion (which only ever knew about ThrustWise's six
ETFs + indices, and still uses EODHD, completely untouched) rather than
folding stock ingestion into that module — see each module's own
docstring for the OLD-vs-NEW split.

Bounded concurrency is used throughout (see
app.scanners.yfinance_client.YFINANCE_MAX_CONCURRENCY and
PRICE_INGESTION_CONCURRENCY below) so a ~500-stock universe refresh never
opens an uncontrolled burst of connections to Yahoo Finance — yfinance is
an unofficial, unthrottled wrapper around Yahoo's own endpoints, not an
SLA-backed enterprise API like EODHD (see app.scanners.yfinance_client
module docstring for the documented limitations this implies).

A single bad ticker (delisted, no yfinance coverage, transient network
error, ...) is caught and logged at every stage and never aborts the
rest of the universe's refresh — see `_refresh_prices_bounded` and
app.scanners.data.refresh_fundamentals_snapshots.
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

# yfinance/Yahoo has no documented rate limit but throttles/blocks
# aggressive scraping in practice — this outer bound is deliberately
# close to (and, in effect, tightened further by)
# app.scanners.yfinance_client.YFINANCE_MAX_CONCURRENCY; kept as its own
# constant here because it also bounds how many concurrent short-lived
# DB sessions/writes are in flight (see _refresh_prices_bounded).
PRICE_INGESTION_CONCURRENCY = 5

SCANNER_CACHE_PREFIX = "tw:v1:scanner:"


async def refresh_universe_and_data(db: AsyncSession, universe_key: str) -> dict:
    """Full refresh for one universe: constituent list -> S&P 500 index
    price history -> fundamentals -> stock price history -> bust the
    scanner's own cache namespace.

    Each stage is independent and logged; a failure in one stage does
    not block the others (fundamentals/prices are still worth refreshing
    for whatever constituents are already on file even if, say, the
    universe-membership refresh itself fails because Wikipedia is
    unreachable).
    """
    logger.info("Starting %s yfinance scanner refresh", universe_key)

    result = {
        "universe": universe_key,
        "constituents_ingested": 0,
        "fundamentals_succeeded": 0,
        "fundamentals_failed": 0,
        "fundamentals_updated": 0,  # legacy alias, kept for anything still reading this key
        "price_symbols_updated": 0,
        "price_symbols_failed": 0,
        "price_rows_inserted": 0,
        "market_index_rows_inserted": 0,
        "errors": [],
    }

    try:
        result["constituents_ingested"] = await scanner_universe.refresh_universe(
            db, universe_key
        )
        logger.info("Universe members discovered: %d", result["constituents_ingested"])
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

    # S&P 500 INDEX price history (yfinance "^GSPC") — needed for the M
    # criterion. Ingested HERE, by the scanner's own pipeline, rather
    # than reusing the OLD ETF/dashboard pipeline's EODHD-sourced "GSPC"
    # (app.constants.INDEX_REGISTRY) — see app.scanners.data module
    # docstring for why M must never depend on EODHD, even indirectly.
    try:
        async with AsyncSessionLocal() as session:
            result["market_index_rows_inserted"] = await scanner_data.ingest_stock_prices(
                session,
                scanner_data.MARKET_INDEX_SYMBOL,
                history_years=settings.scanner_price_history_years,
            )
        logger.info(
            "%s prices: OK (%d new row(s))",
            scanner_data.MARKET_INDEX_SYMBOL,
            result["market_index_rows_inserted"],
        )
    except Exception as exc:  # noqa: BLE001 — the market index alone must not abort the refresh
        logger.warning(
            "yfinance failed for %s (market index): %s", scanner_data.MARKET_INDEX_SYMBOL, exc
        )
        result["errors"].append(f"market_index: {exc}")

    try:
        succeeded, failed = await scanner_data.refresh_fundamentals_snapshots(db, symbols)
        result["fundamentals_succeeded"] = succeeded
        result["fundamentals_failed"] = failed
        result["fundamentals_updated"] = succeeded
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fundamentals refresh failed for %s", universe_key)
        result["errors"].append(f"fundamentals: {exc}")

    price_rows, price_symbols_ok, price_symbols_failed, price_errors = (
        await _refresh_prices_bounded(db, symbols)
    )
    result["price_rows_inserted"] = price_rows
    result["price_symbols_updated"] = price_symbols_ok
    result["price_symbols_failed"] = price_symbols_failed
    result["errors"].extend(price_errors)

    await cache.delete_prefix(f"{SCANNER_CACHE_PREFIX}canslim:{universe_key}")

    logger.info(
        "Scanner refresh completed\n"
        "Universe: %s\n"
        "Members: %d\n"
        "Fundamentals succeeded: %d\n"
        "Fundamentals failed: %d\n"
        "Prices succeeded: %d\n"
        "Prices failed: %d\n"
        "Errors: %d",
        universe_key,
        result["constituents_ingested"],
        result["fundamentals_succeeded"],
        result["fundamentals_failed"],
        result["price_symbols_updated"],
        result["price_symbols_failed"],
        len(result["errors"]),
    )
    return result


async def _refresh_prices_bounded(
    db: AsyncSession, symbols: list[str]
) -> tuple[int, int, int, list[str]]:
    """Bounded-concurrency stock price backfill via yfinance.

    A SQLAlchemy AsyncSession is not safe for concurrent use from
    multiple coroutines, so each concurrent task opens its OWN short-
    lived session for its DB write (same pattern as
    app.scanners.data.refresh_fundamentals_snapshots and
    app.services.ingestion.ingest_all's per-symbol sessions) rather than
    sharing the caller's `db` — only the outer refresh_universe_and_data
    orchestration (universe read/write) uses the passed-in session.

    Returns (rows_inserted, symbols_ok, symbols_failed, error_messages).
    """
    semaphore = asyncio.Semaphore(PRICE_INGESTION_CONCURRENCY)
    errors: list[str] = []
    total_rows = 0
    symbols_ok = 0
    symbols_failed = 0

    async def _one(symbol: str) -> None:
        nonlocal total_rows, symbols_ok, symbols_failed
        async with semaphore:
            logger.info("Processing %s", symbol)
            try:
                async with AsyncSessionLocal() as session:
                    inserted = await scanner_data.ingest_stock_prices(
                        session, symbol, history_years=settings.scanner_price_history_years
                    )
                total_rows += inserted
                symbols_ok += 1
                logger.info("%s prices: OK (%d new row(s))", symbol, inserted)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the rest
                symbols_failed += 1
                logger.warning("yfinance failed for %s: %s", symbol, exc)
                errors.append(f"prices:{symbol}: {exc}")

    await asyncio.gather(*(_one(s) for s in symbols))
    return total_rows, symbols_ok, symbols_failed, errors


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
