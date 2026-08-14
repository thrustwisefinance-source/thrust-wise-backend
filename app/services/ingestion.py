"""
Background ingestion: pulls daily OHLCV history from EODHD into Postgres.

Runs once at startup (backfilling anything missing) and then nightly
after US market close via APScheduler.
"""

import datetime
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.constants import ETF_REGISTRY, INDEX_REGISTRY, PORTFOLIO_COMPARISON_REGISTRY
from app.database import AsyncSessionLocal
from app.models import DailyPrice, IngestionRun
from app.services import cache, eodhd_client

logger = logging.getLogger(__name__)


# Everything we pull daily bars for: explorable ETFs + dashboard indices +
# portfolio-comparison-only symbols (e.g. VT, VXUS — see
# PORTFOLIO_COMPARISON_REGISTRY docstring in app.constants).
def _all_price_sources() -> dict:
    return {**ETF_REGISTRY, **INDEX_REGISTRY, **PORTFOLIO_COMPARISON_REGISTRY}


async def ingest_symbol(session: AsyncSession, symbol: str) -> int:
    """Fetch missing daily bars for one plain symbol (e.g. "VOO" or "GSPC")."""
    meta = _all_price_sources()[symbol]

    last_date = await session.scalar(
        select(func.max(DailyPrice.date)).where(DailyPrice.symbol == symbol)
    )

    # Use UTC-anchored date to avoid timezone off-by-one near midnight
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()

    if last_date:
        from_date = last_date + datetime.timedelta(days=1)
        if from_date > today_utc:
            return 0
    else:
        from_date = today_utc - datetime.timedelta(
            days=365 * settings.ingestion_history_years + 30
        )

    rows = await eodhd_client.fetch_eod_history(
        meta.eodhd_symbol(), from_date=from_date
    )
    if not rows:
        return 0

    # Integrity checks: drop rows with invalid data
    valid_payload = []
    rejected = 0
    for row in rows:
        close_val = row.get("close")
        high_val = row.get("high")
        low_val = row.get("low")

        if close_val is None or float(close_val) <= 0:
            rejected += 1
            continue
        if (
            high_val is not None
            and low_val is not None
            and float(high_val) < float(low_val)
        ):
            rejected += 1
            logger.warning(
                "Rejected row for %s on %s: high (%.2f) < low (%.2f)",
                symbol,
                row.get("date"),
                float(high_val),
                float(low_val),
            )
            continue

        valid_payload.append(
            {
                "symbol": symbol,
                "date": datetime.date.fromisoformat(row["date"]),
                "open": float(row.get("open") or 0),
                "high": float(high_val or 0),
                "low": float(low_val or 0),
                "close": float(close_val),
                "adjusted_close": float(row.get("adjusted_close") or close_val),
                "volume": int(row.get("volume") or 0),
            }
        )

    if rejected:
        logger.warning("Rejected %d invalid rows for %s", rejected, symbol)

    if not valid_payload:
        return 0

    stmt = pg_insert(DailyPrice).values(valid_payload)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_daily_prices_symbol_date")
    await session.execute(stmt)
    await session.commit()
    return len(valid_payload)


async def ingest_all(trigger: str = "schedule") -> None:
    """Ingest every ETF and index symbol, audit the run, then drop stale cache."""
    total_inserted = 0
    symbols_ok = 0
    symbols_failed = 0
    first_error: str | None = None

    # Open the audit row up front so a crash mid-run stays visible as "running"
    async with AsyncSessionLocal() as session:
        run = IngestionRun(trigger=trigger, status="running")
        session.add(run)
        await session.commit()
        run_id = run.id

    async with AsyncSessionLocal() as session:
        for symbol in _all_price_sources():
            try:
                inserted = await ingest_symbol(session, symbol)
                if inserted:
                    logger.info("Ingested %s rows for %s", inserted, symbol)
                    total_inserted += inserted
                symbols_ok += 1
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the rest
                logger.exception("Ingestion failed for %s", symbol)
                symbols_failed += 1
                if first_error is None:
                    first_error = f"{symbol}: {exc}"

    if symbols_failed == 0:
        status = "success"
    elif symbols_ok > 0:
        status = "partial"
    else:
        status = "failed"

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                finished_at=datetime.datetime.now(datetime.timezone.utc),
                status=status,
                symbols_ok=symbols_ok,
                symbols_failed=symbols_failed,
                rows_inserted=total_inserted,
                detail=first_error,
            )
        )
        await session.commit()

    # Invalidate caches with the namespaced prefix
    for prefix in (
        "tw:v1:snapshot:",
        "tw:v1:details:",
        "tw:v1:quote:",
        "tw:v1:performance:",
        "tw:v1:compare:",
        "tw:v1:allocation:",
        "tw:v1:dashboard",
        # Strategies cache (six technical strategies + VT vs VTI+VXUS
        # portfolio comparison). Was previously never busted here, so a
        # cached "insufficient data" / null result for a strategy could
        # survive up to cache_ttl_seconds after the underlying data
        # actually became available. Required for the new VT vs
        # VTI+VXUS strategy to reliably pick up VT/VXUS data as soon as
        # it's ingested, rather than waiting out the full cache TTL.
        "tw:v3:strategies:",
    ):
        await cache.delete_prefix(prefix)

    logger.info(
        "Ingestion complete (%s, run %s): %d symbols ok, %d failed, %d rows inserted",
        status,
        run_id,
        symbols_ok,
        symbols_failed,
        total_inserted,
    )


def create_scheduler() -> AsyncIOScheduler:
    """Nightly refresh at 22:30 UTC, Mon-Fri (after US market close)."""
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.add_job(
        ingest_all,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30, timezone="UTC"),
        id="nightly_eod_ingestion",
        replace_existing=True,
        max_instances=1,
    )
    return scheduler