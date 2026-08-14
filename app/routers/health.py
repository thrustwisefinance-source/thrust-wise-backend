"""
Health probes: liveness (no deps) and readiness (DB + freshness check).
"""

import datetime
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import DailyPrice, IngestionRun

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Data older than this many calendar days triggers a 503 readiness failure
STALENESS_THRESHOLD_DAYS = 5


@router.get("/health/live")
async def health_live():
    """Dependency-free liveness probe — 200 if the process serves requests."""
    return {"status": "ok"}


@router.get("/health/ready")
async def health_ready(db: AsyncSession = Depends(get_db)):
    """Readiness probe: checks DB connectivity and data freshness."""
    try:
        # DB connectivity
        await db.execute(text("SELECT 1"))

        # Data freshness
        latest_date = await db.scalar(select(func.max(DailyPrice.date)))
        price_rows = await db.scalar(select(func.count(DailyPrice.id)))

        if latest_date is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "No price data ingested yet",
                    "priceRows": 0,
                    "environment": settings.environment,
                },
            )

        today = datetime.date.today()
        stale_days = (today - latest_date).days

        last_run = (
            await db.execute(
                select(IngestionRun).order_by(IngestionRun.id.desc()).limit(1)
            )
        ).scalar_one_or_none()

        result = {
            "status": "ready" if stale_days <= STALENESS_THRESHOLD_DAYS else "stale",
            "priceRows": price_rows or 0,
            "latestDate": latest_date.isoformat(),
            "staleDays": stale_days,
            "environment": settings.environment,
            "lastIngestion": (
                {
                    "status": last_run.status,
                    "trigger": last_run.trigger,
                    "startedAt": last_run.started_at.isoformat()
                    if last_run.started_at
                    else None,
                    "rowsInserted": last_run.rows_inserted,
                    "symbolsFailed": last_run.symbols_failed,
                }
                if last_run
                else None
            ),
        }

        if stale_days > STALENESS_THRESHOLD_DAYS:
            return JSONResponse(status_code=503, content=result)
        return result

    except Exception:
        logger.exception("Health ready check failed")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "Database unreachable"},
        )


# Legacy alias — keep /api/health pointing to readiness
@router.get("/api/health", tags=["health"], include_in_schema=False)
async def health_legacy(db: AsyncSession = Depends(get_db)):
    return await health_ready(db=db)
