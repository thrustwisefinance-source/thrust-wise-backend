"""
Admin endpoints — protected by X-Admin-Token header.
When ADMIN_API_TOKEN is not configured, these endpoints return 404 (invisible).
"""

import asyncio
import hmac
import logging

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import AsyncSessionLocal
from app.scanners import ingestion as scanner_ingestion
from app.scanners import universe as scanner_universe
from app.services import ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Simple in-memory lock to prevent overlapping refreshes.
# For multi-replica deployments, replace with a Postgres advisory lock.
_refresh_lock = asyncio.Lock()

# Separate lock for scanner refreshes (universe + fundamentals + stock
# prices) so a long-running scanner refresh never blocks, or is blocked
# by, the unrelated ETF/index nightly refresh above.
_scanner_refresh_lock = asyncio.Lock()


def _verify_admin_token(request: Request) -> None:
    """Verify the admin token. Raises 404 if unconfigured, 404 if wrong (don't reveal existence)."""
    if not settings.admin_api_token:
        raise HTTPException(status_code=404, detail="Not found")
    token = request.headers.get("X-Admin-Token", "")
    if not hmac.compare_digest(token, settings.admin_api_token):
        raise HTTPException(status_code=404, detail="Not found")


@router.post("/refresh")
async def trigger_refresh(request: Request):
    """Manually trigger a data refresh. Admin-only, rate-limited."""
    _verify_admin_token(request)

    if _refresh_lock.locked():
        return JSONResponse(
            status_code=409,
            content={
                "data": None,
                "success": False,
                "message": "Ingestion run already in progress",
            },
        )

    async def _run_refresh():
        async with _refresh_lock:
            try:
                await ingestion.ingest_all(trigger="manual")
                logger.info("Admin-triggered refresh completed successfully")
            except Exception:
                logger.exception("Admin-triggered refresh failed")

    asyncio.create_task(_run_refresh())

    return JSONResponse(
        status_code=202,
        content={
            "data": {"status": "started"},
            "success": True,
            "message": "Ingestion started",
        },
    )


@router.post("/scanners/refresh")
async def trigger_scanner_refresh(
    request: Request,
    universe: str = Query(default="sp500", description="Universe key to refresh, e.g. 'sp500'."),
):
    """Manually (re)ingest one scanner universe: constituent list,
    fundamentals snapshots, and stock price history — then bust that
    universe's scanner cache. Admin-only, same auth convention as
    POST /admin/refresh. Separate from that endpoint because it refreshes
    a completely different dataset (scanner stock universe vs.
    ThrustWise's ETF/index registry) — see app/scanners/ingestion.py.
    """
    _verify_admin_token(request)

    if scanner_universe.get_universe_source(universe) is None:
        raise HTTPException(status_code=404, detail=f"Unknown universe: {universe}")

    if _scanner_refresh_lock.locked():
        return JSONResponse(
            status_code=409,
            content={
                "data": None,
                "success": False,
                "message": "A scanner refresh is already in progress",
            },
        )

    async def _run_scanner_refresh():
        async with _scanner_refresh_lock:
            try:
                async with AsyncSessionLocal() as session:
                    result = await scanner_ingestion.refresh_universe_and_data(
                        session, universe
                    )
                logger.info("Admin-triggered scanner refresh completed: %s", result)
            except Exception:
                logger.exception("Admin-triggered scanner refresh failed")

    asyncio.create_task(_run_scanner_refresh())

    return JSONResponse(
        status_code=202,
        content={
            "data": {"status": "started", "universe": universe},
            "success": True,
            "message": f"Scanner refresh started for universe '{universe}'",
        },
    )
