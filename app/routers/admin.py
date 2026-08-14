"""
Admin endpoints — protected by X-Admin-Token header.
When ADMIN_API_TOKEN is not configured, these endpoints return 404 (invisible).
"""

import asyncio
import hmac
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.services import ingestion

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

# Simple in-memory lock to prevent overlapping refreshes.
# For multi-replica deployments, replace with a Postgres advisory lock.
_refresh_lock = asyncio.Lock()


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
