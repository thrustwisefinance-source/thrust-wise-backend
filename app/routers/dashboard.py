import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ETF_REGISTRY, INDEX_REGISTRY
from app.database import get_db
from app.routers.deps import load_prices
from app.schemas import ApiResponse, DashboardSummary
from app.services import analytics, cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


async def _dashboard_handler(db: AsyncSession = Depends(get_db)):
    cached = await cache.get_json("tw:v1:dashboard")
    if cached is not None:
        return ApiResponse(data=cached)

    snapshots = []
    skipped: list[str] = []
    as_of = ""
    for symbol, meta in ETF_REGISTRY.items():
        prices = await load_prices(db, symbol, limit=40)
        if not prices:
            skipped.append(symbol)
            logger.warning("No price data for %s — omitting from dashboard", symbol)
            continue
        snapshots.append(analytics.compute_snapshot(meta, prices))
        as_of = max(as_of, prices[-1].date.isoformat())

    if not snapshots:
        return ApiResponse(
            data=None,
            success=False,
            message="No price data available yet. Try again shortly.",
        )

    ranked = sorted(snapshots, key=lambda s: s["change_percent"], reverse=True)
    top, worst = ranked[0], ranked[-1]

    # Index-level widgets (S&P 500, NASDAQ, Dow) — omit any not yet ingested
    indices = []
    for index_symbol, index_meta in INDEX_REGISTRY.items():
        index_prices = await load_prices(db, index_symbol, limit=40)
        if not index_prices:
            logger.warning(
                "No index data for %s — omitting from dashboard", index_symbol
            )
            continue
        indices.append(analytics.compute_index_snapshot(index_meta, index_prices))

    summary = {
        "as_of": as_of,
        "total_etfs": len(snapshots),
        "top_performer": {
            "symbol": top["symbol"],
            "name": top["name"],
            "change_percent": top["change_percent"],
        },
        "worst_performer": {
            "symbol": worst["symbol"],
            "name": worst["name"],
            "change_percent": worst["change_percent"],
        },
        "average_expense_ratio": round(
            sum(s["expense_ratio"] for s in snapshots) / len(snapshots), 4
        ),
        "indices": indices,
        "snapshots": snapshots,
    }

    message = None
    if skipped:
        message = f"Price data unavailable for {', '.join(skipped)}; omitted"

    await cache.set_json("tw:v1:dashboard", summary)
    return ApiResponse(data=summary, message=message)


@router.get("", response_model=ApiResponse[DashboardSummary])
async def get_dashboard(db: AsyncSession = Depends(get_db)):
    return await _dashboard_handler(db=db)


@router.get("/market", response_model=ApiResponse[DashboardSummary])
async def get_dashboard_market(db: AsyncSession = Depends(get_db)):
    """Alias — frontend expects /dashboard/market."""
    return await _dashboard_handler(db=db)
