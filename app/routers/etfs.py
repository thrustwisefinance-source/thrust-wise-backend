import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import BENCHMARK_SYMBOL, ETF_REGISTRY, FILTER_CATEGORIES
from app.database import get_db
from app.routers.deps import load_prices, require_prices, resolve_meta
from app.schemas import ApiResponse, EtfDetails, EtfQuote, EtfSnapshot, PerformancePoint
from app.services import analytics, cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/etfs", tags=["etfs"])

# Enough history for a 30-point sparkline plus previous-close change
SNAPSHOT_LOOKBACK_ROWS = 40


@router.get("", response_model=ApiResponse[list[EtfSnapshot]])
async def list_etfs(
    category: str | None = Query(default=None, description="Explorer filter chip"),
    db: AsyncSession = Depends(get_db),
):
    """ETF Explorer cards, optionally filtered by chip (Equity, Gold, ...)."""
    # Validate category against known filter chips
    if category and category not in FILTER_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{category}'. Valid values: {', '.join(FILTER_CATEGORIES)}",
        )

    cache_key = f"tw:v1:snapshot:all:{category or 'All'}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    snapshots = []
    skipped: list[str] = []
    for symbol, meta in ETF_REGISTRY.items():
        if category and category != "All" and category not in meta.tags:
            continue
        prices = await load_prices(db, symbol, limit=SNAPSHOT_LOOKBACK_ROWS)
        if not prices:
            skipped.append(symbol)
            logger.warning("No price data for %s — omitting from Explorer list", symbol)
            continue
        snapshots.append(analytics.compute_snapshot(meta, prices))

    message = None
    if skipped:
        message = (
            f"Price data unavailable for {', '.join(skipped)}; omitted from results"
        )

    await cache.set_json(cache_key, snapshots)
    return ApiResponse(data=snapshots, message=message)


@router.get("/{symbol}", response_model=ApiResponse[EtfDetails])
async def get_etf_details(symbol: str, db: AsyncSession = Depends(get_db)):
    meta = resolve_meta(symbol)

    cache_key = f"tw:v1:details:{meta.symbol}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices = await load_prices(db, meta.symbol)
    require_prices(prices, meta.symbol)
    benchmark = (
        prices
        if meta.symbol == BENCHMARK_SYMBOL
        else await load_prices(db, BENCHMARK_SYMBOL)
    )

    details = analytics.compute_details(meta, prices, benchmark)
    await cache.set_json(cache_key, details)
    return ApiResponse(data=details)


@router.get("/{symbol}/quote", response_model=ApiResponse[EtfQuote])
async def get_etf_quote(symbol: str, db: AsyncSession = Depends(get_db)):
    meta = resolve_meta(symbol)

    cache_key = f"tw:v1:quote:{meta.symbol}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices = await load_prices(db, meta.symbol, limit=2)
    require_prices(prices, meta.symbol)

    quote = analytics.compute_quote(meta, prices)
    await cache.set_json(cache_key, quote)
    return ApiResponse(data=quote)


@router.get(
    "/{symbol}/performance",
    response_model=ApiResponse[dict[str, list[PerformancePoint]]],
)
async def get_etf_performance(symbol: str, db: AsyncSession = Depends(get_db)):
    """Chart series grouped by range: {"1M": [...], ..., "5Y": [...]}."""
    meta = resolve_meta(symbol)

    cache_key = f"tw:v1:performance:{meta.symbol}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices = await load_prices(db, meta.symbol)
    require_prices(prices, meta.symbol)

    series = analytics.compute_performance_series(prices)
    await cache.set_json(cache_key, series)
    return ApiResponse(data=series)
