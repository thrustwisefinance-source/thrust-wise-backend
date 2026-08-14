from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import BENCHMARK_SYMBOL
from app.database import get_db
from app.routers.deps import load_prices, require_prices, resolve_meta
from app.schemas import AllocationResult, ApiResponse, CompareResult
from app.services import analytics, cache

# Registered BEFORE the etfs router so /etfs/compare and /etfs/allocation
# win over the dynamic /etfs/{symbol} route
router = APIRouter(prefix="/etfs", tags=["compare"])


async def _compare_handler(
    symbols: str = Query(..., description="Comma-separated, e.g. VOO,SPY,QQQ"),
    db: AsyncSession = Depends(get_db),
):
    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in requested:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    requested = unique

    if not 2 <= len(requested) <= 4:
        raise HTTPException(
            status_code=422,
            detail="Provide between 2 and 4 distinct symbols to compare.",
        )

    cache_key = f"tw:v1:compare:{','.join(sorted(requested))}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    benchmark = await load_prices(db, BENCHMARK_SYMBOL)

    entries = []
    all_prices_for_corr = []
    for symbol in requested:
        meta = resolve_meta(symbol)
        prices = (
            benchmark
            if meta.symbol == BENCHMARK_SYMBOL and benchmark
            else await load_prices(db, meta.symbol)
        )
        require_prices(prices, meta.symbol)
        all_prices_for_corr.append(prices)
        snapshot = analytics.compute_snapshot(meta, prices)
        entries.append(
            {
                "symbol": meta.symbol,
                "name": meta.name,
                "issuer": meta.issuer,
                "category": meta.category,
                "price": snapshot["price"],
                "change_percent": snapshot["change_percent"],
                "expense_ratio": meta.expense_ratio,
                "risk_level": meta.risk_level,
                "dividend_yield": meta.dividend_yield,
                "aum": meta.aum_usd,
                "performance_stats": analytics.compute_performance_stats(prices),
                "risk_metrics": analytics.compute_risk_metrics(
                    prices, benchmark, is_benchmark=meta.symbol == BENCHMARK_SYMBOL
                ),
            }
        )

    # Compute correlation matrix across compared ETFs
    correlation_matrix = analytics.compute_correlation_matrix(all_prices_for_corr)

    result = {
        "entries": entries,
        "correlation_matrix": correlation_matrix,
    }

    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)


@router.get("/compare", response_model=ApiResponse[CompareResult])
async def compare_etfs(
    symbols: str = Query(..., description="Comma-separated, e.g. VOO,SPY,QQQ"),
    db: AsyncSession = Depends(get_db),
):
    return await _compare_handler(symbols=symbols, db=db)


@router.get("/allocation", response_model=ApiResponse[AllocationResult])
async def allocation_mix(
    allocations: str = Query(
        ...,
        description="Comma-separated SYMBOL:WEIGHT pairs, e.g. VOO:60,SHY:30,GLD:10",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Blended portfolio metrics for an allocation mix (rule-based)."""
    symbols: list[str] = []
    weights: list[float] = []
    for part in allocations.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise HTTPException(
                status_code=422,
                detail=f"Bad allocation '{part}' — expected SYMBOL:WEIGHT, e.g. VOO:60",
            )
        raw_symbol, raw_weight = part.split(":", 1)
        symbol = raw_symbol.strip().upper()
        try:
            weight = float(raw_weight)
        except ValueError:
            raise HTTPException(
                status_code=422, detail=f"Weight for {symbol} is not a number"
            ) from None
        if weight <= 0:
            raise HTTPException(
                status_code=422, detail=f"Weight for {symbol} must be positive"
            )
        if symbol in symbols:
            raise HTTPException(status_code=422, detail=f"Duplicate symbol {symbol}")
        symbols.append(symbol)
        weights.append(weight)

    if not 2 <= len(symbols) <= 6:
        raise HTTPException(
            status_code=422, detail="Provide between 2 and 6 allocations."
        )

    normalized_key = ",".join(f"{s}:{w:g}" for s, w in sorted(zip(symbols, weights)))
    cache_key = f"tw:v1:allocation:{normalized_key}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    all_prices = []
    for symbol in symbols:
        resolve_meta(symbol)  # 404 on unknown symbol
        prices = await load_prices(db, symbol)
        require_prices(prices, symbol)
        all_prices.append(prices)

    result = analytics.compute_allocation(symbols, weights, all_prices)
    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)
