"""
Set 3 strategy endpoints (independent of routers/strategy.py).

    GET /api/etfs/{symbol}/strategies/technical_set3
    GET /api/etfs/{symbol}/strategies/quantitative_set3
    GET /api/etfs/{symbol}/strategies/regime_set3
    GET /api/etfs/{symbol}/strategies/calendar_set3
    GET /api/etfs/{symbol}/strategies/portfolio_set3

Each endpoint returns ONLY the strategies registered for its category (see
app/strategies/set_3_strategies/<category>.py). Adding a strategy never needs
a new endpoint or a change in this file's generic flow: register it in its
category module, add its schema field, and - only if it takes query
parameters - declare those parameters on the endpoint below.

Cache: ``tw:set3:{category}:{symbol}:{newest_price_date}:{fingerprint}`` -
a separate namespace per category; existing ``tw:v*`` keys are never read,
written or purged. Requests that are not deterministic (e.g. the random
simulation without a ``seed``) bypass the cache.

Existing infrastructure is reused through imports only: resolve_meta /
load_prices (routers.deps), the Redis cache (services.cache), ApiResponse,
and the global exception handlers (error envelope).
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import DailyPrice
from app.routers.deps import load_prices, resolve_meta
from app.schemas.common import ApiResponse
from app.schemas.strategy_set3 import (
    CalendarSet3Response,
    PortfolioSet3Response,
    QuantitativeSet3Response,
    RandomizedSimulationParams,
    RegimeSet3Response,
    Set3StrategyInfo,
    TechnicalSet3Response,
)
from app.services import cache
from app.strategies.set_3_strategies import CATEGORY_REGISTRIES
from app.strategies.set_3_strategies.common import (
    Set3Context,
    Set3Strategy,
    Set3StrategyUnavailable,
    build_cache_key,
    params_fingerprint,
    prices_to_frame,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/etfs", tags=["strategy-set3"])

StrategiesQuery = Annotated[
    str | None,
    Query(
        description=(
            "Comma-separated strategy keys to run (default: every strategy "
            "registered for this category)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_strategies(category: str, csv: str | None) -> list[Set3Strategy]:
    registry = CATEGORY_REGISTRIES[category]
    if not csv:
        return list(registry.values())
    keys = [k.strip() for k in csv.split(",") if k.strip()]
    unknown = [k for k in keys if k not in registry]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown {category}_set3 strategies: {', '.join(unknown)}. "
                f"Available: {', '.join(registry) or '(none registered yet)'}"
            ),
        )
    return [registry[k] for k in dict.fromkeys(keys)]


async def _latest_price_date(db: AsyncSession, symbol: str) -> datetime.date | None:
    return await db.scalar(
        select(func.max(DailyPrice.date)).where(DailyPrice.symbol == symbol)
    )


async def _load_frame(db: AsyncSession, symbol: str):
    return prices_to_frame(await load_prices(db, symbol))


def _dump(params: Any) -> Any:
    return params.model_dump(mode="json") if hasattr(params, "model_dump") else params


def _params_dict(params_by_key: dict[str, Any], key: str) -> Any:
    return params_by_key.get(key)


async def _run_category(
    category: str,
    symbol: str,
    db: AsyncSession,
    strategies_csv: str | None,
    params_by_key: dict[str, Any] | None = None,
) -> ApiResponse:
    params_by_key = params_by_key or {}
    meta = resolve_meta(symbol)  # 404 for unknown symbols (same as existing API)
    specs = _select_strategies(category, strategies_csv)
    base = {
        "symbol": meta.symbol,
        "category": category,
        "registered_strategies": [
            Set3StrategyInfo(key=s.key, title=s.title)
            for s in CATEGORY_REGISTRIES[category].values()
        ],
    }
    response_cls = _RESPONSE_MODELS[category]

    if not specs:
        return ApiResponse(
            data=response_cls(**base),
            message=f"No Set 3 {category} strategies are registered yet.",
        )

    # ---- cache lookup (symbol + extra symbols' newest dates are part of the key)
    last_date = await _latest_price_date(db, meta.symbol)
    if last_date is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Price data for {meta.symbol} has not been ingested yet. "
                "Try again shortly."
            ),
        )
    extra_symbols = sorted(
        {s for spec in specs for s in spec.required_symbols(_params_dict(params_by_key, spec.key))}
        - {meta.symbol}
    )
    extra_dates = {s: await _latest_price_date(db, s) for s in extra_symbols}
    cacheable = all(spec.cacheable(_params_dict(params_by_key, spec.key)) for spec in specs)
    cache_key = build_cache_key(
        category,
        meta.symbol,
        data_version=last_date.isoformat(),
        fingerprint=params_fingerprint(
            {
                "strategies": [s.key for s in specs],
                "params": {
                    s.key: _dump(_params_dict(params_by_key, s.key))
                    for s in specs
                    if _params_dict(params_by_key, s.key) is not None
                },
                "extra_dates": {k: str(v) for k, v in extra_dates.items()},
            }
        ),
    )

    payload = await cache.get_json(cache_key) if cacheable else None
    if payload is None:
        payload, cache_ok = await _compute_payload(
            db, meta.symbol, specs, params_by_key, extra_symbols
        )
        if cacheable and cache_ok:
            await cache.set_json(cache_key, payload)

    unavailable: dict[str, str] = payload["unavailable"]
    data = response_cls(**base, strategies=payload["strategies"], unavailable=unavailable)
    message = (
        "; ".join(f"{k}: {v}" for k, v in unavailable.items()) if unavailable else None
    )
    return ApiResponse(data=data, message=message)


async def _compute_payload(
    db: AsyncSession,
    symbol: str,
    specs: list[Set3Strategy],
    params_by_key: dict[str, Any],
    extra_symbols: list[str],
) -> tuple[dict, bool]:
    """Compute every selected strategy. One failing strategy never breaks the
    others. Returns (payload, cacheable) - transient failures are not cached."""
    frame = await _load_frame(db, symbol)
    extra = {s: await _load_frame(db, s) for s in extra_symbols}
    strategies_out: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    cache_ok = True
    for spec in specs:
        ctx = Set3Context(
            symbol=symbol,
            prices=frame,
            extra_prices=extra,
            params=_params_dict(params_by_key, spec.key),
        )
        try:
            # CPU-bound numpy work: keep it off the event loop
            strategies_out[spec.key] = await asyncio.to_thread(spec.compute, ctx)
        except Set3StrategyUnavailable as exc:
            strategies_out[spec.key] = None
            unavailable[spec.key] = str(exc)
        except Exception:  # noqa: BLE001 - never break the other strategies
            logger.exception("Set 3 strategy %s failed for %s", spec.key, symbol)
            strategies_out[spec.key] = None
            unavailable[spec.key] = "Computation failed; please try again later."
            cache_ok = False
    return {"strategies": strategies_out, "unavailable": unavailable}, cache_ok


_RESPONSE_MODELS = {
    "technical": TechnicalSet3Response,
    "quantitative": QuantitativeSet3Response,
    "regime": RegimeSet3Response,
    "calendar": CalendarSet3Response,
    "portfolio": PortfolioSet3Response,
}


# ---------------------------------------------------------------------------
# Query-parameter dependency: Randomized Investment Simulation
# (pydantic models as Query params need FastAPI >= 0.115; the repo pins 0.111,
#  so parameters are declared explicitly and validated by the model.)
# ---------------------------------------------------------------------------


def randomized_simulation_params(
    initial_capital: float = Query(10_000.0, gt=0, le=1e12, description="Starting cash per simulation"),
    start_date: datetime.date | None = Query(None, description="Historical window start (YYYY-MM-DD); default: first stored price"),
    end_date: datetime.date | None = Query(None, description="Historical window end (YYYY-MM-DD); default: last stored price"),
    buy_probability: float = Query(0.5, ge=0, le=1, description="P(buy) each day"),
    sell_probability: float = Query(0.5, ge=0, le=1, description="P(sell) each day; buy+sell must be <= 1, remainder = hold"),
    max_purchase_allocation_percent: float = Query(50.0, gt=0, le=100, description="Max % of cash spent on one buy"),
    max_sell_percent: float = Query(50.0, gt=0, le=100, description="Max % of held shares sold in one sell"),
    iterations: int = Query(100, ge=1, le=1000, description="Number of independent simulations"),
    seed: int | None = Query(None, ge=0, le=2**31 - 1, description="Random seed; supply it for reproducible (and cacheable) runs"),
    benchmark_symbol: str = Query("SPY", min_length=1, max_length=10, description="Benchmark Buy & Hold symbol (S&P 500 proxy)"),
    commission_percent: float = Query(0.05, ge=0, le=5, description="Commission as % of trade value"),
    commission_per_trade: float = Query(0.0, ge=0, le=1000, description="Flat commission per trade"),
    chart_max_points: int = Query(400, ge=10, le=2000, description="Max points in chartData"),
) -> RandomizedSimulationParams:
    try:
        return RandomizedSimulationParams(
            initial_capital=initial_capital,
            start_date=start_date,
            end_date=end_date,
            buy_probability=buy_probability,
            sell_probability=sell_probability,
            max_purchase_allocation_percent=max_purchase_allocation_percent,
            max_sell_percent=max_sell_percent,
            iterations=iterations,
            seed=seed,
            benchmark_symbol=benchmark_symbol,
            commission_percent=commission_percent,
            commission_per_trade=commission_per_trade,
            chart_max_points=chart_max_points,
        )
    except ValidationError as exc:
        msg = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        raise HTTPException(status_code=422, detail=msg) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/strategies/technical_set3",
    response_model=ApiResponse[TechnicalSet3Response],
)
async def get_technical_set3(
    symbol: str, strategies: StrategiesQuery = None, db: AsyncSession = Depends(get_db)
):
    """Set 3 technical strategies (indicators, price action, trend)."""
    return await _run_category("technical", symbol, db, strategies)


@router.get(
    "/{symbol}/strategies/quantitative_set3",
    response_model=ApiResponse[QuantitativeSet3Response],
)
async def get_quantitative_set3(
    symbol: str,
    strategies: StrategiesQuery = None,
    sim: RandomizedSimulationParams = Depends(randomized_simulation_params),
    db: AsyncSession = Depends(get_db),
):
    """Set 3 quantitative strategies, incl. the Randomized Investment Simulation
    (parameters above apply to ``randomized_investment_simulation``)."""
    return await _run_category(
        "quantitative",
        symbol,
        db,
        strategies,
        {"randomized_investment_simulation": sim},
    )


@router.get(
    "/{symbol}/strategies/regime_set3",
    response_model=ApiResponse[RegimeSet3Response],
)
async def get_regime_set3(
    symbol: str, strategies: StrategiesQuery = None, db: AsyncSession = Depends(get_db)
):
    """Set 3 market-regime strategies."""
    return await _run_category("regime", symbol, db, strategies)


@router.get(
    "/{symbol}/strategies/calendar_set3",
    response_model=ApiResponse[CalendarSet3Response],
)
async def get_calendar_set3(
    symbol: str, strategies: StrategiesQuery = None, db: AsyncSession = Depends(get_db)
):
    """Set 3 calendar / seasonality strategies."""
    return await _run_category("calendar", symbol, db, strategies)


@router.get(
    "/{symbol}/strategies/portfolio_set3",
    response_model=ApiResponse[PortfolioSet3Response],
)
async def get_portfolio_set3(
    symbol: str, strategies: StrategiesQuery = None, db: AsyncSession = Depends(get_db)
):
    """Set 3 portfolio / allocation strategies."""
    return await _run_category("portfolio", symbol, db, strategies)
