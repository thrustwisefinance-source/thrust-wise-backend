"""
Additive Set 3 quantitative ETF strategies.

    GET /api/etfs/{symbol}/strategies/quantitative_set3

Adds two strategies next to the existing ``randomized_investment_simulation``:

    asymmetric_volatility_persistence   (Quantitative -> Risk Management)
    fed_model_relative_valuation        (Quantitative -> Asset Allocation /
                                         Relative Valuation)

HOW THIS COEXISTS WITH app/routers/strategy_set3.py
The repository's registry-based Set 3 router already serves this exact path
(and the other four ``*_set3`` category paths, which stay untouched and keep
answering). A fixed-field response model there cannot carry new strategies
without editing existing files, which this delivery must not do. So this router
is included BEFORE the registry router in app/main.py and answers the
quantitative path with a strict superset: the same envelope, the unchanged
``randomizedInvestmentSimulation`` entry (computed by the existing code, same
parameters, same cache) plus the two new strategies.

Every strategy is computed independently and error-isolated: one failing never
removes another's result. Cache keys are
``tw:set3:quantitative:{SYMBOL}:{newest_price_date}:{fingerprint}``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import DailyPrice
from app.routers import strategy_set3 as registry_router
from app.routers.deps import resolve_meta
from app.routers.strategy_set3 import randomized_simulation_params
from app.schemas.common import ApiResponse
from app.schemas.strategies_set3 import (
    QUANTITATIVE_SET3_STRATEGY_KEYS,
    RANDOMIZED_STRATEGY_ID,
    AsymmetricVolatilityParams,
    FedModelParams,
    QuantitativeSet3ExtendedResponse,
    QuantitativeSet3ExtendedStrategies,
)
from app.schemas.strategy_set3 import RandomizedSimulationParams, Set3StrategyInfo
from app.strategies.set_3_strategies.base import run_isolated
from app.strategies.set_3_strategies.quantitative_models import (
    asymmetric_volatility_persistence as avp,
)
from app.strategies.set_3_strategies.quantitative_models import (
    fed_model_relative_valuation as fed,
)
from app.strategies.set_3_strategies.schemas.quantitative import (
    AVP_STRATEGY_ID,
    FED_STRATEGY_ID,
    AsymmetricVolatilityResult,
    FedModelResult,
)
from app.strategies.set_3_strategies.services import cache_service, data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/etfs", tags=["strategy-set3-quantitative"])

CATEGORY = "quantitative"
# The Fed Model uses live provider data (P/E snapshot + Treasury series), so its
# cached result expires sooner than the price-date-keyed AVP result.
FED_CACHE_TTL_SECONDS = 6 * 3600

_TITLES = {
    RANDOMIZED_STRATEGY_ID: "Randomized Investment Simulation (Antifragility Experiment)",
    AVP_STRATEGY_ID: avp.DESCRIPTOR.name,
    FED_STRATEGY_ID: fed.DESCRIPTOR.name,
}

StrategiesQuery = Annotated[
    str | None,
    Query(
        description=(
            "Comma-separated strategy keys to run (default: all of "
            + ", ".join(QUANTITATIVE_SET3_STRATEGY_KEYS)
            + ")."
        )
    ),
]


# ---------------------------------------------------------------------------
# Query-parameter dependencies (validated by the pydantic parameter models)
# ---------------------------------------------------------------------------


def _validation_error(exc: ValidationError) -> HTTPException:
    msg = "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg'].removeprefix('Value error, ')}"
        for e in exc.errors()
    )
    return HTTPException(status_code=422, detail=msg)


def avp_params(
    avp_return_type: Literal["log", "simple"] = Query("log", description="Return definition"),
    avp_lookback_days: int | None = Query(None, ge=90, le=36500, description="Calendar-day window ending at the newest price; default all stored history"),
    avp_vol_window: int = Query(21, ge=2, le=252, description="Rolling volatility window (trading days)"),
    avp_acf_max_lag: int = Query(20, ge=1, le=100, description="Maximum lag for the squared-return autocorrelation / Ljung-Box test"),
    avp_shock_horizon: int = Query(5, ge=1, le=60, description="Days of forward realised volatility measured after each shock"),
    avp_shock_threshold_sigma: float = Query(0.0, ge=0, le=3, description="Shock size in standard deviations; 0 = any negative / positive return"),
    avp_annualization_factor: int = Query(252, ge=1, le=366, description="Trading days per year used to annualise volatility"),
    avp_garch_enabled: bool = Query(True, description="Estimate GARCH / GJR-GARCH when enough data exist"),
    avp_garch_model: Literal["garch", "gjr"] = Query("gjr", description="gjr = asymmetric GJR-GARCH(1,1); garch = symmetric GARCH(1,1)"),
    avp_garch_max_observations: int = Query(2500, ge=250, le=6000, description="Most recent returns used for GARCH estimation"),
    avp_series_points: int = Query(500, ge=10, le=5000, description="Recent observations returned in historicalSeries"),
    avp_chart_max_points: int = Query(400, ge=10, le=2000, description="Max points of the full-history volatility chart"),
) -> AsymmetricVolatilityParams:
    try:
        return AsymmetricVolatilityParams(
            return_type=avp_return_type,
            lookback_days=avp_lookback_days,
            vol_window=avp_vol_window,
            acf_max_lag=avp_acf_max_lag,
            shock_horizon=avp_shock_horizon,
            shock_threshold_sigma=avp_shock_threshold_sigma,
            annualization_factor=avp_annualization_factor,
            garch_enabled=avp_garch_enabled,
            garch_model=avp_garch_model,
            garch_max_observations=avp_garch_max_observations,
            series_points=avp_series_points,
            chart_max_points=avp_chart_max_points,
        )
    except ValidationError as exc:
        raise _validation_error(exc) from exc


def fed_params(
    fed_lookback_years: int = Query(5, ge=1, le=40, description="Years of Treasury-yield history"),
    fed_pe_basis: Literal["trailing", "forward"] = Query("trailing", description="Which ETF-level P/E to invert"),
    fed_neutral_band_pct: float = Query(0.25, ge=0, le=5, description="|spread| (percentage points) below which yields are 'approximately equal'"),
    fed_chart_max_points: int = Query(400, ge=10, le=2000, description="Max points in chartData"),
    fed_include_granger: bool = Query(True, description="Include the optional Granger-causality availability block"),
) -> FedModelParams:
    try:
        return FedModelParams(
            lookback_years=fed_lookback_years,
            pe_basis=fed_pe_basis,
            neutral_band_pct=fed_neutral_band_pct,
            chart_max_points=fed_chart_max_points,
            include_granger=fed_include_granger,
        )
    except ValidationError as exc:
        raise _validation_error(exc) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select(csv: str | None) -> list[str]:
    if not csv:
        return list(QUANTITATIVE_SET3_STRATEGY_KEYS)
    keys = [k.strip() for k in csv.split(",") if k.strip()]
    unknown = [k for k in keys if k not in QUANTITATIVE_SET3_STRATEGY_KEYS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown quantitative_set3 strategies: {', '.join(unknown)}. "
                f"Available: {', '.join(QUANTITATIVE_SET3_STRATEGY_KEYS)}"
            ),
        )
    return list(dict.fromkeys(keys))


async def _newest_price_date(db: AsyncSession, symbol: str) -> datetime.date | None:
    return await db.scalar(
        select(func.max(DailyPrice.date)).where(DailyPrice.symbol == symbol)
    )


async def _run_avp(
    db: AsyncSession, symbol: str, params: AsymmetricVolatilityParams
) -> AsymmetricVolatilityResult:
    async def run() -> AsymmetricVolatilityResult:
        newest = await _newest_price_date(db, symbol)
        if newest is None:
            return avp.unavailable_result(
                symbol, params,
                f"Price data for {symbol} has not been ingested yet. Try again shortly.",
                observations=0,
            )
        key = cache_service.strategy_cache_key(
            CATEGORY, symbol, newest.isoformat(),
            cache_service.strategy_fingerprint(AVP_STRATEGY_ID, params, avp.CALCULATION_VERSION),
        )
        cached = await cache_service.get_cached(key)
        if cached is not None:
            try:
                return AsymmetricVolatilityResult.model_validate(cached)
            except Exception:  # noqa: BLE001 - stale / incompatible entry: recompute
                logger.warning("Ignoring unreadable Set 3 cache entry %s", key)
        frame = await data_service.load_etf_price_frame(db, symbol)
        result = await asyncio.to_thread(
            avp.compute_asymmetric_volatility_persistence, symbol, frame, params
        )
        if result.calculation_status in ("ok", "partial"):
            await cache_service.set_cached(key, result.model_dump(mode="json", by_alias=False))
        return result

    return await run_isolated(
        avp.DESCRIPTOR, run,
        lambda msg: avp.unavailable_result(symbol, params, msg, status="error", error=msg),
    )


async def _run_fed(
    db: AsyncSession, meta, params: FedModelParams
) -> FedModelResult:
    symbol = meta.symbol

    async def run() -> FedModelResult:
        profile = fed.EtfProfile(
            symbol=symbol, name=meta.name, tags=tuple(meta.tags), asset_class=meta.asset_class
        )
        applicable, _ = fed.check_equity_applicability(profile)
        if not applicable:  # deterministic, no network, nothing to cache
            return await fed.run_fed_model_relative_valuation(symbol, profile, params)
        today = data_service.utc_today()
        newest = await _newest_price_date(db, symbol)
        key = cache_service.strategy_cache_key(
            CATEGORY, symbol, (newest or today).isoformat(),
            cache_service.strategy_fingerprint(
                FED_STRATEGY_ID, params, fed.CALCULATION_VERSION,
                extra={"snapshot_date": today.isoformat()},
            ),
        )
        cached = await cache_service.get_cached(key)
        if cached is not None:
            try:
                return FedModelResult.model_validate(cached)
            except Exception:  # noqa: BLE001
                logger.warning("Ignoring unreadable Set 3 cache entry %s", key)
        result = await fed.run_fed_model_relative_valuation(symbol, profile, params)
        if result.calculation_status == "ok":
            await cache_service.set_cached(
                key, result.model_dump(mode="json", by_alias=False), ttl=FED_CACHE_TTL_SECONDS
            )
        return result

    return await run_isolated(
        fed.DESCRIPTOR, run,
        lambda msg: fed.unavailable_result(symbol, params, msg, status="error", error=msg),
    )


async def _run_randomized(
    db: AsyncSession, symbol: str, sim: RandomizedSimulationParams
) -> tuple[object | None, str | None]:
    """Delegate to the EXISTING Set 3 implementation (same code, cache, output)."""
    try:
        resp = await registry_router._run_category(  # noqa: SLF001 - reuse, don't fork
            CATEGORY, symbol, db, RANDOMIZED_STRATEGY_ID, {RANDOMIZED_STRATEGY_ID: sim}
        )
    except HTTPException as exc:
        return None, str(exc.detail)
    except Exception:  # noqa: BLE001 - isolation boundary
        logger.exception("Set 3 randomized_investment_simulation failed for %s", symbol)
        return None, "Computation failed; please try again later."
    reason = resp.data.unavailable.get(RANDOMIZED_STRATEGY_ID)
    return resp.data.strategies.randomized_investment_simulation, reason


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/strategies/quantitative_set3",
    response_model=ApiResponse[QuantitativeSet3ExtendedResponse],
)
async def get_quantitative_set3_with_articles(
    symbol: str,
    strategies: StrategiesQuery = None,
    sim: RandomizedSimulationParams = Depends(randomized_simulation_params),
    avp_p: AsymmetricVolatilityParams = Depends(avp_params),
    fed_p: FedModelParams = Depends(fed_params),
    db: AsyncSession = Depends(get_db),
):
    """Set 3 quantitative strategies: Randomized Investment Simulation (existing),
    Asymmetric Volatility Persistence and Fed Model Relative Valuation (new)."""
    meta = resolve_meta(symbol)  # 404 for unknown symbols, as everywhere else
    selected = _select(strategies)
    out = QuantitativeSet3ExtendedStrategies()
    unavailable: dict[str, str] = {}

    # Sequential on purpose: they share one AsyncSession (not safe for concurrent use).
    # Heavy numeric work runs in worker threads inside each strategy.
    if AVP_STRATEGY_ID in selected:
        out.asymmetric_volatility_persistence = await _run_avp(db, meta.symbol, avp_p)
    if FED_STRATEGY_ID in selected:
        out.fed_model_relative_valuation = await _run_fed(db, meta, fed_p)
    if RANDOMIZED_STRATEGY_ID in selected:
        value, reason = await _run_randomized(db, meta.symbol, sim)
        out.randomized_investment_simulation = value  # type: ignore[assignment]
        if reason:
            unavailable[RANDOMIZED_STRATEGY_ID] = reason

    for key, res in (
        (AVP_STRATEGY_ID, out.asymmetric_volatility_persistence),
        (FED_STRATEGY_ID, out.fed_model_relative_valuation),
    ):
        # No calculated numbers (unavailable / not applicable / error / partial-without-
        # core-values) -> tell the client why, same convention as the existing router.
        if res is not None and res.calculated_values is None:
            reason = res.error or res.data_availability.message
            if reason:
                unavailable[key] = reason

    data = QuantitativeSet3ExtendedResponse(
        symbol=meta.symbol,
        category=CATEGORY,
        registered_strategies=[
            Set3StrategyInfo(key=k, title=_TITLES[k]) for k in QUANTITATIVE_SET3_STRATEGY_KEYS
        ],
        strategies=out,
        unavailable=unavailable,
    )
    message = "; ".join(f"{k}: {v}" for k, v in unavailable.items()) or None
    return ApiResponse(data=data, message=message)
