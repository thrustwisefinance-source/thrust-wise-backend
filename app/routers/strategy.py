"""
Strategy Analytics endpoint.

Analytical / visualization only: returns computed indicator values and
descriptive flags for pre-defined technical strategies. This is NOT a
trading bot and does NOT return buy/sell/hold signals, recommendations,
or anything ML-derived.
"""

import datetime
import hashlib
import json
import logging
import time
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.constants import ETF_REGISTRY, PORTFOLIO_COMPARISON_REGISTRY, STRATEGY_ONLY_REGISTRY
from app.database import get_db
from app.models import StockUniverseMember
from app.routers.deps import load_prices, require_prices, resolve_meta
from app.scanners import data as scanner_data
from app.scanners import universe as scanner_universe
from app.schemas import (
    ApiResponse,
    DiversificationHeatMapRequest,
    DiversificationHeatMapStrategy,
    EmaBollingerAtrStrategy,
    PortfolioOptimizationRequest,
    PortfolioOptimizationResult,
    SeasonalHeatMapStrategy,
    StrategyAnalytics,
)
from app.services import cache, eodhd_client, portfolio_comparison, regime, strategies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/etfs", tags=["strategy-analytics"])

# Portfolio Optimization (Kelly Criterion + Mean-Variance Optimization,
# see services.strategies.compute_portfolio_optimization) is genuinely
# multi-asset — the one dedicated endpoint the reorg spec calls for,
# separate from the single-symbol /etfs/{symbol}/strategies route above.
portfolio_router = APIRouter(prefix="/portfolio", tags=["portfolio-optimization"])

# Symbol under which the VT vs VTI+VXUS portfolio comparison is surfaced
# ("VTI tile" placement per product request). VTI's own price list (already
# loaded below for the six technical strategies) is reused; VT and VXUS are
# portfolio-comparison-only symbols (see
# app.constants.PORTFOLIO_COMPARISON_REGISTRY) fetched directly by plain
# symbol — they are not Explorer products, so they are intentionally not
# passed through resolve_meta.
PORTFOLIO_COMPARISON_SYMBOL = "VTI"

# Risk-On/Risk-Off is cross-ETF and expensive (pairwise DTW + clustering
# across the whole universe), and its result doesn't depend on which
# symbol the frontend is currently viewing. It is cached separately, once,
# under its own key — see _load_market_regime below — rather than being
# recomputed as part of every single symbol's /strategies request.
MARKET_REGIME_CACHE_KEY = "tw:v1:regime:market"

# TLT Monthly Cycle is calendar-based off TLT's own price history, not the
# symbol currently being viewed — cross-cutting like Risk-On/Risk-Off, so
# it's computed once and cached separately rather than recomputed per
# symbol. See app.constants.STRATEGY_ONLY_REGISTRY.
TLT_MONTHLY_CYCLE_CACHE_KEY = "tw:v1:strategy:tlt_monthly_cycle"
TLT_SYMBOL = "TLT"

# TQQQ / TMF / IEF Rebalancing is a multi-ETF portfolio-allocation strategy,
# not tied to whichever symbol's page the caller is viewing — same
# cross-cutting treatment as TLT Monthly Cycle and Risk-On/Risk-Off above:
# computed once from TQQQ/TMF/IEF's own price history and cached
# separately rather than recomputed per symbol. See
# app.constants.STRATEGY_ONLY_REGISTRY and
# services.strategies.compute_tqqq_tmf_ief_rebalancing (Strategy 11).
TQQQ_TMF_IEF_CACHE_KEY = "tw:v1:strategy:tqqq_tmf_ief_rebalancing"
TQQQ_SYMBOL = "TQQQ"
TMF_SYMBOL = "TMF"
IEF_SYMBOL = "IEF"

# Mswing Momentum's relative-strength leg needs the index's (QQQ's) own
# Mswing reading. QQQ is already part of ETF_REGISTRY, so — same
# cross-cutting treatment as TLT Monthly Cycle and TQQQ/TMF/IEF above —
# it's computed once and cached separately rather than recomputed per
# symbol (avoiding an extra DB round-trip on every /strategies request).
MSWING_INDEX_CACHE_KEY = "tw:v1:strategy:mswing_index_value"

# One-Month Factor Momentum is cross-ETF (it ranks the existing
# ETF_REGISTRY category tags by their previous-month average return) and
# its result doesn't depend on which symbol the frontend is currently
# viewing — same cross-cutting treatment as Risk-On/Risk-Off, TLT Monthly
# Cycle, and TQQQ/TMF/IEF Rebalancing above. See
# services.strategies.compute_factor_momentum (Strategy 14).
FACTOR_MOMENTUM_CACHE_KEY = "tw:v1:strategy:factor_momentum"


async def _load_mswing_index_value(db: AsyncSession) -> float | None:
    index_prices = await load_prices(db, strategies.MSWING_INDEX_SYMBOL)
    return strategies.compute_mswing_index_value(index_prices)


async def _load_tlt_monthly_cycle(db: AsyncSession) -> dict | None:
    tlt_prices = await load_prices(db, TLT_SYMBOL)
    return strategies.compute_tlt_monthly_cycle(tlt_prices)


async def _load_tqqq_tmf_ief_rebalancing(db: AsyncSession) -> dict | None:
    tqqq_prices = await load_prices(db, TQQQ_SYMBOL)
    tmf_prices = await load_prices(db, TMF_SYMBOL)
    ief_prices = await load_prices(db, IEF_SYMBOL)
    return strategies.compute_tqqq_tmf_ief_rebalancing(
        tqqq_prices=tqqq_prices, tmf_prices=tmf_prices, ief_prices=ief_prices
    )


async def _load_factor_momentum(db: AsyncSession) -> dict | None:
    """Load each ETF_REGISTRY symbol's own daily price history and run
    the One-Month Factor Momentum ranking once. Only called on a cache
    miss for FACTOR_MOMENTUM_CACHE_KEY (via cache.get_or_compute below),
    same pattern as _load_market_regime above.
    """
    prices_by_symbol = {
        registry_symbol: await load_prices(db, registry_symbol)
        for registry_symbol in ETF_REGISTRY
    }
    return strategies.compute_factor_momentum(prices_by_symbol)


async def _load_market_regime(db: AsyncSession) -> dict | None:
    """Load weekly price history for the whole ETF universe and run the
    Risk-On/Risk-Off computation once. Only called on a cache miss for
    MARKET_REGIME_CACHE_KEY (via cache.get_or_compute below), so the
    price history is not re-downloaded and DTW/clustering does not
    re-run on every /strategies request.
    """
    prices_by_symbol = {
        registry_symbol: await load_prices(db, registry_symbol)
        for registry_symbol in ETF_REGISTRY
    }
    return regime.compute_market_regime(prices_by_symbol)


# ---------------------------------------------------------------------------
# Quantitative valuation strategies 27-28: Graham valuation + Greenblatt
# Magic Formula (services.strategies). Both ride on the existing
# /etfs/{symbol}/strategies response (`quantitative_strategies` group):
#   - Greenblatt is a stock-UNIVERSE ranking, so - like Factor Momentum - it
#     is computed once, cached separately and attached to every response.
#   - Graham is per-stock: for a stock symbol (a current member of an
#     ingested stock universe) the same route returns the valuation
#     strategies for that stock; for an ETF it reports "unavailable".
# The cache key lives under the scanner cache prefix so a scanner refresh
# (which re-ingests fundamentals) busts it automatically.
# ---------------------------------------------------------------------------

VALUATION_DEFAULT_UNIVERSE = "sp500"
GREENBLATT_CACHE_KEY_PREFIX = "tw:v1:scanner:greenblatt:"
GRAHAM_NOT_APPLICABLE_ETF = (
    "Graham valuation applies to individual stocks, not ETFs. Request a stock "
    "symbol from an ingested universe (e.g. AAPL) on this same route."
)


class _ValuationNotReady(Exception):
    """Raised inside the cached compute so an 'unavailable' result is never cached."""


async def _find_stock_member(db: AsyncSession, symbol: str) -> StockUniverseMember | None:
    return await db.scalar(
        select(StockUniverseMember)
        .where(
            StockUniverseMember.symbol == symbol.strip().upper(),
            StockUniverseMember.is_current.is_(True),
        )
        .order_by(StockUniverseMember.universe.asc())
        .limit(1)
    )


def _unavailable_magic_formula(universe_key: str, reasons: list[str]) -> dict:
    return {
        "status": strategies.UNAVAILABLE,
        "unavailable_reasons": reasons,
        "universe": universe_key,
        "methodology": strategies.MAGIC_FORMULA_METHODOLOGY,
        "disclaimer": strategies.MAGIC_FORMULA_DISCLAIMER,
    }


async def _run_greenblatt(db: AsyncSession, universe_key: str) -> dict:
    """Universe-wide Magic Formula screen (the expensive compute wrapped by
    cache.get_or_compute). One DB query for the members, one for all
    fundamentals snapshots - no per-stock queries or provider calls."""
    members = await scanner_universe.get_stock_universe(db, universe_key)
    if not members:
        raise _ValuationNotReady(
            f"Universe '{universe_key}' has not been ingested yet "
            "(POST /api/admin/scanners/refresh)."
        )
    snapshots = await scanner_data.load_fundamentals_snapshots(db, [m.symbol for m in members])
    candidates = []
    for m in members:
        snapshot = snapshots.get(m.symbol)
        candidates.append(
            scanner_data.build_magic_formula_candidate(
                m.symbol,
                (snapshot.company_name if snapshot else None) or m.name,
                (snapshot.sector if snapshot else None) or m.sector,
                (snapshot.industry if snapshot else None) or m.industry,
                scanner_data.valuation_inputs_from_snapshot(snapshot),
            )
        )
    screen = strategies.screen_magic_formula(candidates, universe_key)
    if not screen["ranked_count"] and screen["excluded_by_reason"].get("no_fundamentals_data") == len(members):
        raise _ValuationNotReady(
            "Valuation inputs (EBIT, balance sheet, enterprise value) have not been "
            "ingested yet; run a scanner fundamentals refresh."
        )
    screen["data_updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return screen


async def _load_magic_formula_view(
    db: AsyncSession, universe_key: str, requested_symbol: str | None
) -> dict:
    """Greenblatt response object: top-N rows + the requested stock's own
    row. Never raises - provider/DB trouble becomes an explicit
    'unavailable' object (logged), so it can't break the rest of the response."""
    try:
        screen = await cache.get_or_compute(
            f"{GREENBLATT_CACHE_KEY_PREFIX}{universe_key}",
            lambda: _run_greenblatt(db, universe_key),
        )
    except _ValuationNotReady as exc:
        return _unavailable_magic_formula(universe_key, [str(exc)])
    except Exception:  # must not break the strategies response
        logger.exception("Greenblatt Magic Formula computation failed")
        return _unavailable_magic_formula(
            universe_key, ["Magic Formula computation failed; see server logs."]
        )

    if not isinstance(screen, dict) or "ranked" not in screen:
        return _unavailable_magic_formula(
            universe_key, ["Magic Formula result is unavailable."]
        )

    source = scanner_universe.get_universe_source(universe_key)
    ranked = screen["ranked"]
    mine = None
    if requested_symbol:
        mine = next(
            (r for r in ranked + screen["unranked"] if r.get("symbol") == requested_symbol), None
        )
    warnings = []
    no_data = screen["excluded_by_reason"].get("no_fundamentals_data", 0)
    if no_data:
        warnings.append(
            f"{no_data} universe member(s) have no ingested valuation inputs yet and are not ranked."
        )
    return {
        "status": screen["status"],
        "unavailable_reasons": [] if screen["status"] == strategies.AVAILABLE else [
            "No stock in the universe has complete, valid inputs for both factors."
        ],
        "universe": universe_key,
        "universe_label": source.label if source else universe_key,
        "universe_size": screen["universe_size"],
        "eligible_universe_definition": (
            f"Current members of the '{universe_key}' stock universe, excluding financials and "
            "utilities and any stock with missing/invalid EBIT, enterprise value or invested capital."
        ),
        "ranked_count": screen["ranked_count"],
        "excluded_count": screen["excluded_count"],
        "excluded_by_reason": screen["excluded_by_reason"],
        "top_n": strategies.MAGIC_FORMULA_RESPONSE_TOP_N,
        "top_ranked": ranked[: strategies.MAGIC_FORMULA_RESPONSE_TOP_N],
        "requested_symbol_result": mine,
        "methodology": strategies.MAGIC_FORMULA_METHODOLOGY,
        "data_updated_at": screen.get("data_updated_at"),
        "data_quality_warnings": warnings,
        "disclaimer": strategies.MAGIC_FORMULA_DISCLAIMER,
    }


def _graham_unavailable(symbol: str, reasons: list[str]) -> dict:
    out = strategies.compute_graham_valuation(eps=None, growth_percent=None, aaa_yield_percent=None)
    out["unavailable_reasons"] = reasons
    out["data_quality_warnings"] = []
    out["symbol"] = symbol
    return out


async def _compute_graham_for_stock(db: AsyncSession, symbol: str) -> dict:
    """Graham valuation for one stock from its stored fundamentals snapshot,
    its latest ingested close, and the operator-configured AAA yield."""
    try:
        snapshot = (await scanner_data.load_fundamentals_snapshots(db, [symbol])).get(symbol)
        inputs = scanner_data.valuation_inputs_from_snapshot(snapshot) or {}
        latest = await load_prices(db, symbol, limit=1)
    except Exception:
        logger.exception("Graham valuation input load failed for %s", symbol)
        return _graham_unavailable(symbol, ["Valuation inputs could not be loaded; see server logs."])

    warnings: list[str] = []
    if snapshot is None:
        warnings.append("No fundamentals snapshot on file for this symbol.")

    # EPS: trailing-twelve-month EPS if present, else latest annual diluted EPS.
    eps = inputs.get("trailing_eps")
    eps_period = eps_basis = None
    if eps is not None:
        stamp = (inputs.get("extracted_at") or "")[:10] or "unknown date"
        eps_period = f"TTM (as reported by Yahoo Finance on {stamp})"
        eps_basis = "Trailing-twelve-month EPS (Yahoo Finance trailingEps)"
    else:
        annual = scanner_data.annual_eps_history_from_snapshot(snapshot)
        if annual:
            eps = annual[-1].get("eps")
            eps_period = f"FY{annual[-1].get('year')}"
            eps_basis = "Latest annual diluted EPS (annual income statement)"

    # Growth: forward analyst estimate if present, else a LABELLED historical proxy.
    growth = inputs.get("forward_growth") or {}
    growth_percent = growth.get("percent")
    growth_basis = strategies.GRAHAM_GROWTH_BASIS_FORWARD
    growth_source, growth_horizon = growth.get("source"), growth.get("horizon")
    if growth_percent is None:
        proxy = scanner_data.historical_eps_cagr_percent(
            scanner_data.annual_eps_history_from_snapshot(snapshot)
        )
        if proxy is not None:
            growth_percent = proxy["percent"]
            growth_basis = strategies.GRAHAM_GROWTH_BASIS_HISTORICAL
            growth_source = "ThrustWise calculation: historical annual diluted-EPS CAGR (not a forecast)"
            growth_horizon = f"historical FY{proxy['from_year']}-FY{proxy['to_year']}"
        else:
            growth_basis = None

    price = latest[-1].close if latest else None
    price_as_of = f"{latest[-1].date.isoformat()} (end-of-day close)" if latest else None

    out = strategies.compute_graham_valuation(
        eps=eps,
        eps_period=eps_period,
        eps_basis=eps_basis,
        growth_percent=growth_percent,
        growth_basis=growth_basis,
        growth_source=growth_source,
        growth_horizon=growth_horizon,
        aaa_yield_percent=settings.graham_aaa_yield_percent,
        aaa_yield_source=settings.graham_aaa_yield_source if settings.graham_aaa_yield_percent is not None else None,
        aaa_yield_as_of=settings.graham_aaa_yield_as_of or None,
        price=price,
        price_as_of=price_as_of,
        margin_of_safety_percent=settings.graham_margin_of_safety_percent,
        extra_warnings=warnings,
    )
    if settings.graham_aaa_yield_percent is not None and not settings.graham_aaa_yield_as_of:
        out["data_quality_warnings"].append("AAA yield observation date is not configured.")
    if settings.graham_aaa_yield_percent is None:
        out["unavailable_reasons"] = [
            r.replace("AAA corporate bond yield is missing or not a finite number.",
                      "AAA corporate bond yield is not configured (set GRAHAM_AAA_YIELD_PERCENT).")
            for r in out["unavailable_reasons"]
        ]
    out["symbol"] = symbol
    return out


def _with_valuation_strategies(result: dict, graham: dict, magic_formula: dict) -> dict:
    """Copy of `result` with the two valuation strategies added as flat
    fields and inside the Quantitative group (existing keys untouched)."""
    merged = dict(result)
    merged["graham_valuation"] = graham
    merged["greenblatt_magic_formula"] = magic_formula
    quantitative = dict(merged.get("quantitative_strategies") or {})
    quantitative["graham_valuation"] = graham
    quantitative["greenblatt_magic_formula"] = magic_formula
    merged["quantitative_strategies"] = quantitative
    return merged


@router.get("/{symbol}/strategies", response_model=ApiResponse[StrategyAnalytics])
async def get_strategy_analytics(symbol: str, db: AsyncSession = Depends(get_db)):
    """Indicator readouts for all technical strategies:
    EMA50+RSI Pullback, EMA8/EMA21 Pullback, MACD, Bollinger Bands,
    Better Breakout, SMA Trend, Triple-MA Pullback, and EVaR Risk /
    Position Sizing.

    Also carries:
    - The VT vs VTI+VXUS Portfolio Comparison (a multi-ETF
      portfolio-analytics strategy, not a technical indicator) when the
      requested symbol is VTI — the "VTI tile" placement requested by the
      client. It is computed from VT, VTI, and VXUS price history, not
      just VTI's, and is null for every other symbol.
    - The Risk-On / Risk-Off Market Regime (a cross-ETF strategy — see
      services.regime) for every symbol, computed once across the whole
      ETF universe and cached separately (see _load_market_regime) since
      it doesn't depend on which symbol was requested.
    - The TLT Monthly Cycle (a calendar-based strategy — see
      services.strategies) for every symbol, computed once from TLT's own
      price history and cached separately (see _load_tlt_monthly_cycle)
      since it doesn't depend on which symbol was requested, same
      treatment as Risk-On/Risk-Off above.
    - The TQQQ / TMF / IEF Rebalancing strategy (a multi-ETF portfolio-
      allocation and crash-defense strategy — see services.strategies
      Strategy 11) for every symbol, computed once from TQQQ/TMF/IEF's
      own price history and cached separately (see
      _load_tqqq_tmf_ief_rebalancing), same cross-cutting treatment as
      TLT Monthly Cycle and Risk-On/Risk-Off above.

    - Mswing Momentum (a single-symbol technical strategy — see
      services.strategies Strategy 12) for every symbol, computed from
      that symbol's own price history the same way as the six core
      technical strategies above. Its relative-strength leg uses QQQ's
      own Mswing reading, computed once and cached separately (see
      _load_mswing_index_value) since it doesn't depend on which symbol
      was requested — same cross-cutting treatment as TLT Monthly Cycle
      and TQQQ/TMF/IEF Rebalancing above.
    - Momentum Reversal (a single-symbol strategy — see
      services.strategies Strategy 13) for every symbol, computed from
      that symbol's own price history the same way as the six core
      technical strategies above.
    - One-Month Factor Momentum (a cross-ETF strategy — see
      services.strategies Strategy 14) for every symbol, computed once
      across the whole ETF universe and cached separately (see
      _load_factor_momentum) since the factor ranking itself doesn't
      depend on which symbol was requested — same cross-cutting
      treatment as Risk-On/Risk-Off above.
    - HMM Regime-Switching (a single-symbol strategy — see
      services.strategies Strategy 15) for every symbol, computed from
      that symbol's own price history the same way as the six core
      technical strategies above.

    Reuses the existing price retrieval (`load_prices`) and symbol
    validation (`resolve_meta`, `require_prices`) helpers, and follows
    the same cache-then-compute pattern as the other /etfs routes.

    Also carries four new single-symbol Quantitative Strategies, each
    computed from the requested symbol's own price history the same way
    as the existing single-symbol strategies above (see
    services.strategies Strategies 16-19): Squeeze Momentum Indicator,
    Self-Organized Criticality (Avalanche Distribution), Adaptive
    Causal Wavelet Trend Filter, and First Passage Time Distribution
    Analysis.

    The response additionally groups every strategy above into
    `technicalStrategies` / `quantitativeStrategies` / `regimeStrategies`
    / `calendarStrategies` / `portfolioStrategies` (see
    schemas.strategy's "Strategy categories" note). This is purely an
    ADDITIVE, alternate view of the same data already carried by the
    existing flat fields (`data.emaRsi`, `data.evarRisk`, ...) — the
    flat fields are left in place unchanged for backward compatibility,
    since this repository ships no frontend code to migrate and an
    external frontend may depend on them (see BACKEND_TRANSFORMATION_PLAN
    and the reorg spec's frontend-compatibility requirement).

    Also carries seven additional single-symbol TECHNICAL Strategies
    (services.strategies Strategies 20-26), each computed from the
    requested symbol's own daily OHLC history like the strategies above
    and surfaced both as flat fields and inside `technicalStrategies`
    (no new category): TD Setup / TD 9 (`tdSetup`), Supertrend
    Confirmation (`supertrendConfirmation`), UFX + Vervoort Heiken Ashi
    (`ufxVervoortHeikenAshi`, public-formula stand-ins - see its
    schema), Trend Signals + Trend Tide Oscillator
    (`trendSignalsTrendTide`, proprietary parts not reproducible), Pivot
    Levels (`pivotLevels`), CPR (`cpr`), and TA Candles / Trend Health
    (`taCandlesTrendHealth`). Analytical state only - no action field.

    NOTE: bumping the cache key version (v9 -> v10) so previously cached
    responses (which don't contain the seven new technical strategies)
    are not served for the new fields; the response shape stays backward
    compatible either way since all fields are optional.

    Also carries three article-based strategies (services.strategies
    Strategies 29, 31, 32) computed with their default parameters from the
    symbol's own daily history and surfaced as flat fields and inside
    their category group: `seasonalHeatMap` (calendarStrategies),
    `emaBollingerAtr` (technicalStrategies, with a train / out-of-sample
    backtest) and `crashWarning` (quantitativeStrategies; an EXPERIMENTAL
    technical-risk monitor, not a crash forecast). Cache key bumped
    v10 -> v11 so older cached payloads are not served; all new fields are
    optional. Strategy 30 (Diversification Heat Map) is multi-asset and is
    served by POST /api/portfolio/diversification_heat_map; the parameterised
    seasonal / EMA-BB-ATR variants are GET /api/etfs/{symbol}/strategies/
    seasonal_heat_map and .../ema_bollinger_atr.

    Also carries two fundamental-valuation Quantitative strategies
    (services.strategies Strategies 27-28), added WITHOUT a new endpoint
    or cache-key change (they are attached after the cached ETF payload):
    `graham_valuation` (per-stock; "unavailable" for ETFs) and
    `greenblatt_magic_formula` (stock-universe ranking, cached separately,
    attached to every response). A symbol that is not an ETF but is a
    current member of an ingested stock universe is no longer a 404: it
    returns only these two strategies for that stock (ETF price-based
    strategies are not computed for stocks).
    """
    try:
        meta = resolve_meta(symbol)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        stock = await _find_stock_member(db, symbol)
        if stock is None:
            raise
        graham = await _compute_graham_for_stock(db, stock.symbol)
        magic = await _load_magic_formula_view(db, stock.universe, stock.symbol)
        return ApiResponse(
            data=_with_valuation_strategies({}, graham, magic),
            message=(
                f"{stock.symbol} is a stock ({stock.universe}): only the Quantitative "
                "valuation strategies (Graham Valuation, Greenblatt Magic Formula) apply; "
                "ETF price-based strategies are not computed for stocks."
            ),
        )

    cache_key = f"tw:v11:strategies:{meta.symbol}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        result = cached
    else:
        prices = await load_prices(db, meta.symbol)
        require_prices(prices, meta.symbol)

        result = {
            "ema50_rsi": strategies.compute_ema50_rsi_pullback(prices),
            "ema821": strategies.compute_ema8_21_pullback(prices),
            "macd": strategies.compute_macd(prices),
            "bollinger_bands": strategies.compute_bollinger_bands(prices),
            "better_breakout": strategies.compute_better_breakout(prices),
            "sma_trend": strategies.compute_sma_trend(prices),
            "vt_vs_vti_vxus": None,
            "evar_risk": strategies.compute_evar_risk(prices),
            "risk_on_risk_off": None,
            "triple_ma_pullback": strategies.compute_triple_ma_pullback(prices),
            "tlt_monthly_cycle": None,
            "tqqq_tmf_ief_rebalancing": None,
            "mswing": None,
            "momentum_reversal": strategies.compute_momentum_reversal(prices),
            "factor_momentum": None,
            "hmm_regime_switching": strategies.compute_hmm_regime_switching(prices),
            "squeeze_momentum": strategies.compute_squeeze_momentum(prices),
            "soc_avalanche": strategies.compute_soc_avalanche(prices),
            "wavelet_trend_filter": strategies.compute_wavelet_trend_filter(prices),
            "first_passage_time": strategies.compute_first_passage_time(prices),
            "td_setup": strategies.compute_td_setup(prices),
            "supertrend_confirmation": strategies.compute_supertrend_confirmation(prices),
            "ufx_vervoort_heiken_ashi": strategies.compute_ufx_vervoort_heiken_ashi(prices),
            "trend_signals_trend_tide": strategies.compute_trend_signals_trend_tide(prices),
            "pivot_levels": strategies.compute_pivot_levels(prices),
            "cpr": strategies.compute_cpr(prices),
            "ta_candles_trend_health": strategies.compute_ta_candles_trend_health(prices),
            # Strategies 29, 31, 32 (default parameters; the parameterised
            # variants are the dedicated routes at the bottom of this module).
            "seasonal_heat_map": strategies.compute_seasonal_heat_map(prices, symbol=meta.symbol),
            "ema_bollinger_atr": strategies.compute_ema_bollinger_atr(prices),
            "crash_warning": strategies.compute_crash_warning(prices),
        }

        if meta.symbol == PORTFOLIO_COMPARISON_SYMBOL:
            vt_prices = await load_prices(db, "VT")
            vxus_prices = await load_prices(db, "VXUS")
            result["vt_vs_vti_vxus"] = portfolio_comparison.compute_vt_vs_vti_vxus(
                vt_prices=vt_prices,
                vti_prices=prices,
                vxus_prices=vxus_prices,
            )

        market_regime = await cache.get_or_compute(
            MARKET_REGIME_CACHE_KEY, lambda: _load_market_regime(db)
        )
        result["risk_on_risk_off"] = regime.extract_symbol_view(
            market_regime, meta.symbol
        )

        result["tlt_monthly_cycle"] = await cache.get_or_compute(
            TLT_MONTHLY_CYCLE_CACHE_KEY, lambda: _load_tlt_monthly_cycle(db)
        )

        result["tqqq_tmf_ief_rebalancing"] = await cache.get_or_compute(
            TQQQ_TMF_IEF_CACHE_KEY, lambda: _load_tqqq_tmf_ief_rebalancing(db)
        )

        mswing_index_value = await cache.get_or_compute(
            MSWING_INDEX_CACHE_KEY, lambda: _load_mswing_index_value(db)
        )
        result["mswing"] = strategies.compute_mswing(
            prices, index_mswing=mswing_index_value
        )

        factor_momentum = await cache.get_or_compute(
            FACTOR_MOMENTUM_CACHE_KEY, lambda: _load_factor_momentum(db)
        )
        result["factor_momentum"] = strategies.extract_factor_momentum_view(
            factor_momentum, meta.symbol, prices
        )

        # Additive category groupings (see docstring above) — built from
        # the exact same values already assembled into `result`, so
        # there is nothing new to compute here.
        result["technical_strategies"] = {
            "ema50_rsi": result["ema50_rsi"],
            "ema821": result["ema821"],
            "macd": result["macd"],
            "bollinger_bands": result["bollinger_bands"],
            "better_breakout": result["better_breakout"],
            "sma_trend": result["sma_trend"],
            "triple_ma_pullback": result["triple_ma_pullback"],
            "mswing": result["mswing"],
            "td_setup": result["td_setup"],
            "supertrend_confirmation": result["supertrend_confirmation"],
            "ufx_vervoort_heiken_ashi": result["ufx_vervoort_heiken_ashi"],
            "trend_signals_trend_tide": result["trend_signals_trend_tide"],
            "pivot_levels": result["pivot_levels"],
            "cpr": result["cpr"],
            "ta_candles_trend_health": result["ta_candles_trend_health"],
            "ema_bollinger_atr": result["ema_bollinger_atr"],
        }
        result["quantitative_strategies"] = {
            "evar_risk": result["evar_risk"],
            "momentum_reversal": result["momentum_reversal"],
            "squeeze_momentum": result["squeeze_momentum"],
            "soc_avalanche": result["soc_avalanche"],
            "wavelet_trend_filter": result["wavelet_trend_filter"],
            "first_passage_time": result["first_passage_time"],
            "crash_warning": result["crash_warning"],
        }
        result["regime_strategies"] = {
            "risk_on_risk_off": result["risk_on_risk_off"],
            "hmm_regime_switching": result["hmm_regime_switching"],
            "factor_momentum": result["factor_momentum"],
        }
        result["calendar_strategies"] = {
            "tlt_monthly_cycle": result["tlt_monthly_cycle"],
            "seasonal_heat_map": result["seasonal_heat_map"],
        }
        result["portfolio_strategies"] = {
            "tqqq_tmf_ief_rebalancing": result["tqqq_tmf_ief_rebalancing"],
            "vt_vs_vti_vxus": result["vt_vs_vti_vxus"],
        }

        await cache.set_json(cache_key, result)

    # Strategies 27-28 (attached after the cached ETF payload; see docstring).
    result = _with_valuation_strategies(
        result,
        _graham_unavailable(meta.symbol, [GRAHAM_NOT_APPLICABLE_ETF]),
        await _load_magic_formula_view(db, VALUATION_DEFAULT_UNIVERSE, None),
    )

    # Insufficient history is not an error — surface it as an informational
    # message (same pattern as the "skipped" symbols message in etfs.py),
    # with the affected strategy's field left null in `data`.
    strategy_labels = {
        "ema50_rsi": "EMA50+RSI Pullback",
        "ema821": "EMA8/EMA21 Pullback",
        "macd": "MACD",
        "bollinger_bands": "Bollinger Bands",
        "better_breakout": "Better Breakout",
        "sma_trend": "SMA Trend",
        "evar_risk": "EVaR Risk",
        "risk_on_risk_off": "Risk-On / Risk-Off Market Regime",
        "triple_ma_pullback": "Triple-MA Pullback",
        "tlt_monthly_cycle": "TLT Monthly Cycle",
        "tqqq_tmf_ief_rebalancing": "TQQQ / TMF / IEF Rebalancing",
        "mswing": "Mswing Momentum",
        "momentum_reversal": "Momentum Reversal",
        "factor_momentum": "One-Month Factor Momentum",
        "hmm_regime_switching": "HMM Regime Switching",
        "squeeze_momentum": "Squeeze Momentum Indicator",
        "soc_avalanche": "Self-Organized Criticality (Avalanche Distribution)",
        "wavelet_trend_filter": "Adaptive Causal Wavelet Trend Filter",
        "first_passage_time": "First Passage Time Distribution Analysis",
        "td_setup": "TD Setup / TD 9",
        "supertrend_confirmation": "Supertrend Confirmation",
        "ufx_vervoort_heiken_ashi": "UFX + Vervoort Heiken Ashi",
        "trend_signals_trend_tide": "Trend Signals + Trend Tide Oscillator",
        "pivot_levels": "Pivot Levels",
        "cpr": "CPR (Central Pivot Range)",
        "ta_candles_trend_health": "TA Candles / Trend Health",
        "seasonal_heat_map": "Seasonal Performance Heat Map",
        "ema_bollinger_atr": "EMA + Bollinger Bands + ATR",
        "crash_warning": "Technical Crash-Warning Risk Indicator",
    }
    if meta.symbol == PORTFOLIO_COMPARISON_SYMBOL:
        strategy_labels["vt_vs_vti_vxus"] = "VT vs VTI+VXUS Portfolio Comparison"

    missing = [
        label for key, label in strategy_labels.items() if result.get(key) is None
    ]

    message = None
    if missing:
        message = (
            f"Insufficient price history for {', '.join(missing)} — "
            f"need more trading days of data for {meta.symbol}."
        )

    return ApiResponse(data=result, message=message)


# ---------------------------------------------------------------------------
# Portfolio Optimization: Kelly Criterion + Mean-Variance Optimization
#
# Genuinely multi-asset — the request specifies an explicit basket of
# symbols, unlike every route above (always one symbol's own history).
# See services.strategies.compute_portfolio_optimization for the full
# methodology and documented assumptions.
# ---------------------------------------------------------------------------

PORTFOLIO_OPT_MIN_SYMBOLS = 2
PORTFOLIO_OPT_MAX_SYMBOLS = 8


def _resolve_portfolio_symbol(symbol: str) -> str:
    """Validates a symbol against every known price-source registry
    (Explorer ETFs, portfolio-comparison-only symbols, and
    strategy-only symbols) — Portfolio Optimization is a genuinely
    multi-asset endpoint that may reasonably span all of them, not just
    the six Explorer products resolve_meta() checks.
    """
    upper = symbol.strip().upper()
    if (
        upper in ETF_REGISTRY
        or upper in PORTFOLIO_COMPARISON_REGISTRY
        or upper in STRATEGY_ONLY_REGISTRY
    ):
        return upper
    raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


@portfolio_router.post(
    "/optimization", response_model=ApiResponse[PortfolioOptimizationResult]
)
async def post_portfolio_optimization(
    request: PortfolioOptimizationRequest, db: AsyncSession = Depends(get_db)
):
    """Kelly Criterion + Mean-Variance Optimization across a basket of
    symbols supplied in the request body (see
    services.strategies.compute_portfolio_optimization).

    Deliberately NOT folded into /etfs/{symbol}/strategies: that route
    always analyzes one symbol's own price history, while this
    genuinely combines multiple symbols' return/covariance structure —
    the one case in this reorg where a dedicated endpoint was called
    for instead of stretching the existing per-symbol shape.
    """
    requested = [s.strip().upper() for s in request.symbols if s and s.strip()]
    seen: set[str] = set()
    unique_symbols: list[str] = []
    for s in requested:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    if not PORTFOLIO_OPT_MIN_SYMBOLS <= len(unique_symbols) <= PORTFOLIO_OPT_MAX_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Provide between {PORTFOLIO_OPT_MIN_SYMBOLS} and "
                f"{PORTFOLIO_OPT_MAX_SYMBOLS} distinct symbols."
            ),
        )
    for s in unique_symbols:
        _resolve_portfolio_symbol(s)

    normalized_key = ",".join(sorted(unique_symbols))
    cache_key = (
        f"tw:v1:portfolio:optimization:{normalized_key}:"
        f"{request.risk_free_rate_percent:g}"
    )
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices_by_symbol = {}
    for s in unique_symbols:
        prices = await load_prices(db, s)
        require_prices(prices, s)
        prices_by_symbol[s] = prices

    result = strategies.compute_portfolio_optimization(
        prices_by_symbol, risk_free_rate_percent=request.risk_free_rate_percent
    )
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Insufficient overlapping price history across the "
                "requested symbols for a portfolio optimization."
            ),
        )

    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)


# ---------------------------------------------------------------------------
# Strategies 29-32: parameterised variants of the article-based strategies.
# The default-parameter results of 29, 31 and 32 are already part of
# GET /etfs/{symbol}/strategies above; these routes exist only because the
# date range (29), the tickers/weights (30) and the backtest configuration /
# hourly timeframe (31) cannot be expressed through that route.
# Results are cached under "tw:v1:article:*", which nightly ingestion busts.
# ---------------------------------------------------------------------------

ARTICLE_CACHE_PREFIX = "tw:v1:article:"
DIVERSIFICATION_MIN_SYMBOLS = 2
DIVERSIFICATION_MAX_SYMBOLS = 12
# Hourly bars are fetched on demand from EODHD (5 API calls per request) and
# NOT stored in PostgreSQL, so they are memoised to protect the API quota even
# when Redis is down.
HOURLY_BARS_TTL_SECONDS = 6 * 3600
HOURLY_MEMO_SECONDS = 900
HOURLY_MAX_LOOKBACK_DAYS = 1825
_hourly_memo: dict[str, tuple[float, list[dict]]] = {}


def _digest(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _price_source_meta(symbol: str):
    return (
        ETF_REGISTRY.get(symbol)
        or PORTFOLIO_COMPARISON_REGISTRY.get(symbol)
        or STRATEGY_ONLY_REGISTRY[symbol]
    )


async def _load_hourly_bars(symbol: str, lookback_days: int) -> list[strategies.PriceBar]:
    """Hourly OHLCV for `symbol` straight from EODHD's intraday endpoint.

    Never substitutes daily data: an entitlement problem or missing coverage is
    reported as an explicit, actionable error."""
    memo_key = f"{symbol}:{lookback_days}"
    cache_key = f"{ARTICLE_CACHE_PREFIX}bars1h:{memo_key}"
    now = time.monotonic()
    memo = _hourly_memo.get(memo_key)
    rows = memo[1] if memo and now - memo[0] < HOURLY_MEMO_SECONDS else None

    if rows is None:
        async def _fetch() -> list[dict]:
            to_dt = datetime.datetime.now(datetime.timezone.utc)
            from_dt = to_dt - datetime.timedelta(days=lookback_days)
            try:
                return await eodhd_client.fetch_intraday_history(
                    _price_source_meta(symbol).eodhd_symbol(), "1h", from_dt=from_dt, to_dt=to_dt
                )
            except eodhd_client.EodhdEntitlementError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                logger.warning("EODHD intraday request for %s failed with HTTP %s", symbol, status)
                raise HTTPException(
                    status_code=404 if status == 404 else 502,
                    detail=(
                        f"EODHD has no hourly data for {symbol}." if status == 404
                        else "EODHD intraday request failed; try again later."
                    ),
                ) from exc
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                logger.warning("EODHD intraday request for %s failed: %s", symbol, type(exc).__name__)
                raise HTTPException(status_code=502, detail="EODHD intraday request timed out; try again later.") from exc

        rows = await cache.get_or_compute(cache_key, _fetch, ttl=HOURLY_BARS_TTL_SECONDS)
        _hourly_memo[memo_key] = (now, rows)

    bars: list[strategies.PriceBar] = []
    for row in rows or []:
        try:
            bars.append(
                strategies.PriceBar(
                    date=datetime.datetime.strptime(str(row["datetime"])[:19], "%Y-%m-%d %H:%M:%S"),
                    open=float(row["open"]), high=float(row["high"]), low=float(row["low"]),
                    close=float(row["close"]), volume=float(row.get("volume") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue  # malformed bar: dropped (counted below), never guessed
    if not bars:
        raise HTTPException(
            status_code=404,
            detail=(
                f"EODHD returned no usable hourly bars for {symbol} in the last {lookback_days} days "
                "(symbol or period not covered). Daily data was NOT substituted."
            ),
        )
    return bars


@router.get(
    "/{symbol}/strategies/seasonal_heat_map",
    response_model=ApiResponse[SeasonalHeatMapStrategy],
)
async def get_seasonal_heat_map(
    symbol: str,
    start: datetime.date | None = Query(default=None, description="First month shown (month granularity)."),
    end: datetime.date | None = Query(default=None, description="Last month shown (month granularity)."),
    price_basis: Literal["adjusted", "close"] = Query(
        default="adjusted",
        description="'adjusted' = EODHD adjusted_close (splits+dividends); 'close' = raw close.",
    ),
    db: AsyncSession = Depends(get_db),
):
    """Strategy 29 - monthly + geometrically-linked quarterly returns by
    calendar year for a supported symbol, over an optional date range."""
    sym = _resolve_portfolio_symbol(symbol)
    if start and end and start > end:
        raise HTTPException(status_code=422, detail="start must not be after end.")

    cache_key = f"{ARTICLE_CACHE_PREFIX}seasonal:{sym}:{price_basis}:{start}:{end}"
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices = await load_prices(db, sym)
    require_prices(prices, sym)
    result = strategies.compute_seasonal_heat_map(
        prices, symbol=sym, start=start, end=end, price_basis=price_basis
    )
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient price history for {sym} in the requested range: at least two calendar "
                "months of data are needed. Extend history with POST /api/admin/backfill if needed."
            ),
        )
    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)


@router.get(
    "/{symbol}/strategies/ema_bollinger_atr",
    response_model=ApiResponse[EmaBollingerAtrStrategy],
)
async def get_ema_bollinger_atr(
    symbol: str,
    timeframe: Literal["1d", "1h"] = Query(
        default="1d",
        description="'1d' uses stored daily bars. '1h' fetches hourly bars from EODHD (needs an intraday plan).",
    ),
    lookback_days: int = Query(
        default=730, ge=30, le=HOURLY_MAX_LOOKBACK_DAYS, description="Hourly timeframe only: history window."
    ),
    optimize: bool = Query(
        default=False, description="Pick stop multiple / take-profit ratio on the TRAINING segment only."
    ),
    ema_period: int | None = Query(default=None),
    bb_period: int | None = Query(default=None),
    bb_std: float | None = Query(default=None),
    rsi_period: int | None = Query(default=None),
    atr_period: int | None = Query(default=None),
    trend_lookback: int | None = Query(default=None),
    stop_loss_atr_mult: float | None = Query(default=None),
    take_profit_ratio: float | None = Query(default=None),
    rsi_long_exit: float | None = Query(default=None),
    rsi_short_exit: float | None = Query(default=None),
    commission_percent: float | None = Query(default=None),
    slippage_percent: float | None = Query(default=None),
    position_fraction: float | None = Query(default=None),
    train_fraction: float | None = Query(default=None),
    allow_short: bool | None = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Strategy 31 - EMA trend + Bollinger entry conditions + ATR risk levels
    with a train / out-of-sample historical backtest. Analytics only: no
    orders are placed. Omitted parameters keep the article defaults."""
    sym = _resolve_portfolio_symbol(symbol)
    overrides = {
        "ema_period": ema_period, "bb_period": bb_period, "bb_std": bb_std, "rsi_period": rsi_period,
        "atr_period": atr_period, "trend_lookback": trend_lookback, "stop_loss_atr_mult": stop_loss_atr_mult,
        "take_profit_ratio": take_profit_ratio, "rsi_long_exit": rsi_long_exit,
        "rsi_short_exit": rsi_short_exit, "commission_percent": commission_percent,
        "slippage_percent": slippage_percent, "position_fraction": position_fraction,
        "train_fraction": train_fraction, "allow_short": allow_short,
    }
    try:
        params = strategies.resolve_ema_bollinger_atr_params(overrides)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cache_key = (
        f"{ARTICLE_CACHE_PREFIX}eba:{sym}:{timeframe}:{lookback_days if timeframe == '1h' else 0}:"
        f"{int(optimize)}:{_digest(params)}"
    )
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    if timeframe == "1h":
        bars = await _load_hourly_bars(sym, lookback_days)
        source = "EODHD intraday API (interval 1h, raw/unadjusted, UTC); not stored in the database"
    else:
        bars = await load_prices(db, sym)
        require_prices(bars, sym)
        source = None
    result = strategies.compute_ema_bollinger_atr(
        bars, timeframe=timeframe, params=overrides, optimize=optimize, data_source=source
    )
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Insufficient {timeframe} history for {sym}: the indicators need a warm-up period plus at "
                f"least {strategies.EBA_MIN_SEGMENT_BARS} bars in each of the training and out-of-sample segments."
            ),
        )
    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)


@portfolio_router.post(
    "/diversification_heat_map",
    response_model=ApiResponse[DiversificationHeatMapStrategy],
)
async def post_diversification_heat_map(
    request: DiversificationHeatMapRequest, db: AsyncSession = Depends(get_db)
):
    """Strategy 30 - annual returns of several supported instruments ranked
    within each calendar year, plus an optional user-weighted portfolio
    (annual rebalancing - see services.strategies Strategy 30). Descriptive
    only; it does not establish diversification benefits."""
    unique_symbols: list[str] = []
    for raw in request.symbols:
        sym = (raw or "").strip().upper()
        if sym and sym not in unique_symbols:
            unique_symbols.append(sym)
    if not DIVERSIFICATION_MIN_SYMBOLS <= len(unique_symbols) <= DIVERSIFICATION_MAX_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Provide between {DIVERSIFICATION_MIN_SYMBOLS} and "
                f"{DIVERSIFICATION_MAX_SYMBOLS} distinct symbols."
            ),
        )
    for sym in unique_symbols:
        _resolve_portfolio_symbol(sym)
    if request.start and request.end and request.start > request.end:
        raise HTTPException(status_code=422, detail="start must not be after end.")

    weights = None
    if request.weights_percent is not None:
        try:
            weights = strategies.validate_portfolio_weights(unique_symbols, request.weights_percent)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    cache_key = ARTICLE_CACHE_PREFIX + "diversification:" + _digest(
        {
            "s": sorted(unique_symbols), "w": weights, "start": request.start, "end": request.end,
            "basis": request.price_basis,
        }
    )
    cached = await cache.get_json(cache_key)
    if cached is not None:
        return ApiResponse(data=cached)

    prices_by_symbol = {}
    for sym in unique_symbols:
        prices = await load_prices(db, sym)
        require_prices(prices, sym)
        prices_by_symbol[sym] = prices

    result = strategies.compute_diversification_heat_map(
        prices_by_symbol, weights_percent=weights, start=request.start, end=request.end,
        price_basis=request.price_basis,
    )
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "No complete calendar-year returns could be formed for the requested symbols and range "
                "(at least two calendar years of history are needed). Extend history with "
                "POST /api/admin/backfill if needed."
            ),
        )
    await cache.set_json(cache_key, result)
    return ApiResponse(data=result)
