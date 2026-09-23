"""
Strategy Analytics endpoint.

Analytical / visualization only: returns computed indicator values and
descriptive flags for pre-defined technical strategies. This is NOT a
trading bot and does NOT return buy/sell/hold signals, recommendations,
or anything ML-derived.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ETF_REGISTRY, PORTFOLIO_COMPARISON_REGISTRY, STRATEGY_ONLY_REGISTRY
from app.database import get_db
from app.routers.deps import load_prices, require_prices, resolve_meta
from app.schemas import ApiResponse, PortfolioOptimizationRequest, PortfolioOptimizationResult, StrategyAnalytics
from app.services import cache, portfolio_comparison, regime, strategies

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

    NOTE: bumping the cache key version (v8 -> v9) so previously cached
    responses (which don't contain the new quantitative strategies or
    the category groupings) are not served for the new fields; the
    response shape stays backward compatible either way since all
    fields are optional.
    """
    meta = resolve_meta(symbol)

    cache_key = f"tw:v9:strategies:{meta.symbol}"
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
        }
        result["quantitative_strategies"] = {
            "evar_risk": result["evar_risk"],
            "momentum_reversal": result["momentum_reversal"],
            "squeeze_momentum": result["squeeze_momentum"],
            "soc_avalanche": result["soc_avalanche"],
            "wavelet_trend_filter": result["wavelet_trend_filter"],
            "first_passage_time": result["first_passage_time"],
        }
        result["regime_strategies"] = {
            "risk_on_risk_off": result["risk_on_risk_off"],
            "hmm_regime_switching": result["hmm_regime_switching"],
            "factor_momentum": result["factor_momentum"],
        }
        result["calendar_strategies"] = {
            "tlt_monthly_cycle": result["tlt_monthly_cycle"],
        }
        result["portfolio_strategies"] = {
            "tqqq_tmf_ief_rebalancing": result["tqqq_tmf_ief_rebalancing"],
            "vt_vs_vti_vxus": result["vt_vs_vti_vxus"],
        }

        await cache.set_json(cache_key, result)

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