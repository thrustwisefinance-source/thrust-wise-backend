"""
Strategy Analytics endpoint.

Analytical / visualization only: returns computed indicator values and
descriptive flags for pre-defined technical strategies. This is NOT a
trading bot and does NOT return buy/sell/hold signals, recommendations,
or anything ML-derived.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.routers.deps import load_prices, require_prices, resolve_meta
from app.schemas import ApiResponse, StrategyAnalytics
from app.services import cache, portfolio_comparison, strategies

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/etfs", tags=["strategy-analytics"])

# Symbol under which the VT vs VTI+VXUS portfolio comparison is surfaced
# ("VTI tile" placement per product request). VTI's own price list (already
# loaded below for the six technical strategies) is reused; VT and VXUS are
# portfolio-comparison-only symbols (see
# app.constants.PORTFOLIO_COMPARISON_REGISTRY) fetched directly by plain
# symbol — they are not Explorer products, so they are intentionally not
# passed through resolve_meta.
PORTFOLIO_COMPARISON_SYMBOL = "VTI"


@router.get("/{symbol}/strategies", response_model=ApiResponse[StrategyAnalytics])
async def get_strategy_analytics(symbol: str, db: AsyncSession = Depends(get_db)):
    """Indicator readouts for all technical strategies:
    EMA50+RSI Pullback, EMA8/EMA21 Pullback, MACD, Bollinger Bands,
    Better Breakout, and SMA Trend.

    Also carries the VT vs VTI+VXUS Portfolio Comparison (a multi-ETF
    portfolio-analytics strategy, not a technical indicator) when the
    requested symbol is VTI — the "VTI tile" placement requested by the
    client. It is computed from VT, VTI, and VXUS price history, not
    just VTI's, and is null for every other symbol.

    Reuses the existing price retrieval (`load_prices`) and symbol
    validation (`resolve_meta`, `require_prices`) helpers, and follows
    the same cache-then-compute pattern as the other /etfs routes.

    NOTE: bumping the cache key version (v2 -> v3) so previously cached
    responses (which don't contain vt_vs_vti_vxus) are not served for
    the new field; the response shape stays backward compatible either
    way since all fields are optional.
    """
    meta = resolve_meta(symbol)

    cache_key = f"tw:v3:strategies:{meta.symbol}"
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
        }

        if meta.symbol == PORTFOLIO_COMPARISON_SYMBOL:
            vt_prices = await load_prices(db, "VT")
            vxus_prices = await load_prices(db, "VXUS")
            result["vt_vs_vti_vxus"] = portfolio_comparison.compute_vt_vs_vti_vxus(
                vt_prices=vt_prices,
                vti_prices=prices,
                vxus_prices=vxus_prices,
            )

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