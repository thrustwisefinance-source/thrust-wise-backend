"""
CANSLIM Screener endpoint.

Analytical / screening only — returns a computed CANSLIM score and its
underlying criteria per symbol. This is NOT a trading bot and does NOT
return buy/sell/hold signals or recommendations, matching the convention
in routers/strategy.py.

CANSLIM is a cross-sectional stock screener, not a single-symbol
technical indicator (see services.canslim module docstring), so it is
intentionally NOT nested under GET /etfs/{symbol}/strategies — it gets
its own top-level resource, following the same "the URL names the
resource being returned" convention as /etfs/compare (a separate router,
see routers/compare.py) rather than being force-fit into the per-symbol
strategies endpoint.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ETF_REGISTRY
from app.database import get_db
from app.routers.deps import load_prices
from app.schemas import ApiResponse, CanslimScreenerResponse
from app.services import cache, canslim

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategy-analytics"])

# Cross-sectional like Risk-On/Risk-Off and the CANSLIM universe doesn't
# depend on any single symbol's page — computed once across the whole
# ETF universe and cached separately, same treatment as
# routers.strategy._load_market_regime. Reuses the existing Redis JSON
# cache (app.services.cache) rather than a second cache system.
CANSLIM_SCREENER_CACHE_KEY = "tw:v1:screener:canslim"

DATA_SOURCE_LIMITATIONS = [
    (
        "C (current quarterly earnings), A (annual growth proxy), S "
        "(supply/float shares), and I (institutional sponsorship) are "
        "unavailable: ThrustWise's data source is EODHD daily OHLCV "
        "price bars only — no fundamentals or ownership data is "
        "ingested. These criteria are not counted in `score` and are "
        "listed per-result in `unavailableCriteria`, never fabricated."
    ),
    (
        "N (52-week-high proximity), L (relative-strength-from-52-week-"
        "low proxy), and M (price vs. SMA200) are computed from price "
        "history and are available for every screened symbol with "
        "enough history."
    ),
    (
        "CANSLIM is designed for individual equities. ThrustWise's "
        "existing universe (app.constants.ETF_REGISTRY) is six "
        "diversified ETFs, not individual stocks — there is no stock "
        "universe in this codebase, so the ETF universe is screened "
        "instead. See services.canslim module docstring."
    ),
    (
        "A and L use the source article's documented simplified proxies "
        "(revenue growth + ROE; percent above the 52-week low), not the "
        "official CANSLIM methodology's annual EPS growth or RS Rating."
    ),
]


async def _load_canslim_screen(db: AsyncSession) -> dict:
    prices_by_symbol = {
        registry_symbol: await load_prices(db, registry_symbol)
        for registry_symbol in ETF_REGISTRY
    }
    return canslim.screen_universe(prices_by_symbol)


@router.get("/canslim", response_model=ApiResponse[CanslimScreenerResponse])
async def get_canslim_screener(
    min_score: int
    | None = Query(
        default=None,
        ge=0,
        le=canslim.CANSLIM_MAX_SCORE,
        description=(
            "Only return symbols scoring at least this many of the 7 "
            "CANSLIM criteria. Omit to return the full ranked universe "
            "(default minimum-passing threshold is still reported in "
            "the response as `minimumScore`)."
        ),
    ),
    db: AsyncSession = Depends(get_db),
):
    """CANSLIM score (0-7) and per-criterion pass/fail for every symbol
    in ThrustWise's ETF universe, ranked highest-score first.

    See services.canslim module docstring for the full seven-criteria
    definitions, the source article's simplified proxies used for A and
    L, and why C/A/S/I are currently unavailable (no fundamentals data
    source). `dataSourceLimitations` in the response documents this
    explicitly rather than silently returning partial results.

    Reuses the existing price retrieval (`load_prices`) and Redis JSON
    cache (`cache.get_or_compute`), same cache-then-compute pattern as
    the other strategy endpoints — cached cross-sectionally (once for
    the whole universe, not per-request), same treatment as Risk-On/
    Risk-Off in routers/strategy.py.
    """
    screen = await cache.get_or_compute(
        CANSLIM_SCREENER_CACHE_KEY, lambda: _load_canslim_screen(db)
    )

    results = screen["results"]
    if min_score is not None:
        results = [r for r in results if r["score"] >= min_score]

    data = {
        "universe": list(ETF_REGISTRY.keys()),
        "results": results,
        "unavailable_symbols": screen["unavailable_symbols"],
        "max_score": screen["max_score"],
        "minimum_score": screen["minimum_score"],
        "data_source_limitations": DATA_SOURCE_LIMITATIONS,
    }

    message = None
    if screen["unavailable_symbols"]:
        message = (
            "Insufficient price history to score "
            f"{', '.join(screen['unavailable_symbols'])}."
        )

    return ApiResponse(data=data, message=message)