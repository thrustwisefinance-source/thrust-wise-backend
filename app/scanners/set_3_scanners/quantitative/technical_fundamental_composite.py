"""
Stock Scanner -> Quantitative Screening: Technical-Fundamental Composite Scoring
(Stock Grade System).

Scores each stock 0-4 with four independent conditions (thresholds configurable;
article defaults in parentheses):

    price above the MA (50-day) | RSI below (70) | forward P/E below (20)
    | debt-to-equity below (1)

and exposes "RSI below 30" as a SEPARATE analytical flag (it never adds a
point). This is a screening analytic, not a trading system.

Multi-symbol scans process every symbol independently: an exception for one
symbol becomes an ``error`` result for that symbol only. Results are cached per
symbol under ``tw:set3:scanner:*``.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections import Counter
from typing import Awaitable, Callable

from app.scanners.set_3_scanners.base import CATEGORY_QUANTITATIVE, ScannerDescriptor
from app.scanners.set_3_scanners.schemas.quantitative import (
    SCANNER_ID,
    EffectiveParameters,
    ScannerRequest,
    ScannerResponse,
    ScanSummary,
    StockScoreResult,
)
from app.scanners.set_3_scanners.services import (
    cache_service,
    scoring_service,
    stock_data_service,
)

logger = logging.getLogger(__name__)

CALCULATION_VERSION = "tfcs-1.0.0"
DESCRIPTOR = ScannerDescriptor(
    scanner_id=SCANNER_ID,
    name="Technical-Fundamental Composite Scoring",
    category=CATEGORY_QUANTITATIVE,
    description=(
        "Scores a stock 0-4: price above its moving average, RSI below a threshold, "
        "forward P/E below a threshold, debt-to-equity below a threshold."
    ),
    calculation_version=CALCULATION_VERSION,
)

NOTES = [
    "Screening analytics only - not a trading system and not investment advice.",
    "A condition whose input is unavailable gets NO point and a null score (it is never "
    "treated as passed or as a zero-valued metric); `scoreComplete` is false for such stocks.",
    "Forward P/E <= 0 and negative debt-to-equity are 'not_meaningful': they cannot satisfy "
    "a 'below threshold' rule and earn no point.",
    "Yahoo's debtToEquity is a percentage; it is divided by 100 to give a ratio "
    "(the raw provider value is returned in debtToEquityProviderValue).",
    "RSI uses Wilder smoothing on daily closes. 'RSI below 30' is a separate opportunity "
    "flag and does not change the 0-4 score.",
    "Prices and fundamentals come from Yahoo Finance via yfinance (free, unofficial source).",
]

Fetcher = Callable[[str, int, int], Awaitable[stock_data_service.StockInputs]]


def effective_parameters(req: ScannerRequest) -> EffectiveParameters:
    return EffectiveParameters(
        ma_period=req.ma_period,
        rsi_period=req.rsi_period,
        rsi_upper_threshold=req.rsi_upper_threshold,
        rsi_opportunity_threshold=req.rsi_opportunity_threshold,
        forward_pe_threshold=req.forward_pe_threshold,
        debt_to_equity_threshold=req.debt_to_equity_threshold,
    )


def error_result(symbol: str, message: str) -> StockScoreResult:
    return StockScoreResult(symbol=symbol, calculation_status="error", error=message)


async def analyze_symbol(
    symbol: str,
    params: EffectiveParameters,
    *,
    use_cache: bool = True,
    fetcher: Fetcher | None = None,
    as_of: str | None = None,
) -> StockScoreResult:
    """Score ONE symbol. Never raises: failures become an ``error`` result."""
    as_of = as_of or datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    fp = cache_service.scanner_fingerprint(SCANNER_ID, params, CALCULATION_VERSION)
    key = cache_service.scanner_cache_key(CATEGORY_QUANTITATIVE, symbol, as_of, fp)
    try:
        if use_cache:
            cached = await cache_service.get_cached(key)
            if cached is not None:
                try:
                    return StockScoreResult.model_validate(cached)
                except Exception:  # noqa: BLE001 - stale / incompatible entry
                    logger.warning("Ignoring unreadable Set 3 scanner cache entry %s", key)

        fetch = fetcher or stock_data_service.fetch_stock_inputs
        inputs = await fetch(symbol, params.ma_period, params.rsi_period)
        result = await asyncio.to_thread(
            scoring_service.score_stock,
            symbol,
            inputs.close,
            inputs.forward_pe,
            inputs.debt_to_equity,
            params,
            debt_to_equity_provider_value=inputs.debt_to_equity_provider_value,
            extra_messages=inputs.messages,
        )
        if use_cache and result.calculation_status in ("complete", "partial"):
            ttl = (
                cache_service.TTL_COMPLETE_SECONDS
                if result.calculation_status == "complete"
                else cache_service.TTL_PARTIAL_SECONDS
            )
            await cache_service.set_cached(key, result.model_dump(mode="json", by_alias=False), ttl)
        return result
    except Exception:  # noqa: BLE001 - per-symbol isolation boundary
        logger.exception("Set 3 composite scan failed for %s", symbol)
        return error_result(symbol, "Calculation failed unexpectedly for this symbol.")


async def scan_symbols(
    symbols: list[str],
    req: ScannerRequest,
    *,
    fetcher: Fetcher | None = None,
    universe: str | None = None,
) -> ScannerResponse:
    """Scan every symbol independently (one failure never blocks the others)."""
    params = effective_parameters(req)
    as_of = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    results = await asyncio.gather(
        *(
            analyze_symbol(s, params, use_cache=req.use_cache, fetcher=fetcher, as_of=as_of)
            for s in symbols
        )
    )
    results = list(results)
    status = Counter(r.calculation_status for r in results)
    dist = Counter(str(r.total_score) for r in results if r.total_score is not None)
    summary = ScanSummary(
        requested=len(symbols),
        complete=status.get("complete", 0),
        partial=status.get("partial", 0),
        unavailable=status.get("unavailable", 0),
        error=status.get("error", 0),
        score_distribution={str(k): dist.get(str(k), 0) for k in range(5)},
        rsi_opportunity_count=sum(
            1 for r in results if r.rsi_opportunity and r.rsi_opportunity.flagged
        ),
    )
    return ScannerResponse(
        calculation_version=CALCULATION_VERSION,
        as_of=as_of,
        parameters=params,
        universe=universe,
        results=results,
        summary=summary,
        notes=NOTES,
    )
