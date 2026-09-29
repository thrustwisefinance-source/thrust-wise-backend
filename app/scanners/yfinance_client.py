"""
Low-level, resilient wrapper around `yfinance` for the CANSLIM STOCK
SCANNER (app/scanners/) ONLY.

THIS MODULE IS NOT USED BY THE OLD ETF/INDEX PIPELINE
(app.services.eodhd_client, app.services.ingestion) — that pipeline
keeps using EODHD, completely untouched (see GET /api/strategies/canslim
and the ETF Explorer, which are unaffected by anything in app/scanners/).
This module is the NEW scanner's sole third-party stock-data dependency,
replacing EODHD entirely for S&P 500 constituent/fundamental/price data
— see app.scanners.universe, app.scanners.data, and
app.scanners.ingestion module docstrings for how the pieces fit
together.

yfinance is a synchronous, unofficial wrapper around Yahoo Finance's
undocumented endpoints — not an enterprise SLA-backed API like EODHD.
Every call here is:
  - wrapped in `asyncio.to_thread` (yfinance is fully synchronous /
    blocking under the hood, so it must never run directly on the event
    loop),
  - bounded by a shared semaphore (YFINANCE_MAX_CONCURRENCY) so a
    ~500-stock refresh never opens hundreds of simultaneous connections
    to Yahoo,
  - retried a small, fixed number of times with backoff on transient
    failures (timeouts, empty/malformed responses), and
  - never allowed to raise past its caller for a single bad symbol —
    every public function here returns None on failure and logs a
    WARNING, so one bad ticker never aborts a whole-universe refresh
    (see app.scanners.ingestion / app.scanners.data).

Known yfinance limitations (documented here once so callers — and the
API's `dataSourceLimitations` field, see app.routers.scanners — don't
have to re-derive them):
  - `Ticker.institutional_holders` returns Yahoo's own "top ~10
    institutional holders" table, not a complete count of every
    institutional owner — the ingested `institutional_holders_count` is
    a lower-bound proxy, never an authoritative total.
  - `Ticker.income_stmt` (annual) typically exposes ~4 fiscal years for
    a US large-cap, which satisfies CANSLIM_A_MIN_YEARS (3) but not
    always CANSLIM_A_MAX_YEARS (5).
  - `Ticker.get_earnings_dates()` returns a rolling window of recent
    (+ upcoming) quarters, not a deep multi-year quarterly archive —
    enough for the C criterion's "vs. year-ago quarter" comparison in
    the normal case, but can be UNAVAILABLE for thinly-covered symbols.
  - Yahoo has no documented rate limit; in practice, sustained
    high-concurrency scraping gets throttled or temporarily blocked.
    Concurrency here is deliberately conservative — see
    YFINANCE_MAX_CONCURRENCY — and should be lowered further (not
    raised) if a refresh starts seeing a lot of failures.
  - `Ticker.info` occasionally returns a near-empty dict for a symbol
    Yahoo is temporarily failing to serve fundamentals for; this module
    treats that the same as "no data" (None), never as zero/false.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import random

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# --- Concurrency / retry tuning ---------------------------------------
# Conservative on purpose — see module docstring on yfinance/Yahoo's
# lack of a documented rate limit and its tendency to throttle
# aggressive scraping. Lower these further, don't raise them, if a
# refresh starts seeing elevated failure rates.
YFINANCE_MAX_CONCURRENCY = 5
YFINANCE_MAX_RETRIES = 2
YFINANCE_INITIAL_BACKOFF_S = 1.5
YFINANCE_BACKOFF_MULTIPLIER = 3.0
YFINANCE_JITTER_MAX_S = 1.0

_semaphore = asyncio.Semaphore(YFINANCE_MAX_CONCURRENCY)


async def _run_with_retry(func, *args, log_ctx: str, **kwargs):
    """Run a blocking yfinance call in a thread, bounded by the shared
    semaphore, with a small number of retries on exception.

    Returns None (and logs a WARNING) if every attempt fails — never
    raises to the caller. This is deliberate: a single symbol's
    transient yfinance failure must never abort a whole-universe
    refresh (see app.scanners.ingestion / app.scanners.data, both of
    which rely on this contract).
    """
    backoff = YFINANCE_INITIAL_BACKOFF_S
    last_exc: Exception | None = None

    async with _semaphore:
        for attempt in range(1, YFINANCE_MAX_RETRIES + 1):
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — yfinance raises many exception types
                last_exc = exc
                if attempt < YFINANCE_MAX_RETRIES:
                    wait = backoff + random.uniform(0, YFINANCE_JITTER_MAX_S)
                    logger.warning(
                        "yfinance call failed for %s (attempt %d/%d): %s — retrying in %.1fs",
                        log_ctx,
                        attempt,
                        YFINANCE_MAX_RETRIES,
                        last_exc,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    backoff *= YFINANCE_BACKOFF_MULTIPLIER

    logger.warning(
        "yfinance call failed for %s after %d attempt(s): %s",
        log_ctx,
        YFINANCE_MAX_RETRIES,
        last_exc,
    )
    return None


# ---------------------------------------------------------------------------
# Blocking (sync) yfinance calls — always invoked via asyncio.to_thread,
# never directly.
# ---------------------------------------------------------------------------


def _fetch_info_sync(symbol: str) -> dict | None:
    info = yf.Ticker(symbol).info
    if not info or not isinstance(info, dict):
        return None
    return info


def _fetch_earnings_dates_sync(symbol: str, limit: int) -> pd.DataFrame | None:
    df = yf.Ticker(symbol).get_earnings_dates(limit=limit)
    if df is None or df.empty:
        return None
    return df


def _fetch_annual_income_stmt_sync(symbol: str) -> pd.DataFrame | None:
    df = yf.Ticker(symbol).income_stmt
    if df is None or df.empty:
        return None
    return df


def _fetch_institutional_holders_sync(symbol: str) -> pd.DataFrame | None:
    df = yf.Ticker(symbol).institutional_holders
    if df is None or df.empty:
        return None
    return df


def _fetch_history_range_sync(
    symbol: str, start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    # yfinance's `end` is exclusive, so nudge it forward a day to include
    # `end` itself (matching the inclusive [from_date, today] range the
    # caller asked for — same convention app.services.eodhd_client used).
    df = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=(end + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def fetch_info(symbol: str) -> dict | None:
    """Ticker.info — company profile + a large grab-bag of snapshot
    fields (currentPrice, fiftyTwoWeekHigh, floatShares,
    sharesOutstanding, returnOnEquity, heldPercentInstitutions,
    heldPercentInsiders, sector, industry, longName, ...). See
    app.scanners.data.fetch_and_extract_fundamentals_for_symbol for
    exactly which fields are used and how.
    """
    return await _run_with_retry(_fetch_info_sync, symbol, log_ctx=f"info:{symbol}")


async def fetch_earnings_dates(symbol: str, limit: int = 16) -> pd.DataFrame | None:
    """Reported (+ upcoming) quarterly EPS, indexed by report date, with
    a "Reported EPS" column — yfinance's closest equivalent to EODHD's
    Earnings.History. `limit` controls how far back Yahoo looks; 16
    requests should comfortably reach back beyond one year for the C
    criterion's year-ago-quarter comparison in the normal case.
    """
    return await _run_with_retry(
        _fetch_earnings_dates_sync, symbol, limit, log_ctx=f"earnings_dates:{symbol}"
    )


async def fetch_annual_income_stmt(symbol: str) -> pd.DataFrame | None:
    """Annual income statement (line items x fiscal-year-end columns).
    Used only for its "Diluted EPS" / "Basic EPS" row — yfinance's
    closest equivalent to EODHD's Earnings.Annual. See module docstring
    for the ~4-year coverage limitation.
    """
    return await _run_with_retry(
        _fetch_annual_income_stmt_sync, symbol, log_ctx=f"income_stmt:{symbol}"
    )


async def fetch_institutional_holders(symbol: str) -> pd.DataFrame | None:
    """Yahoo's top institutional holders table. See module docstring —
    this is NOT a complete institutional-ownership count.
    """
    return await _run_with_retry(
        _fetch_institutional_holders_sync, symbol, log_ctx=f"institutional_holders:{symbol}"
    )


async def fetch_price_history_range(
    symbol: str, start: datetime.date, end: datetime.date
) -> pd.DataFrame | None:
    """Daily OHLCV bars for `symbol` over [start, end] inclusive, via
    yfinance — used for both individual stocks and the S&P 500 index
    itself (symbol "^GSPC"; see app.scanners.data.MARKET_INDEX_SYMBOL).
    """
    return await _run_with_retry(
        _fetch_history_range_sync, symbol, start, end, log_ctx=f"history_range:{symbol}"
    )