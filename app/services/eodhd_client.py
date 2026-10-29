"""
Thin async wrapper around the EODHD APIs with retry/backoff.

Historical daily bars: https://eodhd.com/financial-apis/api-for-historical-data-and-volumes
Fundamentals:          https://eodhd.com/financial-apis/stock-etfs-fundamental-data-feeds
Bulk fundamentals:     https://eodhd.com/financial-apis/bulk-api-for-eod-splits-and-dividends
                       (bulk-fundamentals shares the same /bulk-fundamentals/{EXCHANGE}
                       shape, filtered to a `symbols` list for the scanner's use case)

The fundamentals/bulk-fundamentals/index-components functions below were
added for the CANSLIM stock scanner (app/scanners/) — see that package's
module docstrings for exactly which fields of each payload are used and
why. They reuse the same retry/backoff/error-handling helper as the
pre-existing `fetch_eod_history` rather than duplicating the HTTP logic.
"""

import asyncio
import datetime
import logging
import random

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://eodhd.com/api"

# Retry configuration
MAX_RETRIES = 3
INITIAL_BACKOFF_S = 1.0
BACKOFF_MULTIPLIER = 4.0
JITTER_MAX_S = 1.0


def _redact(value: object) -> str:
    """str(value) with the API token masked (httpx exception text embeds the full URL)."""
    text = str(value)
    token = settings.eodhd_api_token
    return text.replace(token, "***") if token else text


async def _get_json_with_retry(url: str, params: dict[str, str], log_ctx: str) -> object:
    """Shared GET-with-retry/backoff core used by every EODHD call below.

    Retries on 5xx/timeout with exponential backoff; fails immediately on
    4xx (bad request, not found, etc.). Returns the parsed JSON body
    as-is (list or dict, depending on endpoint) — callers validate shape.
    """
    last_exception: Exception | None = None
    backoff = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)

                if 400 <= response.status_code < 500:
                    logger.error(
                        "EODHD %s returned %s for %s — not retrying",
                        response.status_code,
                        response.text[:200],
                        log_ctx,
                    )
                    response.raise_for_status()

                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                raise  # Don't retry client errors
            last_exception = exc
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exception = exc

        if attempt < MAX_RETRIES:
            jitter = random.uniform(0, JITTER_MAX_S)
            wait = backoff + jitter
            logger.warning(
                "EODHD request for %s failed (attempt %d/%d): %s — retrying in %.1fs",
                log_ctx,
                attempt,
                MAX_RETRIES,
                _redact(last_exception),
                wait,
            )
            await asyncio.sleep(wait)
            backoff *= BACKOFF_MULTIPLIER

    logger.error(
        "EODHD request for %s failed after %d attempts: %s",
        log_ctx,
        MAX_RETRIES,
        _redact(last_exception),
    )
    raise last_exception  # type: ignore[misc]


async def fetch_eod_history(
    eodhd_symbol: str,
    from_date: datetime.date | None = None,
    to_date: datetime.date | None = None,
) -> list[dict]:
    """
    Fetch daily OHLCV rows for a symbol like "VOO.US".

    Returns a list of dicts:
    {date, open, high, low, close, adjusted_close, volume}
    """
    params: dict[str, str] = {
        "api_token": settings.eodhd_api_token,
        "fmt": "json",
        "period": "d",
    }
    if from_date:
        params["from"] = from_date.isoformat()
    if to_date:
        params["to"] = to_date.isoformat()

    url = f"{BASE_URL}/eod/{eodhd_symbol}"
    data = await _get_json_with_retry(url, params, log_ctx=eodhd_symbol)

    if not isinstance(data, list):
        logger.warning("Unexpected EODHD payload for %s: %r", eodhd_symbol, data)
        return []
    return data


async def fetch_fundamentals(eodhd_symbol: str) -> dict | None:
    """
    Fetch the full EODHD Fundamentals payload for one symbol (e.g. "AAPL.US").

    Returns the raw dict with (among others) the top-level sections used
    by the CANSLIM scanner: General, Highlights, SharesStats, Earnings,
    Holders. Returns None (never a fabricated/empty-but-truthy dict) if
    EODHD has no fundamentals coverage for the symbol (404) or the
    payload isn't a dict.
    """
    params = {"api_token": settings.eodhd_api_token, "fmt": "json"}
    url = f"{BASE_URL}/fundamentals/{eodhd_symbol}"
    try:
        data = await _get_json_with_retry(url, params, log_ctx=eodhd_symbol)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            logger.info("No EODHD fundamentals coverage for %s", eodhd_symbol)
            return None
        raise

    if not isinstance(data, dict) or not data:
        logger.warning(
            "Unexpected EODHD fundamentals payload for %s: %r", eodhd_symbol, data
        )
        return None
    return data


async def fetch_bulk_fundamentals(
    exchange: str, symbols: list[str]
) -> dict[str, dict]:
    """
    Fetch EODHD Fundamentals for many symbols on one exchange in a single
    HTTP call via the bulk-fundamentals endpoint, keyed by plain symbol.

    EODHD's bulk-fundamentals endpoint accepts a comma-separated
    `symbols` filter capped at a couple hundred tickers per call — chunk
    upstream (see app.scanners.data.fetch_universe_fundamentals) if
    calling for a full S&P 500-sized universe. This is what keeps a
    ~500-stock scan to a small number of HTTP calls instead of 500
    sequential /fundamentals/{symbol} requests (see scanner performance
    requirements).

    Symbols with no returned entry are simply absent from the result
    dict — never fabricated.
    """
    if not symbols:
        return {}

    params = {
        "api_token": settings.eodhd_api_token,
        "fmt": "json",
        "symbols": ",".join(symbols),
    }
    url = f"{BASE_URL}/bulk-fundamentals/{exchange}"
    data = await _get_json_with_retry(url, params, log_ctx=f"bulk:{exchange}:{len(symbols)} symbols")

    if not isinstance(data, dict):
        logger.warning(
            "Unexpected EODHD bulk-fundamentals payload for %s: %r", exchange, type(data)
        )
        return {}
    return data


async def fetch_index_fundamentals(index_eodhd_symbol: str) -> dict | None:
    """
    Fetch fundamentals for an index (e.g. "GSPC.INDX"), which for a
    major index includes a "Components" section: a dict keyed by
    constituent order, each value containing at least {Code, Name,
    Sector, Industry, Exchange}.

    Used by app.scanners.universe to source the S&P 500 constituent
    list from EODHD directly rather than hard-coding it (see that
    module's docstring). Returns None if unavailable.
    """
    return await fetch_fundamentals(index_eodhd_symbol)


class EodhdEntitlementError(Exception):
    """EODHD rejected the request because the API token / subscription does not
    cover the endpoint (401/402/403). Carries an actionable, token-free message."""


# EODHD intraday: interval -> maximum (to - from) span per request, in days.
INTRADAY_MAX_SPAN_DAYS = {"1m": 120, "5m": 600, "1h": 7200}


async def fetch_intraday_history(
    eodhd_symbol: str,
    interval: str = "1h",
    from_dt: datetime.datetime | None = None,
    to_dt: datetime.datetime | None = None,
) -> list[dict]:
    """
    Fetch intraday OHLCV bars for a symbol like "SPY.US" from
    https://eodhd.com/api/intraday/{symbol} (raw, UNADJUSTED bars; `datetime`
    is UTC). Requires an EODHD plan that includes intraday data
    (EOD+Intraday / All-in-One); each request costs 5 API calls.

    Returns a list of dicts: {timestamp, datetime, open, high, low, close, volume}.

    Raises:
      ValueError             - unsupported interval or a range beyond EODHD's limit.
      EodhdEntitlementError  - 401/402/403: the plan/token does not include intraday.
    No silent fallback to daily data is ever performed here.
    """
    if interval not in INTRADAY_MAX_SPAN_DAYS:
        raise ValueError(f"Unsupported intraday interval '{interval}'.")
    params: dict[str, str] = {
        "api_token": settings.eodhd_api_token,
        "fmt": "json",
        "interval": interval,
    }
    if from_dt and to_dt:
        span_days = (to_dt - from_dt).total_seconds() / 86400.0
        if span_days > INTRADAY_MAX_SPAN_DAYS[interval]:
            raise ValueError(
                f"EODHD allows at most {INTRADAY_MAX_SPAN_DAYS[interval]} days per {interval} request."
            )
    if from_dt:
        params["from"] = str(int(from_dt.replace(tzinfo=from_dt.tzinfo or datetime.timezone.utc).timestamp()))
    if to_dt:
        params["to"] = str(int(to_dt.replace(tzinfo=to_dt.tzinfo or datetime.timezone.utc).timestamp()))

    url = f"{BASE_URL}/intraday/{eodhd_symbol}"
    try:
        data = await _get_json_with_retry(url, params, log_ctx=f"{eodhd_symbol}:{interval}")
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code in (401, 402, 403):
            raise EodhdEntitlementError(
                f"EODHD denied access to intraday {interval} data for {eodhd_symbol} "
                f"(HTTP {exc.response.status_code}). The configured EODHD plan/token must include "
                "intraday data (EOD+Intraday or All-in-One). Use timeframe=1d, or upgrade the plan."
            ) from exc
        raise

    if not isinstance(data, list):
        logger.warning("Unexpected EODHD intraday payload for %s: %r", eodhd_symbol, type(data))
        return []
    return data
