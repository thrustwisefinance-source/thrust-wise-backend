"""
Thin async wrapper around the EODHD end-of-day API with retry/backoff.
Docs: https://eodhd.com/financial-apis/api-for-historical-data-and-volumes
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


async def fetch_eod_history(
    eodhd_symbol: str,
    from_date: datetime.date | None = None,
    to_date: datetime.date | None = None,
) -> list[dict]:
    """
    Fetch daily OHLCV rows for a symbol like "VOO.US".

    Returns a list of dicts:
    {date, open, high, low, close, adjusted_close, volume}

    Retries on 5xx/timeout with exponential backoff.
    Fails immediately on 4xx (bad request, not found, etc.).
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

    last_exception: Exception | None = None
    backoff = INITIAL_BACKOFF_S

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)

                # 4xx — client error, don't retry
                if 400 <= response.status_code < 500:
                    logger.error(
                        "EODHD %s returned %s for %s — not retrying",
                        response.status_code,
                        response.text[:200],
                        eodhd_symbol,
                    )
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()

            if not isinstance(data, list):
                logger.warning(
                    "Unexpected EODHD payload for %s: %r", eodhd_symbol, data
                )
                return []
            return data

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
                eodhd_symbol,
                attempt,
                MAX_RETRIES,
                last_exception,
                wait,
            )
            await asyncio.sleep(wait)
            backoff *= BACKOFF_MULTIPLIER

    logger.error(
        "EODHD request for %s failed after %d attempts: %s",
        eodhd_symbol,
        MAX_RETRIES,
        last_exception,
    )
    raise last_exception  # type: ignore[misc]
