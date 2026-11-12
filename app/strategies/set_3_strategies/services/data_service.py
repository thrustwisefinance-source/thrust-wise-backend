"""
Data access for the additive Set 3 quantitative strategies.

Only EXISTING providers are used - no new external service is introduced:

* ETF prices          -> the existing ``daily_prices`` table through
                         ``app.routers.deps.load_prices`` (EODHD-ingested).
* ETF P/E             -> ``app.scanners.yfinance_client.fetch_info``
                         (yfinance ``Ticker.info`` -> trailingPE / forwardPE).
* 10-year Treasury    -> the same yfinance wrapper, index ``^TNX`` (CBOE 10-year
  yield                  Treasury note yield, quoted in percent).

Nothing is ever fabricated: when a source is missing, malformed or outside a
plausible range the functions raise ``Set3DataUnavailable`` with a clear,
client-safe message and the strategy reports ``unavailable``.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from app.scanners import yfinance_client
from app.strategies.set_3_strategies.common import prices_to_frame

TREASURY_SYMBOL = "^TNX"
TREASURY_SOURCE = "Yahoo Finance via yfinance, ^TNX (CBOE 10-Year Treasury Note Yield Index)"
EARNINGS_YIELD_SOURCE = "Yahoo Finance via yfinance, Ticker.info.{field} (ETF-level P/E proxy)"
# A US 10-year yield outside (0, 20] percent is treated as a bad / differently
# scaled quote and is DROPPED (never silently rescaled).
TREASURY_MIN_EXCLUSIVE = 0.0
TREASURY_MAX_INCLUSIVE = 20.0
# If more than this share of observations is rejected the whole series is
# distrusted (probable quote-convention change) and reported unavailable.
TREASURY_MAX_DROPPED_SHARE = 0.2


class Set3DataUnavailable(Exception):
    """A required data source is missing / unusable (message is client-safe)."""


@dataclass(frozen=True)
class EarningsYieldInput:
    pe_ratio: float
    pe_basis: str  # "trailing" | "forward"
    field_name: str
    source: str
    as_of: datetime.date  # snapshot (fetch) date, UTC
    is_proxy: bool = True


@dataclass(frozen=True)
class TreasuryYieldInput:
    series: pd.Series  # percent units, tz-naive daily DatetimeIndex, ascending
    source: str
    dropped_observations: int
    as_of: datetime.date  # newest observation date


def utc_today() -> datetime.date:
    return datetime.datetime.now(datetime.timezone.utc).date()


# ---------------------------------------------------------------------------
# ETF prices (existing DB)
# ---------------------------------------------------------------------------


async def load_etf_price_frame(db: AsyncSession, symbol: str) -> pd.DataFrame:
    """Adjusted-close frame (column ``price``) indexed by date; may be empty."""
    # Imported lazily so the calculation modules stay importable without the
    # database / settings layer (same convention as set_3_strategies/common.py).
    from app.routers.deps import load_prices

    return prices_to_frame(await load_prices(db, symbol))


# ---------------------------------------------------------------------------
# Parsing helpers (pure, unit-tested)
# ---------------------------------------------------------------------------


def positive_finite(value: Any) -> float | None:
    """float(value) when it is a finite number > 0, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def parse_treasury_close(close: pd.Series) -> tuple[pd.Series, int]:
    """
    Clean a ``^TNX`` close series. Returns (series, dropped_count).

    Values must be finite and in (0, 20] percent; others are dropped and
    counted. Raises ``Set3DataUnavailable`` if nothing usable remains or if
    too many values were rejected to trust the quote convention.
    """
    values = pd.to_numeric(close, errors="coerce").astype(float)
    idx = pd.DatetimeIndex(values.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    values.index = idx.normalize()
    values = values[~values.index.duplicated(keep="last")].sort_index()
    total = int(values.notna().sum())
    values = values[values.notna()]
    ok = (values > TREASURY_MIN_EXCLUSIVE) & (values <= TREASURY_MAX_INCLUSIVE)
    dropped = int((~ok).sum())
    cleaned = values[ok]
    if cleaned.empty:
        raise Set3DataUnavailable(
            "No valid 10-year Treasury yield observations were returned by the provider."
        )
    if total and dropped / total > TREASURY_MAX_DROPPED_SHARE:
        raise Set3DataUnavailable(
            "10-year Treasury yield data from the provider looks mis-scaled "
            f"({dropped} of {total} observations outside 0-{TREASURY_MAX_INCLUSIVE:g}%); "
            "refusing to use it."
        )
    return cleaned, dropped


# ---------------------------------------------------------------------------
# External sources
# ---------------------------------------------------------------------------


async def fetch_etf_earnings_yield(symbol: str, pe_basis: str) -> EarningsYieldInput:
    """ETF-level P/E from yfinance (a PROXY for the portfolio's aggregate P/E)."""
    field_name = "trailingPE" if pe_basis == "trailing" else "forwardPE"
    info = await yfinance_client.fetch_info(symbol)
    if not info:
        raise Set3DataUnavailable(
            f"The data provider returned no fundamentals for {symbol}; "
            "an earnings yield cannot be computed."
        )
    pe = positive_finite(info.get(field_name))
    if pe is None:
        raise Set3DataUnavailable(
            f"No valid positive {pe_basis} P/E ({field_name}) is available for {symbol} "
            "from the data provider; an earnings yield cannot be computed."
        )
    return EarningsYieldInput(
        pe_ratio=pe,
        pe_basis=pe_basis,
        field_name=field_name,
        source=EARNINGS_YIELD_SOURCE.format(field=field_name),
        as_of=utc_today(),
    )


async def fetch_treasury_yield_history(lookback_years: int) -> TreasuryYieldInput:
    """Daily 10-year Treasury yield (percent) over the requested lookback."""
    end = utc_today()
    start = end - datetime.timedelta(days=int(lookback_years * 365.25) + 7)
    df = await yfinance_client.fetch_price_history_range(TREASURY_SYMBOL, start, end)
    if df is None or "Close" not in getattr(df, "columns", []):
        raise Set3DataUnavailable(
            "The 10-year Treasury yield could not be retrieved from the data provider."
        )
    series, dropped = parse_treasury_close(df["Close"])
    return TreasuryYieldInput(
        series=series,
        source=TREASURY_SOURCE,
        dropped_observations=dropped,
        as_of=series.index[-1].date(),
    )
