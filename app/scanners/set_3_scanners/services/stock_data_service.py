"""
Stock data for the Technical-Fundamental Composite scanner.

Uses ONLY the existing yfinance wrapper (app.scanners.yfinance_client): no new
data provider. That wrapper already bounds concurrency, retries transient
failures and returns ``None`` instead of raising.

Price history is requested with enough calendar days to yield the required
number of TRADING observations plus a warm-up buffer. (The article's original
code fetched only 5 days of prices while computing a 50-day average; that bug
is intentionally not repeated.)

Fundamental fields come from ``Ticker.info``:
* ``forwardPE``
* ``debtToEquity`` - NOTE: Yahoo reports this as a PERCENTAGE (e.g. 154.6 means
  a ratio of 1.546). It is divided by 100 here so it is comparable with the
  article's "debt-to-equity below 1" rule; the raw provider value is also
  returned. Missing / non-numeric values stay ``None`` - never 0.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.scanners import yfinance_client
from app.scanners.set_3_scanners.services.scoring_service import fetch_target_observations

logger = logging.getLogger(__name__)

CALENDAR_DAYS_PER_TRADING_DAY = 1.6  # weekends + holidays, with margin


@dataclass(frozen=True)
class StockInputs:
    symbol: str
    close: pd.Series | None
    forward_pe: float | None
    debt_to_equity: float | None  # ratio (provider percent / 100)
    debt_to_equity_provider_value: float | None
    messages: list[str] = field(default_factory=list)


def _finite_float(value: Any) -> float | None:
    """float(value) if finite, else None (handles None, 'Infinity', 'N/A')."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def extract_fundamentals(info: dict | None) -> tuple[float | None, float | None, float | None]:
    """(forward_pe, debt_to_equity_ratio, debt_to_equity_provider_value)."""
    if not info:
        return None, None, None
    fpe = _finite_float(info.get("forwardPE"))
    raw_de = _finite_float(info.get("debtToEquity"))
    ratio = None if raw_de is None else raw_de / 100.0
    return fpe, ratio, raw_de


def extract_close_series(df: pd.DataFrame | None) -> pd.Series | None:
    """Clean 'Close' column -> tz-naive date-indexed Series, or None."""
    if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    close.index = idx.normalize()
    close = close[close.notna() & (close > 0)]
    close = close[~close.index.duplicated(keep="last")].sort_index()
    return close if len(close) else None


async def fetch_stock_inputs(symbol: str, ma_period: int, rsi_period: int) -> StockInputs:
    """Fetch prices + fundamentals for one symbol (never raises for provider
    failures; missing pieces are reported through ``messages`` / ``None``)."""
    target = fetch_target_observations(ma_period, rsi_period)
    end = datetime.datetime.now(datetime.timezone.utc).date()
    start = end - datetime.timedelta(days=int(target * CALENDAR_DAYS_PER_TRADING_DAY) + 10)
    hist_df, info = await asyncio.gather(
        yfinance_client.fetch_price_history_range(symbol, start, end),
        yfinance_client.fetch_info(symbol),
    )
    close = extract_close_series(hist_df)
    if close is not None:
        close = close.iloc[-target:]
    fpe, de, raw_de = extract_fundamentals(info)
    messages: list[str] = []
    if close is None:
        messages.append("The data provider returned no price history (invalid symbol or provider failure).")
    if info is None:
        messages.append("The data provider returned no fundamentals for this symbol.")
    return StockInputs(symbol, close, fpe, de, raw_de, messages)
