"""
Scanner data layer: turns raw yfinance payloads + ingested price history
into the typed inputs app/scanners/canslim.py's pure functions need.

THIS MODULE NO LONGER USES EODHD AT ALL. Two data sources, both via
app.scanners.yfinance_client:

1. Fundamentals (C, A, S, I inputs) — yfinance's per-symbol Ticker.info /
   get_earnings_dates() / income_stmt / institutional_holders (there is
   no yfinance bulk-fundamentals endpoint, unlike EODHD's, so this is
   necessarily one set of calls per symbol — see
   `fetch_and_extract_fundamentals_for_symbol` and
   `refresh_fundamentals_snapshots` for the bounded-concurrency,
   per-symbol-resilient orchestration). Extracted fields are cached in
   Postgres (app.models.StockFundamentalsSnapshot) exactly as before —
   see that model's docstring for the freshness fields (`fetched_at`,
   `fundamentals_as_of`).

   IMPORTANT: the quarter-matching and multi-year-growth CALCULATIONS
   themselves (`_latest_and_year_ago_quarter`, `_annual_eps_history`) are
   UNCHANGED from before this migration — they operate on a generic
   {date_str: {"epsActual": float}} shape that used to be built directly
   from EODHD's Earnings.History/Earnings.Annual payload shape and is now
   built from yfinance's very different payload shapes by
   `_quarterly_eps_dict_from_earnings_dates` /
   `_annual_eps_dict_from_income_stmt` below. Only the data ACQUISITION
   and RESHAPING changed; the actual CANSLIM math did not.

2. Price history (N, L, M inputs) — yfinance daily bars
   (app.scanners.yfinance_client.fetch_price_history_range), stored in
   the EXISTING app.models.DailyPrice table (symbol is just a plain
   string column; nothing ETF-specific about it — stock price rows
   share the same table the OLD ETF pipeline uses, with no new
   price-storage model introduced). The S&P 500 INDEX itself is now
   ALSO ingested via yfinance (symbol "^GSPC", see MARKET_INDEX_SYMBOL
   below) by THIS module's ingestion path — deliberately a different
   daily_prices symbol than the OLD ETF/dashboard pipeline's
   EODHD-sourced "GSPC" (app.constants.INDEX_REGISTRY /
   app.services.ingestion), so the M criterion never depends on EODHD,
   directly or indirectly, and the two pipelines' rows never collide.

No fabrication anywhere in this module: a missing/unparseable yfinance
field becomes None and is carried through as None — app/scanners/canslim.py
is what turns "input is None" into an explicit UNAVAILABLE status, never
a silently substituted default (0, False, 0%).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import DailyPrice, StockFundamentalsSnapshot
from app.scanners import yfinance_client
from app.scanners.canslim import CANSLIM_L_LOOKBACK_TRADING_DAYS, CANSLIM_M_SMA_PERIOD
from app.services.strategies import _round, to_close_series

logger = logging.getLogger(__name__)

# yfinance has no bulk-fundamentals endpoint, so fundamentals are fetched
# one symbol at a time — this bounds how many symbols are in flight at
# once (on top of, and effectively tighter than,
# app.scanners.yfinance_client.YFINANCE_MAX_CONCURRENCY, since each
# symbol here issues several yfinance calls concurrently via
# asyncio.gather in fetch_and_extract_fundamentals_for_symbol).
FUNDAMENTALS_CONCURRENCY = 5

# The S&P 500 index itself, via yfinance — used for the M criterion. See
# module docstring for why this is a DIFFERENT daily_prices symbol than
# the OLD ETF/dashboard pipeline's EODHD-sourced "GSPC".
MARKET_INDEX_SYMBOL = "^GSPC"

# 52-week window for the N criterion's high, and floor for a usable
# trailing-return series for the L criterion — same 252-trading-day
# convention as app.services.canslim (kept consistent across both
# CANSLIM implementations even though they are otherwise independent).
PRICE_HISTORY_MIN_ROWS = 252


# ---------------------------------------------------------------------------
# Fundamentals: generic EPS-history parsing (UNCHANGED — see module
# docstring. These operate on a {date_str: {"epsActual": float}} shape,
# not on any EODHD- or yfinance-specific payload directly.)
# ---------------------------------------------------------------------------


def _latest_and_year_ago_quarter(
    earnings_history: dict,
) -> tuple[float | None, str | None, float | None, str | None]:
    """From a {date_str: {"epsActual": float}} dict, return
    (eps_current, eps_current_period, eps_year_ago, eps_year_ago_period).

    "Year ago" is the reported quarter whose date is closest to exactly
    364 days before the latest reported quarter's date (handles the
    normal ~91-day quarterly cadence without assuming a fixed
    4-entries-back offset, which breaks if a quarter is missing).
    """
    entries = []
    for date_str, entry in (earnings_history or {}).items():
        eps_actual = entry.get("epsActual") if isinstance(entry, dict) else None
        if eps_actual is None:
            continue
        try:
            date = datetime.date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            continue
        entries.append((date, float(eps_actual)))

    if not entries:
        return None, None, None, None

    entries.sort(key=lambda pair: pair[0], reverse=True)
    latest_date, latest_eps = entries[0]

    target = latest_date - datetime.timedelta(days=364)
    best = min(
        entries[1:],
        key=lambda pair: abs((pair[0] - target).days),
        default=None,
    )
    # Require the match to be within ~45 days of exactly one year prior,
    # otherwise it isn't really "the same quarter a year ago".
    if best is None or abs((best[0] - target).days) > 45:
        return latest_eps, latest_date.isoformat(), None, None

    return latest_eps, latest_date.isoformat(), best[1], best[0].isoformat()


def _annual_eps_history(earnings_annual: dict, max_years: int) -> list[dict]:
    """From a {date_str: {"epsActual": float}} dict (fiscal-year-end
    keyed), return up to `max_years` most recent {"year": int, "eps":
    float} entries, ascending by year. Skips entries with no reported
    EPS rather than fabricating one.
    """
    entries = []
    for date_str, entry in (earnings_annual or {}).items():
        eps_actual = entry.get("epsActual") if isinstance(entry, dict) else None
        if eps_actual is None:
            continue
        try:
            date = datetime.date.fromisoformat(date_str[:10])
        except (ValueError, TypeError):
            continue
        entries.append((date, float(eps_actual)))

    entries.sort(key=lambda pair: pair[0])
    trimmed = entries[-max_years:]
    return [{"year": date.year, "eps": eps} for date, eps in trimmed]


# ---------------------------------------------------------------------------
# Fundamentals: yfinance payload -> the generic {date_str: {"epsActual"}}
# shape the parsers above expect
# ---------------------------------------------------------------------------


def _quarterly_eps_dict_from_earnings_dates(df: pd.DataFrame | None) -> dict:
    """yfinance `Ticker.get_earnings_dates()` -> the same
    {date_str: {"epsActual": float}} shape `_latest_and_year_ago_quarter`
    expects. Rows with no reported EPS yet (future/upcoming earnings
    dates, or Yahoo hasn't posted the actual yet) are skipped, never
    fabricated as 0.
    """
    if df is None or "Reported EPS" not in df.columns:
        return {}

    result: dict = {}
    for date_index, row in df.iterrows():
        eps = row.get("Reported EPS")
        if eps is None or (isinstance(eps, float) and pd.isna(eps)):
            continue
        try:
            date_str = date_index.date().isoformat()
        except AttributeError:
            continue
        result[date_str] = {"epsActual": float(eps)}
    return result


def _annual_eps_dict_from_income_stmt(df: pd.DataFrame | None) -> dict:
    """yfinance `Ticker.income_stmt` (annual) -> the same
    {date_str: {"epsActual": float}} shape `_annual_eps_history` expects.
    Prefers the 'Diluted EPS' row; falls back to 'Basic EPS' if a
    company's statement doesn't report diluted EPS separately.
    """
    if df is None:
        return {}

    row_label = next(
        (candidate for candidate in ("Diluted EPS", "Basic EPS") if candidate in df.index),
        None,
    )
    if row_label is None:
        return {}

    result: dict = {}
    for column, value in df.loc[row_label].items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        date_str = column.date().isoformat() if hasattr(column, "date") else str(column)[:10]
        result[date_str] = {"epsActual": float(value)}
    return result


async def fetch_and_extract_fundamentals_for_symbol(symbol: str) -> dict | None:
    """One symbol's full fundamentals fetch + extraction via yfinance.

    Issues info/earnings-dates/income-statement/institutional-holders
    calls concurrently (each already individually bounded by
    app.scanners.yfinance_client's own semaphore) then reuses the
    EXISTING, unchanged EPS parsers above. Returns the same flat-field
    dict shape app.scanners.data has always produced for a symbol (still
    carrying "annual_eps_history" as a plain list — the caller JSON-
    encodes it, same as before), or None if yfinance returned nothing
    usable at all for this symbol.
    """
    info, earnings_dates_df, annual_income_df, holders_df = await asyncio.gather(
        yfinance_client.fetch_info(symbol),
        yfinance_client.fetch_earnings_dates(symbol),
        yfinance_client.fetch_annual_income_stmt(symbol),
        yfinance_client.fetch_institutional_holders(symbol),
    )

    if info is None and earnings_dates_df is None and annual_income_df is None:
        return None

    info = info or {}

    quarterly_eps_dict = _quarterly_eps_dict_from_earnings_dates(earnings_dates_df)
    annual_eps_dict = _annual_eps_dict_from_income_stmt(annual_income_df)

    eps_current, eps_current_period, eps_year_ago, eps_year_ago_period = (
        _latest_and_year_ago_quarter(quarterly_eps_dict)
    )
    annual_history = _annual_eps_history(annual_eps_dict, max_years=5)

    institutional_holders_count = len(holders_df) if holders_df is not None else None

    # yfinance's Ticker.info reports heldPercentInstitutions /
    # heldPercentInsiders already as a 0-1 fraction (unlike EODHD's
    # SharesStats, which used a whole number like 62.35 meaning 62.35% —
    # that /100 normalization doesn't apply here).
    percent_institutions = info.get("heldPercentInstitutions")
    percent_insiders = info.get("heldPercentInsiders")

    fundamentals_as_of = None
    if annual_history:
        fundamentals_as_of = datetime.date(annual_history[-1]["year"], 12, 31)
    elif eps_current_period:
        try:
            fundamentals_as_of = datetime.date.fromisoformat(eps_current_period)
        except ValueError:
            fundamentals_as_of = None

    return {
        "symbol": symbol,
        "company_name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "quarterly_eps_current": eps_current,
        "quarterly_eps_current_period": eps_current_period,
        "quarterly_eps_year_ago": eps_year_ago,
        "quarterly_eps_year_ago_period": eps_year_ago_period,
        "annual_eps_history": annual_history,
        "return_on_equity_ttm": info.get("returnOnEquity"),
        "shares_float": info.get("floatShares"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "percent_insiders": percent_insiders,
        "institutional_holders_count": institutional_holders_count,
        "percent_institutions": percent_institutions,
        "fundamental_period": eps_current_period,
        "fundamentals_as_of": fundamentals_as_of,
    }


# ---------------------------------------------------------------------------
# Fundamentals: per-symbol ingestion (bounded concurrency) + DB read-back
# ---------------------------------------------------------------------------


async def refresh_fundamentals_snapshots(
    db: AsyncSession, symbols: list[str]
) -> tuple[int, int]:
    """Per-symbol fundamentals refresh via yfinance.

    No bulk endpoint exists for yfinance (unlike EODHD's
    bulk-fundamentals), so this fetches each symbol independently, with
    FUNDAMENTALS_CONCURRENCY bounding how many are in flight at once.
    Each concurrent task opens its OWN short-lived DB session for its
    write (a SQLAlchemy AsyncSession is not safe for concurrent use from
    multiple coroutines — same convention as
    app.scanners.ingestion._refresh_prices_bounded and
    app.services.ingestion.ingest_all's per-symbol sessions) rather than
    sharing the caller's `db`.

    One bad symbol (no yfinance coverage, transient network error, ...)
    is logged and skipped — it never aborts the rest of the refresh.

    Returns (succeeded, failed) symbol counts.
    """
    semaphore = asyncio.Semaphore(FUNDAMENTALS_CONCURRENCY)
    succeeded = 0
    failed = 0

    async def _one(symbol: str) -> None:
        nonlocal succeeded, failed
        async with semaphore:
            try:
                fields = await fetch_and_extract_fundamentals_for_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the rest
                logger.warning("yfinance failed for %s fundamentals: %s", symbol, exc)
                failed += 1
                return

            if fields is None:
                logger.warning(
                    "%s fundamentals: FAILED (no data returned by yfinance)", symbol
                )
                failed += 1
                return

            annual_history = fields.pop("annual_eps_history")
            fields["annual_eps_history_json"] = json.dumps(annual_history)

            try:
                async with AsyncSessionLocal() as session:
                    existing = await session.scalar(
                        select(StockFundamentalsSnapshot).where(
                            StockFundamentalsSnapshot.symbol == symbol
                        )
                    )
                    if existing:
                        for key, value in fields.items():
                            setattr(existing, key, value)
                    else:
                        session.add(StockFundamentalsSnapshot(**fields))
                    await session.commit()
            except Exception as exc:  # noqa: BLE001 — a DB hiccup on one symbol must not stop the rest
                logger.warning("DB write failed for %s fundamentals: %s", symbol, exc)
                failed += 1
                return

            logger.info("%s fundamentals: OK", symbol)
            succeeded += 1

    await asyncio.gather(*(_one(s) for s in symbols))
    return succeeded, failed


async def load_fundamentals_snapshots(
    db: AsyncSession, symbols: list[str]
) -> dict[str, StockFundamentalsSnapshot]:
    if not symbols:
        return {}
    rows = (
        await db.scalars(
            select(StockFundamentalsSnapshot).where(
                StockFundamentalsSnapshot.symbol.in_(symbols)
            )
        )
    ).all()
    return {row.symbol: row for row in rows}


def annual_eps_history_from_snapshot(snapshot: StockFundamentalsSnapshot | None) -> list[dict]:
    if snapshot is None or not snapshot.annual_eps_history_json:
        return []
    try:
        return json.loads(snapshot.annual_eps_history_json)
    except (TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Price history: ingestion (reuses the existing DailyPrice table)
# ---------------------------------------------------------------------------


async def ingest_stock_prices(db: AsyncSession, symbol: str, history_years: int) -> int:
    """Incremental daily-bar ingestion via yfinance for one plain symbol
    (e.g. "AAPL", or the market index "^GSPC" — see MARKET_INDEX_SYMBOL).
    Same incremental-append shape as before this migration (only fetch
    bars since the last ingested date; a fresh symbol backfills
    `history_years` years), just sourced from yfinance instead of EODHD.
    Reuses the same DailyPrice table and on_conflict_do_nothing upsert
    as the OLD ETF pipeline, but is called ONLY from the scanner's
    ingestion path (app.scanners.ingestion), never from
    app.services.ingestion.

    yfinance ticker symbols need no exchange suffix for US equities/the
    S&P 500 index (unlike EODHD's "AAPL.US" / "GSPC.INDX" convention),
    so `symbol` is used exactly as given.
    """
    from sqlalchemy import func

    last_date = await db.scalar(
        select(func.max(DailyPrice.date)).where(DailyPrice.symbol == symbol)
    )
    today_utc = datetime.datetime.now(datetime.timezone.utc).date()

    if last_date:
        from_date = last_date + datetime.timedelta(days=1)
        if from_date > today_utc:
            return 0
    else:
        from_date = today_utc - datetime.timedelta(days=365 * history_years + 30)

    df = await yfinance_client.fetch_price_history_range(symbol, from_date, today_utc)
    if df is None or df.empty:
        return 0

    valid_payload = []
    for date_index, row in df.iterrows():
        close_val = row.get("Close")
        if close_val is None or pd.isna(close_val) or float(close_val) <= 0:
            continue
        adj_close_val = row.get("Adj Close")
        if adj_close_val is None or (isinstance(adj_close_val, float) and pd.isna(adj_close_val)):
            adj_close_val = close_val
        valid_payload.append(
            {
                "symbol": symbol,
                "date": date_index.date(),
                "open": float(row.get("Open") or 0),
                "high": float(row.get("High") or 0),
                "low": float(row.get("Low") or 0),
                "close": float(close_val),
                "adjusted_close": float(adj_close_val),
                "volume": int(row.get("Volume") or 0),
            }
        )

    if not valid_payload:
        return 0

    stmt = pg_insert(DailyPrice).values(valid_payload)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_daily_prices_symbol_date")
    await db.execute(stmt)
    await db.commit()
    return len(valid_payload)


async def load_prices_for_symbols(
    db: AsyncSession, symbols: list[str]
) -> dict[str, list[DailyPrice]]:
    if not symbols:
        return {}
    rows = (
        await db.scalars(
            select(DailyPrice)
            .where(DailyPrice.symbol.in_(symbols))
            .order_by(DailyPrice.symbol.asc(), DailyPrice.date.asc())
        )
    ).all()
    result: dict[str, list[DailyPrice]] = {s: [] for s in symbols}
    for row in rows:
        result[row.symbol].append(row)
    return result


# ---------------------------------------------------------------------------
# Price-derived inputs (N, L, M)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceDerivedInputs:
    current_price: float | None
    price_date: str | None
    week_52_high: float | None
    trailing_return: float | None  # over CANSLIM_L_LOOKBACK_TRADING_DAYS, for L ranking


def compute_price_derived_inputs(prices: list[DailyPrice]) -> PriceDerivedInputs | None:
    series = to_close_series(prices)
    if len(series) < PRICE_HISTORY_MIN_ROWS:
        return None

    current_price = float(series.iloc[-1])
    price_date = series.index[-1].date().isoformat()
    window = series.tail(PRICE_HISTORY_MIN_ROWS)
    week_52_high = float(window.max())

    lookback = series.tail(CANSLIM_L_LOOKBACK_TRADING_DAYS)
    trailing_return = (
        float(lookback.iloc[-1] / lookback.iloc[0] - 1) if len(lookback) >= 2 else None
    )

    return PriceDerivedInputs(
        current_price=_round(current_price),
        price_date=price_date,
        week_52_high=_round(week_52_high),
        trailing_return=trailing_return,
    )


async def compute_market_direction_inputs(
    db: AsyncSession,
) -> tuple[float | None, float | None]:
    """(index_price, index_sma200) for the S&P 500 index, sourced from
    yfinance's "^GSPC" ticker and ingested ONLY by this scanner's own
    ingestion path (app.scanners.ingestion) into the daily_prices table
    under the MARKET_INDEX_SYMBOL ("^GSPC") symbol — deliberately
    DIFFERENT from the OLD ETF/dashboard pipeline's EODHD-sourced "GSPC"
    symbol (app.constants.INDEX_REGISTRY), so M never depends on EODHD,
    directly or indirectly, and the two pipelines' rows never collide.
    """
    rows = (
        await db.scalars(
            select(DailyPrice)
            .where(DailyPrice.symbol == MARKET_INDEX_SYMBOL)
            .order_by(DailyPrice.date.asc())
        )
    ).all()
    series = to_close_series(list(rows))
    if len(series) < CANSLIM_M_SMA_PERIOD:
        return None, None

    index_price = float(series.iloc[-1])
    index_sma200 = float(series.tail(CANSLIM_M_SMA_PERIOD).mean())
    return index_price, index_sma200
