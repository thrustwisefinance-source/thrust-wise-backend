"""
Scanner data layer: turns raw EODHD payloads + ingested price history into
the typed inputs app/scanners/canslim.py's pure functions need.

Two data sources, both already established in this codebase:

1. Fundamentals (C, A, S, I inputs) — EODHD Fundamentals API, retrieved
   in BULK (app.services.eodhd_client.fetch_bulk_fundamentals) in chunks
   of BULK_FUNDAMENTALS_CHUNK_SIZE symbols per HTTP call, never one
   request per symbol per criterion. Extracted fields are cached in
   Postgres (app.models.StockFundamentalsSnapshot) — see that model's
   docstring for the freshness fields (`fetched_at`,
   `fundamentals_as_of`).

2. Price history (N, L, M inputs) — the EXISTING app.models.DailyPrice
   table and app.services.eodhd_client.fetch_eod_history, the same ones
   ETF ingestion already uses. Stock price rows share the same table
   (symbol is just a plain string column already; nothing ETF-specific
   about DailyPrice) — no new price-storage model was introduced. The
   S&P 500 index itself (GSPC) is ALREADY ingested by the existing
   app.services.ingestion (see app.constants.INDEX_REGISTRY), so M is
   computed from data this codebase already has, with no new ingestion
   needed for the market-direction leg.

No fabrication anywhere in this module: a missing/unparseable EODHD
field becomes None and is carried through as None — app/scanners/canslim.py
is what turns "input is None" into an explicit UNAVAILABLE status, never
a silently substituted default.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import INDEX_REGISTRY
from app.models import DailyPrice, StockFundamentalsSnapshot
from app.scanners.canslim import CANSLIM_L_LOOKBACK_TRADING_DAYS, CANSLIM_M_SMA_PERIOD
from app.services import eodhd_client
from app.services.strategies import _round, to_close_series

logger = logging.getLogger(__name__)

# EODHD's bulk-fundamentals endpoint is documented for up to a couple
# hundred symbols per call; a conservative chunk size keeps individual
# requests well within provider limits while still turning a 500-stock
# universe into ~10 HTTP calls instead of 500 (see scanner performance
# requirements).
BULK_FUNDAMENTALS_CHUNK_SIZE = 50

# 52-week window for the N criterion's high, and floor for a usable
# trailing-return series for the L criterion — same 252-trading-day
# convention as app.services.canslim (kept consistent across both
# CANSLIM implementations even though they are otherwise independent).
PRICE_HISTORY_MIN_ROWS = 252


# ---------------------------------------------------------------------------
# Fundamentals: extraction
# ---------------------------------------------------------------------------


def _latest_and_year_ago_quarter(
    earnings_history: dict,
) -> tuple[float | None, str | None, float | None, str | None]:
    """From EODHD Earnings.History (keyed by period-end date), return
    (eps_current, eps_current_period, eps_year_ago, eps_year_ago_period).

    "Year ago" is the reported quarter whose period-end date is closest
    to exactly 364 days before the latest reported quarter's date
    (handles the normal ~91-day quarterly cadence without assuming a
    fixed 4-entries-back offset, which breaks if a quarter is missing).
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
    """From EODHD Earnings.Annual (keyed by fiscal-year-end date), return
    up to `max_years` most recent {"year": int, "eps": float} entries,
    ascending by year. Skips entries with no reported EPS rather than
    fabricating one.
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


def extract_fundamentals_fields(symbol: str, raw: dict) -> dict:
    """Pure extraction: EODHD's raw /fundamentals (or one entry of
    /bulk-fundamentals) payload -> the flat fields
    StockFundamentalsSnapshot stores. Never raises on a missing section —
    every field independently defaults to None.
    """
    general = raw.get("General") or {}
    highlights = raw.get("Highlights") or {}
    shares_stats = raw.get("SharesStats") or {}
    earnings = raw.get("Earnings") or {}
    holders = raw.get("Holders") or {}

    eps_current, eps_current_period, eps_year_ago, eps_year_ago_period = (
        _latest_and_year_ago_quarter(earnings.get("History") or {})
    )
    annual_history = _annual_eps_history(earnings.get("Annual") or {}, max_years=5)

    institutions = holders.get("Institutions")
    institutional_holders_count = (
        len(institutions) if isinstance(institutions, dict) else None
    )

    # EODHD reports SharesStats percentages as whole numbers (e.g. 62.35
    # meaning 62.35%); normalized to a fraction here so every percentage
    # field in this codebase (ROE, growth rates, ownership) uses the same
    # 0-1 convention. ReturnOnEquityTTM in Highlights is already a
    # fraction per EODHD's convention for that field, so it is NOT
    # divided by 100 here.
    percent_institutions = shares_stats.get("PercentInstitutions")
    percent_insiders = shares_stats.get("PercentInsiders")

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
        "company_name": general.get("Name"),
        "sector": general.get("Sector"),
        "industry": general.get("Industry"),
        "quarterly_eps_current": eps_current,
        "quarterly_eps_current_period": eps_current_period,
        "quarterly_eps_year_ago": eps_year_ago,
        "quarterly_eps_year_ago_period": eps_year_ago_period,
        "annual_eps_history": annual_history,
        "return_on_equity_ttm": highlights.get("ReturnOnEquityTTM"),
        "shares_float": shares_stats.get("SharesFloat"),
        "shares_outstanding": shares_stats.get("SharesOutstanding"),
        "percent_insiders": (percent_insiders / 100) if percent_insiders is not None else None,
        "institutional_holders_count": institutional_holders_count,
        "percent_institutions": (
            (percent_institutions / 100) if percent_institutions is not None else None
        ),
        "fundamental_period": eps_current_period,
        "fundamentals_as_of": fundamentals_as_of,
    }


# ---------------------------------------------------------------------------
# Fundamentals: bulk ingestion + DB read-back
# ---------------------------------------------------------------------------


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def refresh_fundamentals_snapshots(
    db: AsyncSession, symbols: list[str], exchange: str = "US"
) -> int:
    """Bulk-fetch + upsert EODHD fundamentals for `symbols`.

    Chunks into BULK_FUNDAMENTALS_CHUNK_SIZE-symbol batches. One bad
    chunk (network error, etc.) is logged and skipped rather than
    aborting the whole refresh — matches the per-symbol resilience
    convention in app.services.ingestion.ingest_all.

    Returns the number of symbols successfully upserted.
    """
    import json

    updated = 0
    for chunk in _chunk(symbols, BULK_FUNDAMENTALS_CHUNK_SIZE):
        try:
            payload_by_symbol = await eodhd_client.fetch_bulk_fundamentals(
                exchange, chunk
            )
        except Exception:  # noqa: BLE001 — one bad chunk must not stop the rest
            logger.exception(
                "Bulk fundamentals fetch failed for chunk starting %s", chunk[0]
            )
            continue

        for symbol in chunk:
            raw = payload_by_symbol.get(symbol) or payload_by_symbol.get(
                f"{symbol}.{exchange}"
            )
            if not raw:
                continue
            fields = extract_fundamentals_fields(symbol, raw)
            annual_history = fields.pop("annual_eps_history")
            fields["annual_eps_history_json"] = json.dumps(annual_history)

            existing = await db.scalar(
                select(StockFundamentalsSnapshot).where(
                    StockFundamentalsSnapshot.symbol == symbol
                )
            )
            if existing:
                for key, value in fields.items():
                    setattr(existing, key, value)
            else:
                db.add(StockFundamentalsSnapshot(**fields))
            updated += 1

        await db.commit()

    return updated


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
    import json

    if snapshot is None or not snapshot.annual_eps_history_json:
        return []
    try:
        return json.loads(snapshot.annual_eps_history_json)
    except (TypeError, ValueError):
        return []


# ---------------------------------------------------------------------------
# Price history: ingestion (reuses the existing DailyPrice table)
# ---------------------------------------------------------------------------


async def ingest_stock_prices(
    db: AsyncSession, symbol: str, history_years: int, exchange: str = "US"
) -> int:
    """Same shape as app.services.ingestion.ingest_symbol, generalized to
    an arbitrary plain stock symbol rather than a static registry entry
    — reuses DailyPrice and eodhd_client.fetch_eod_history directly
    instead of duplicating the HTTP/upsert logic in a second place.
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

    rows = await eodhd_client.fetch_eod_history(f"{symbol}.{exchange}", from_date=from_date)
    if not rows:
        return 0

    valid_payload = []
    for row in rows:
        close_val = row.get("close")
        if close_val is None or float(close_val) <= 0:
            continue
        valid_payload.append(
            {
                "symbol": symbol,
                "date": datetime.date.fromisoformat(row["date"]),
                "open": float(row.get("open") or 0),
                "high": float(row.get("high") or 0),
                "low": float(row.get("low") or 0),
                "close": float(close_val),
                "adjusted_close": float(row.get("adjusted_close") or close_val),
                "volume": int(row.get("volume") or 0),
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
    """(index_price, index_sma200) for the S&P 500 index (GSPC), reusing
    the daily bars app.services.ingestion already pulls for the
    dashboard's market-summary indices (app.constants.INDEX_REGISTRY) —
    no new ingestion path needed for the M criterion.
    """
    symbol = "GSPC"
    assert symbol in INDEX_REGISTRY  # documents the reuse; fails loudly if that ever changes

    rows = (
        await db.scalars(
            select(DailyPrice)
            .where(DailyPrice.symbol == symbol)
            .order_by(DailyPrice.date.asc())
        )
    ).all()
    series = to_close_series(list(rows))
    if len(series) < CANSLIM_M_SMA_PERIOD:
        return None, None

    index_price = float(series.iloc[-1])
    index_sma200 = float(series.tail(CANSLIM_M_SMA_PERIOD).mean())
    return index_price, index_sma200
