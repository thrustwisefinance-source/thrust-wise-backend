"""
Stock Scanners API — architecturally separate from /api/strategies
(single-symbol / ETF-universe analytics, see routers/strategy.py and the
legacy routers/canslim.py). This is the CANSLIM STOCK SCANNER: a
cross-sectional screen over a real stock universe (S&P 500 today), using
actual fundamental + market data, never a buy/sell/hold signal — see
app.scanners.canslim module docstring for the full methodology.

    STRATEGIES  -> /api/strategies/*      (existing, untouched)
    SCANNERS    -> /api/scanners/*        (this router)

Only the endpoints actually needed by a "Scanners" frontend screen are
exposed, per the task spec: list available universes, run/read the
CANSLIM scan (with the filters a scanner table needs), and look up one
symbol's detail without re-running/re-filtering client-side.
"""

import datetime
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.scanners import canslim as canslim_engine
from app.scanners import data as scanner_data
from app.scanners import universe as scanner_universe
from app.schemas import ApiResponse, CanslimScannerResponse, StockUniverseListResponse
from app.services import cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanners", tags=["scanners"])

DATA_SOURCE_LIMITATIONS = [
    (
        "This scanner's stock data (constituent list, fundamentals, "
        "prices) comes entirely from yfinance/Yahoo Finance — a free, "
        "unofficial data source, NOT EODHD (EODHD is used only by "
        "ThrustWise's separate ETF/index ingestion, e.g. GET "
        "/api/strategies/canslim). yfinance has no enterprise SLA; data "
        "for a given symbol or field can occasionally be missing or "
        "delayed — see each criterion's `status`/`data_source` for "
        "exactly what was and wasn't available."
    ),
    (
        "C (current quarterly earnings) and A (annual earnings growth) "
        "use yfinance's Ticker.get_earnings_dates() (Reported EPS) and "
        "Ticker.income_stmt (Diluted/Basic EPS) + Ticker.info's "
        "returnOnEquity. Yahoo typically exposes only ~4 fiscal years "
        "of annual data (vs. the article's 3-5 year window) and a "
        "rolling window of recent quarters. A zero/negative denominator "
        "(e.g. a loss-making prior period) is reported as "
        "NOT_APPLICABLE, never a fabricated or misleading percentage."
    ),
    (
        "N (New) only evaluates the measurable price-highs component "
        "(proximity to the 52-week high). New products, services, or "
        "management changes are qualitative company events ThrustWise "
        "cannot verify and are NOT scored — see each result's N "
        "explanation."
    ),
    (
        "S (Supply and Demand) uses yfinance's Ticker.info.floatShares "
        "against the article's 25,000,000-share threshold. Insider "
        "buying and buyback activity are NOT evaluated — ThrustWise "
        "does not ingest a time series of insider transactions or "
        "historical shares outstanding."
    ),
    (
        "L (Leader) is ThrustWise's OWN calculated relative-strength "
        "percentile (trailing 12-month total return, ranked against "
        "this scan's universe) — labeled 'Calculated Relative Strength "
        "Percentile', NOT an official IBD RS Rating, which ThrustWise "
        "has no data source for."
    ),
    (
        "I (Institutional Sponsorship) uses yfinance's "
        "Ticker.institutional_holders — Yahoo's own top-holders table, "
        "NOT a complete count of every institutional owner — against a "
        "disclosed interpretive minimum (see "
        "app.scanners.canslim.CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS) "
        "since the source article gives no exact number for 'a few' "
        "institutional investors. 'Increasing over time' and "
        "'over-owned' are informational only, not gating."
    ),
    (
        "M (Market Direction) is evaluated ONCE per scan from the S&P "
        "500 INDEX's own price (yfinance '^GSPC') vs. its 200-day SMA, "
        "and applied identically to every stock — never a per-stock "
        "price-vs-its-own-SMA200 proxy."
    ),
    (
        "This is a screening/ranking tool, not investment advice. "
        "`qualifies` means the stock satisfies the configured screening "
        "thresholds — it is never a buy, sell, or hold recommendation."
    ),
]


def _scan_cache_key(universe_key: str) -> str:
    return f"tw:v1:scanner:canslim:{universe_key}"


async def _run_canslim_scan(db: AsyncSession, universe_key: str) -> dict:
    """Cross-sectional CANSLIM scan for one universe — the expensive
    compute wrapped by cache.get_or_compute in the endpoint below.
    """
    members = await scanner_universe.get_stock_universe(db, universe_key)
    if not members:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Universe '{universe_key}' has not been ingested yet. "
                "Trigger a refresh via POST /api/admin/scanners/refresh."
            ),
        )

    symbols = [m.symbol for m in members]
    member_by_symbol = {m.symbol: m for m in members}

    fundamentals_by_symbol = await scanner_data.load_fundamentals_snapshots(db, symbols)
    prices_by_symbol = await scanner_data.load_prices_for_symbols(db, symbols)

    price_inputs_by_symbol = {}
    for symbol, prices in prices_by_symbol.items():
        derived = scanner_data.compute_price_derived_inputs(prices)
        if derived is not None:
            price_inputs_by_symbol[symbol] = derived

    returns_by_symbol = {
        symbol: inputs.trailing_return
        for symbol, inputs in price_inputs_by_symbol.items()
        if inputs.trailing_return is not None
    }
    rs_by_symbol = canslim_engine.rank_relative_strength(returns_by_symbol)

    index_price, index_sma200 = await scanner_data.compute_market_direction_inputs(db)
    market_direction = canslim_engine.compute_market_direction(index_price, index_sma200)

    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    results = []
    unavailable_symbols = []

    for symbol in symbols:
        snapshot = fundamentals_by_symbol.get(symbol)
        price_inputs = price_inputs_by_symbol.get(symbol)

        if snapshot is None and price_inputs is None:
            unavailable_symbols.append(symbol)
            continue

        annual_eps_history = scanner_data.annual_eps_history_from_snapshot(snapshot)

        c = canslim_engine.compute_criterion_c(
            snapshot.quarterly_eps_current if snapshot else None,
            snapshot.quarterly_eps_current_period if snapshot else None,
            snapshot.quarterly_eps_year_ago if snapshot else None,
            snapshot.quarterly_eps_year_ago_period if snapshot else None,
        )
        a = canslim_engine.compute_criterion_a(
            annual_eps_history,
            snapshot.return_on_equity_ttm if snapshot else None,
        )
        n = canslim_engine.compute_criterion_n(
            price_inputs.current_price if price_inputs else None,
            price_inputs.week_52_high if price_inputs else None,
        )
        s = canslim_engine.compute_criterion_s(
            snapshot.shares_float if snapshot else None
        )
        l = canslim_engine.compute_criterion_l(rs_by_symbol.get(symbol))
        i = canslim_engine.compute_criterion_i(
            snapshot.institutional_holders_count if snapshot else None,
            snapshot.percent_institutions if snapshot else None,
        )

        member = member_by_symbol[symbol]
        company_name = (snapshot.company_name if snapshot else None) or member.name
        sector = (snapshot.sector if snapshot else None) or member.sector
        industry = (snapshot.industry if snapshot else None) or member.industry

        result = canslim_engine.assemble_stock_result(
            symbol=symbol,
            company_name=company_name,
            sector=sector,
            industry=industry,
            universe=universe_key,
            criteria={"c": c, "a": a, "n": n, "s": s, "l": l, "i": i, "m": market_direction},
            price_date=price_inputs.price_date if price_inputs else None,
            fundamentals_as_of=(
                snapshot.fundamentals_as_of.isoformat()
                if snapshot and snapshot.fundamentals_as_of
                else None
            ),
            fundamentals_fetched_at=(
                snapshot.fetched_at.isoformat() if snapshot and snapshot.fetched_at else None
            ),
            data_updated_at=now_iso,
        )
        results.append(result)

    results.sort(key=lambda r: (-r["criteria_passed"], r["symbol"]))

    return {
        "universe": universe_key,
        "universe_size": len(symbols),
        "results": results,
        "unavailable_symbols": unavailable_symbols,
        "market_direction": market_direction,
        "data_updated_at": now_iso,
    }


@router.get("/universes", response_model=ApiResponse[StockUniverseListResponse])
async def list_universes(db: AsyncSession = Depends(get_db)):
    """Universes the scanner architecture supports, and whether each has
    been ingested yet (`sizes` of 0 means "not ingested", not "empty
    index" — see app/scanners/universe.py).
    """
    universes = []
    for source in scanner_universe.list_universes():
        members = await scanner_universe.get_stock_universe(db, source.key)
        universes.append(
            {
                "key": source.key,
                "label": source.label,
                "size": len(members),
                "has_data": len(members) > 0,
                "eodhd_index_symbol": source.eodhd_index_symbol,
            }
        )
    return ApiResponse(data={"universes": universes})


@router.get("/canslim", response_model=ApiResponse[CanslimScannerResponse])
async def get_canslim_scanner(
    universe: str = Query(default="sp500", description="Stock universe key, e.g. 'sp500'."),
    min_criteria_passed: int
    | None = Query(
        default=None,
        ge=0,
        le=canslim_engine.CANSLIM_MAX_CRITERIA,
        description="Only return stocks with at least this many of the 7 criteria passed.",
    ),
    only_complete: bool = Query(
        default=False,
        description="Only return stocks with dataQuality == COMPLETE (no unavailable criteria).",
    ),
    sector: str | None = Query(default=None, description="Filter by GICS-style sector, exact match."),
    industry: str | None = Query(default=None, description="Filter by industry, exact match."),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """CANSLIM screening results for every stock in `universe`, ranked by
    criteria passed (descending). Returns the full per-criterion detail
    for every stock (see CanslimStockResult) so the frontend's summary
    table and detail view can both be built from this one response —
    see GET /scanners/canslim/{symbol} only for a single-symbol
    convenience lookup.

    This is a SCREENING result, not investment advice: `qualifies` means
    "meets the configured pass-count threshold", never a buy/sell/hold
    signal. See `dataSourceLimitations` for exactly what is/isn't
    computed and why.
    """
    source = scanner_universe.get_universe_source(universe)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Unknown universe: {universe}")

    universe_key = source.key
    scan = await cache.get_or_compute(
        _scan_cache_key(universe_key), lambda: _run_canslim_scan(db, universe_key)
    )

    results = scan["results"]
    if min_criteria_passed is not None:
        results = [r for r in results if r["criteria_passed"] >= min_criteria_passed]
    if only_complete:
        results = [r for r in results if r["data_quality"] == "COMPLETE"]
    if sector:
        results = [r for r in results if (r.get("sector") or "").lower() == sector.lower()]
    if industry:
        results = [r for r in results if (r.get("industry") or "").lower() == industry.lower()]

    total_matched = len(results)
    page = results[offset : offset + limit]

    data = {
        "universe": universe_key,
        "universe_label": source.label,
        "universe_size": scan["universe_size"],
        "results": page,
        "unavailable_symbols": scan["unavailable_symbols"],
        "market_direction": scan["market_direction"],
        "filters_applied": {
            "min_criteria_passed": min_criteria_passed,
            "only_complete": only_complete,
            "sector": sector,
            "industry": industry,
            "limit": limit,
            "offset": offset,
            "total_matched": total_matched,
        },
        "data_source_limitations": DATA_SOURCE_LIMITATIONS,
        "data_updated_at": scan["data_updated_at"],
    }

    message = None
    if scan["unavailable_symbols"]:
        message = (
            f"{len(scan['unavailable_symbols'])} universe symbol(s) have no data yet: "
            f"{', '.join(scan['unavailable_symbols'][:10])}"
            + ("..." if len(scan["unavailable_symbols"]) > 10 else "")
        )

    return ApiResponse(data=data, message=message)


@router.get("/canslim/{symbol}", response_model=ApiResponse[CanslimScannerResponse])
async def get_canslim_scanner_symbol(
    symbol: str,
    universe: str = Query(default="sp500"),
    db: AsyncSession = Depends(get_db),
):
    """Convenience single-symbol lookup — same computation and cache as
    GET /scanners/canslim, just pre-filtered to one symbol so the
    frontend's stock-detail view doesn't have to fetch/filter the whole
    universe client-side. Returns the same envelope shape with a
    one-element (or zero-element, if the symbol has no data at all)
    `results` list.
    """
    source = scanner_universe.get_universe_source(universe)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Unknown universe: {universe}")

    universe_key = source.key
    symbol = symbol.upper()

    scan = await cache.get_or_compute(
        _scan_cache_key(universe_key), lambda: _run_canslim_scan(db, universe_key)
    )

    match = next((r for r in scan["results"] if r["symbol"] == symbol), None)
    if match is None and symbol not in scan["unavailable_symbols"]:
        raise HTTPException(
            status_code=404,
            detail=f"{symbol} is not a member of the '{universe_key}' universe.",
        )

    data = {
        "universe": universe_key,
        "universe_label": source.label,
        "universe_size": scan["universe_size"],
        "results": [match] if match else [],
        "unavailable_symbols": [symbol] if match is None else [],
        "market_direction": scan["market_direction"],
        "filters_applied": {"symbol": symbol},
        "data_source_limitations": DATA_SOURCE_LIMITATIONS,
        "data_updated_at": scan["data_updated_at"],
    }
    message = None if match else f"{symbol} has no scannable data yet."
    return ApiResponse(data=data, message=message)
