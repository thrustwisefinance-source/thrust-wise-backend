"""
CANSLIM STOCK SCANNER response shapes.

Separate from app.schemas.canslim (the pre-existing ETF-universe
CANSLIM-proxy schema used by GET /api/strategies/canslim — left
untouched). These schemas back GET /api/scanners/* only.

Purely descriptive/screening data: `qualifies` means "this stock
satisfies the configured screening rules", never a buy/sell/hold signal
or investment recommendation. See app.scanners.canslim module docstring
for the full CANSLIM-letter-by-letter methodology mapping.
"""

from app.schemas.common import CamelModel


class CanslimCriterionResult(CamelModel):
    """One CANSLIM letter's full, explainable result for one stock (or,
    for M, shared across the whole scan — see `criterion` == "M").

    `status` is one of PASS / FAIL / UNAVAILABLE / NOT_APPLICABLE.
    UNAVAILABLE means required data was missing — it is never converted
    to FAIL (see app.scanners.canslim module docstring). NOT_APPLICABLE
    means the data was present but a percentage/ratio comparison against
    it would be undefined or misleading (e.g. a zero/negative
    denominator).
    """

    criterion: str  # "C", "A", "N", "S", "L", "I", "M"
    name: str
    status: str
    value: float | int | None = None
    threshold: float | int | None = None
    unit: str | None = None
    period: str | None = None
    explanation: str
    data_source: str
    as_of: str | None = None


class CanslimStockResult(CamelModel):
    """Full, explainable CANSLIM screening result for one stock.

    `criteria` always has exactly the 7 keys "c","a","n","s","l","i","m"
    (lowercase, matching app.scanners.canslim.CRITERIA_KEYS) so the
    frontend table in the task spec (Symbol | Company | C | A | N | S |
    L | I | M | Passed) can be rendered directly from one record, with
    no follow-up request needed for the detail view either — every
    criterion's full explanation is already here.
    """

    symbol: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    universe: str

    criteria: dict[str, CanslimCriterionResult]

    criteria_passed: int
    criteria_evaluated: int
    criteria_failed: int
    criteria_unavailable: int
    criteria_total: int

    qualifies: bool
    min_criteria_passed: int
    data_quality: str  # COMPLETE | PARTIAL | INSUFFICIENT

    price_date: str | None = None
    fundamental_as_of: str | None = None
    fundamental_fetched_at: str | None = None
    data_updated_at: str | None = None


class CanslimScannerResponse(CamelModel):
    """Top-level payload for GET /api/scanners/canslim.

    `results` is ranked by `criteria_passed` (descending), symbol
    (ascending) as a tiebreaker, already filtered per the request's
    query parameters. `unavailable_symbols` lists universe members that
    could not be scored AT ALL (e.g. no price history yet ingested) —
    distinct from a scored-but-INCOMPLETE result, which still appears in
    `results`. `market_direction` is the ONE shared M-criterion result
    applied to every stock in `results` (see
    app.scanners.canslim.compute_market_direction) — surfaced once at
    the top level in addition to being embedded in every stock's own
    `criteria.m` so the frontend can show it as scan-level context.
    """

    universe: str
    universe_label: str
    universe_size: int
    results: list[CanslimStockResult]
    unavailable_symbols: list[str]
    market_direction: CanslimCriterionResult
    filters_applied: dict
    data_source_limitations: list[str]
    data_updated_at: str | None = None


class StockUniverseInfo(CamelModel):
    key: str
    label: str
    size: int
    has_data: bool
    eodhd_index_symbol: str | None = None


class StockUniverseListResponse(CamelModel):
    universes: list[StockUniverseInfo]
