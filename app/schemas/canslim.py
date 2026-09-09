"""
CANSLIM Screener response shapes.

Analytical / screening data only — a score and its underlying criteria,
never a buy/sell/hold recommendation or guarantee of any outcome. Same
"no signal, no action" convention as app.schemas.strategy.
"""

from app.schemas.common import CamelModel


class CanslimResult(CamelModel):
    """CANSLIM score for a single symbol.

    C, A, S, and I currently always come back null (`unavailable`) —
    ThrustWise has no fundamentals/ownership data source yet (see
    services.canslim module docstring). N, L, and M are computed from
    the symbol's own price history. `score` only counts criteria that
    were actually evaluated (`criteria_evaluated`); see
    `unavailable_criteria` for exactly which ones were skipped, and
    `data_complete` for whether all seven were available.

    A, L, and M use the source article's documented simplified proxies
    (revenue growth + ROE; percent above the 52-week low; price vs.
    SMA200) rather than the official CANSLIM methodology — see
    services.canslim module docstring for the full rationale.
    """

    symbol: str
    score: int
    max_score: int
    minimum_score: int
    meets_minimum_score: bool
    criteria_evaluated: int
    data_complete: bool

    c_pass: bool | None = None
    a_pass: bool | None = None
    n_pass: bool | None = None
    s_pass: bool | None = None
    l_pass: bool | None = None
    i_pass: bool | None = None
    m_pass: bool | None = None

    # Underlying values, exposed where available so the result is
    # explainable (never fabricated when the source data is missing).
    earnings_quarterly_growth: float | None = None
    revenue_growth: float | None = None
    return_on_equity: float | None = None
    current_price: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None
    price_to_52w_high_ratio: float | None = None
    float_shares: float | None = None
    relative_strength_from_52w_low_percent: float | None = None
    institutional_ownership: float | None = None
    sma200: float | None = None

    unavailable_criteria: list[str]


class CanslimScreenerResponse(CamelModel):
    """Top-level payload for GET /strategies/canslim.

    `results` is ranked by score (descending), symbol (ascending) as a
    tiebreaker. `universe` documents which symbols were screened —
    ThrustWise's existing ETF registry (see services.canslim module
    docstring for why CANSLIM, designed for individual equities, is
    applied to ThrustWise's existing ETF universe rather than a stock
    universe this codebase does not have). `unavailable_symbols` lists
    any universe symbols with insufficient price history to score at
    all (never a fabricated score for those).
    """

    universe: list[str]
    results: list[CanslimResult]
    unavailable_symbols: list[str]
    max_score: int
    minimum_score: int
    data_source_limitations: list[str]