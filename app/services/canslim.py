"""
CANSLIM Stock Screener engine.

Cross-sectional screener — unlike every strategy in services/strategies.py
(single-symbol technical indicators) and closer in shape to
services/regime.py (looks across the whole ETF universe together), this
module scores each symbol against the seven CANSLIM criteria and ranks
the universe. It is analytical/descriptive only: a score and its
underlying criteria, never a buy/sell/hold recommendation. See
compute_canslim_score docstring for the full "no recommendation"
convention.

SOURCE-OF-TRUTH NOTE (read before changing any threshold below):
CANSLIM (William O'Neil) is normally:
  C = Current quarterly earnings growth
  A = Annual earnings growth
  N = New products/management/price highs
  S = Supply and demand (share count, volume)
  L = Leader or laggard (RS Rating vs. peers)
  I = Institutional sponsorship
  M = Market direction
The full methodology relies on data ThrustWise does not have (a
proprietary RS Rating, IBD-style earnings estimates, etc.). Per the task
spec, this module implements the SOURCE ARTICLE's simplified
quantitative proxy for each letter, not a complete reproduction of the
official CANSLIM methodology:
  C: earningsQuarterlyGrowth >= 25%
  A: revenueGrowth >= 20% AND returnOnEquity >= 17% (the article's proxy
     for "annual growth" — not trailing annual EPS growth)
  N: current_price / week_52_high >= 0.85 (near a 52-week high)
  S: float_shares < 500,000,000
  L: ((current_price - week_52_low) / week_52_low) * 100 >= 30 (the
     article's proxy for relative strength/leadership — NOT an official
     RS Rating, which ThrustWise has no data source for)
  I: 0.30 <= institutional_ownership <= 0.85
  M: current_price > SMA200 (the article's simplified market-direction
     proxy, applied per-symbol here since ThrustWise has no single
     "the market" index feed beyond the individual ETFs it tracks)
These are documented proxies, not upgrades or corrections to the source
article's simplified implementation — see the task's CANSLIM RULES.

DATA AVAILABILITY NOTE (read before wiring this into a new endpoint):
ThrustWise's only market-data source is EODHD daily OHLCV bars
(app.models.DailyPrice) — there is no fundamentals/ownership data model
or provider anywhere in this codebase (checked: app/models/, app/schemas/,
app/services/eodhd_client.py, app/services/ingestion.py). Per the task
spec ("do NOT fabricate them... do NOT add yfinance"), this module does
NOT invent fundamentals. It is split into two independently testable
layers:
  1. compute_canslim_score — a pure function over explicit keyword
     arguments (no I/O). This is the CANSLIM scoring logic in isolation,
     usable once/if a fundamentals data source is ever integrated.
  2. compute_price_derived_inputs — derives the THREE criteria that are
     actually computable from price history alone (N, L, M) from
     `list[DailyPrice]`, reusing to_close_series from services.strategies
     (no new price-series helper is introduced).
C, A, S, I are always None (documented "unavailable") when scored via
screen_universe below, since no fundamentals/ownership provider exists
in this codebase yet — this is surfaced explicitly in every result's
`unavailable_criteria` field rather than silently defaulting to a pass
or fail.

UNIVERSE NOTE: CANSLIM is designed for individual equities. ThrustWise's
existing "universe" (app.constants.ETF_REGISTRY) is six diversified ETFs,
not individual stocks — there is no stock universe in this codebase.
Per the task spec ("do not add unnecessary APIs... reuse existing
infrastructure"), screen_universe reuses ETF_REGISTRY as the closest
existing analog to a screenable universe rather than introducing a new
stock database. This is a real, documented limitation, not a
substitution of a "different interpretation" of CANSLIM — the scoring
rules themselves are unchanged; only the pool of symbols being screened
is ThrustWise's existing ETF set.
"""

from app.models import DailyPrice
from app.services.strategies import _round, to_close_series

# --- CANSLIM criteria thresholds (source article's simplified values) -----

CANSLIM_C_MIN_EARNINGS_QUARTERLY_GROWTH = 0.25  # 25%
CANSLIM_A_MIN_REVENUE_GROWTH = 0.20  # 20%
CANSLIM_A_MIN_RETURN_ON_EQUITY = 0.17  # 17%
CANSLIM_N_MIN_PRICE_TO_52W_HIGH_RATIO = 0.85
CANSLIM_S_MAX_FLOAT_SHARES = 500_000_000
CANSLIM_L_MIN_RS_PROXY_PERCENT = 30.0
CANSLIM_I_MIN_INSTITUTIONAL_OWNERSHIP = 0.30
CANSLIM_I_MAX_INSTITUTIONAL_OWNERSHIP = 0.85

CANSLIM_MAX_SCORE = 7
CANSLIM_DEFAULT_MINIMUM_SCORE = 5

CANSLIM_CRITERIA_KEYS = ("c", "a", "n", "s", "l", "i", "m")

CANSLIM_CRITERIA_LABELS = {
    "c": "C — Current Quarterly Earnings",
    "a": "A — Annual Growth (revenue + ROE proxy)",
    "n": "N — New / Near 52-Week High",
    "s": "S — Supply (float shares)",
    "l": "L — Leader (relative-strength proxy)",
    "i": "I — Institutional Sponsorship",
    "m": "M — Market Direction (price vs. SMA200)",
}

# Price-history requirement for the price-derived criteria (N, L, M):
# a full trailing year for the 52-week high/low window, which comfortably
# covers the 200-day SMA window used for M as well. Same period-plus-
# buffer sizing convention as MIN_ROWS_* in services.strategies.
CANSLIM_52_WEEK_TRADING_DAYS = 252
CANSLIM_SMA200_PERIOD = 200
MIN_ROWS_CANSLIM_PRICE_INPUTS = CANSLIM_52_WEEK_TRADING_DAYS


def compute_canslim_score(
    *,
    earnings_quarterly_growth: float | None = None,
    revenue_growth: float | None = None,
    return_on_equity: float | None = None,
    current_price: float | None = None,
    week_52_high: float | None = None,
    week_52_low: float | None = None,
    float_shares: float | None = None,
    institutional_ownership: float | None = None,
    sma200: float | None = None,
) -> dict:
    """Pure CANSLIM scoring logic — no I/O, no database, no external
    calls. Every criterion is independently nullable: a criterion whose
    required input(s) are None is scored as unavailable (pass = None,
    not counted toward `score`, listed in `unavailable_criteria`) rather
    than fabricated as a pass or a fail. This is the same "unavailable,
    never fabricated" convention used by every compute_* function in
    services.strategies.

    Returns a dict — never a recommendation, buy/sell signal, or
    guarantee of outcome; `score` is descriptive/analytical only.
    """
    c_pass = (
        earnings_quarterly_growth >= CANSLIM_C_MIN_EARNINGS_QUARTERLY_GROWTH
        if earnings_quarterly_growth is not None
        else None
    )

    a_pass = (
        revenue_growth >= CANSLIM_A_MIN_REVENUE_GROWTH
        and return_on_equity >= CANSLIM_A_MIN_RETURN_ON_EQUITY
        if revenue_growth is not None and return_on_equity is not None
        else None
    )

    price_to_52w_high_ratio = (
        current_price / week_52_high
        if current_price is not None and week_52_high
        else None
    )
    n_pass = (
        price_to_52w_high_ratio >= CANSLIM_N_MIN_PRICE_TO_52W_HIGH_RATIO
        if price_to_52w_high_ratio is not None
        else None
    )

    s_pass = (
        float_shares < CANSLIM_S_MAX_FLOAT_SHARES if float_shares is not None else None
    )

    rs_proxy_percent = (
        ((current_price - week_52_low) / week_52_low) * 100
        if current_price is not None and week_52_low
        else None
    )
    l_pass = (
        rs_proxy_percent >= CANSLIM_L_MIN_RS_PROXY_PERCENT
        if rs_proxy_percent is not None
        else None
    )

    i_pass = (
        CANSLIM_I_MIN_INSTITUTIONAL_OWNERSHIP
        <= institutional_ownership
        <= CANSLIM_I_MAX_INSTITUTIONAL_OWNERSHIP
        if institutional_ownership is not None
        else None
    )

    m_pass = (
        current_price > sma200
        if current_price is not None and sma200 is not None
        else None
    )

    pass_by_key = {
        "c": c_pass,
        "a": a_pass,
        "n": n_pass,
        "s": s_pass,
        "l": l_pass,
        "i": i_pass,
        "m": m_pass,
    }

    score = sum(1 for v in pass_by_key.values() if v is True)
    criteria_evaluated = sum(1 for v in pass_by_key.values() if v is not None)
    data_complete = criteria_evaluated == CANSLIM_MAX_SCORE
    unavailable_criteria = [
        CANSLIM_CRITERIA_LABELS[key]
        for key in CANSLIM_CRITERIA_KEYS
        if pass_by_key[key] is None
    ]

    return {
        "score": score,
        "max_score": CANSLIM_MAX_SCORE,
        "minimum_score": CANSLIM_DEFAULT_MINIMUM_SCORE,
        "meets_minimum_score": score >= CANSLIM_DEFAULT_MINIMUM_SCORE,
        "criteria_evaluated": criteria_evaluated,
        "data_complete": data_complete,
        "c_pass": c_pass,
        "a_pass": a_pass,
        "n_pass": n_pass,
        "s_pass": s_pass,
        "l_pass": l_pass,
        "i_pass": i_pass,
        "m_pass": m_pass,
        "earnings_quarterly_growth": _round(earnings_quarterly_growth, 4),
        "revenue_growth": _round(revenue_growth, 4),
        "return_on_equity": _round(return_on_equity, 4),
        "current_price": _round(current_price),
        "week_52_high": _round(week_52_high),
        "week_52_low": _round(week_52_low),
        "price_to_52w_high_ratio": _round(price_to_52w_high_ratio, 4),
        "float_shares": float_shares,
        "relative_strength_from_52w_low_percent": _round(rs_proxy_percent),
        "institutional_ownership": _round(institutional_ownership, 4),
        "sma200": _round(sma200),
        "unavailable_criteria": unavailable_criteria,
    }


def compute_price_derived_inputs(prices: list[DailyPrice]) -> dict | None:
    """The three CANSLIM inputs derivable purely from price history — N
    (52-week-high proximity), L (the article's relative-strength-from-
    52-week-low proxy), and M (price vs. SMA200) — extracted from
    `list[DailyPrice]` the same way every strategy in services.strategies
    derives its inputs. Returns None when there isn't a full trailing
    year of history yet (see MIN_ROWS_CANSLIM_PRICE_INPUTS) — same
    "unavailable" convention as everywhere else in this codebase, never
    a fabricated 52-week range from partial data.

    Fundamentals-only inputs (earnings/revenue growth, ROE, float
    shares, institutional ownership) are NOT included here — they have
    no price-history equivalent, see module docstring.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_CANSLIM_PRICE_INPUTS:
        return None

    window = series.tail(CANSLIM_52_WEEK_TRADING_DAYS)
    current_price = float(series.iloc[-1])
    week_52_high = float(window.max())
    week_52_low = float(window.min())
    sma200 = float(series.tail(CANSLIM_SMA200_PERIOD).mean())

    return {
        "current_price": current_price,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "sma200": sma200,
    }


def score_symbol_from_prices(symbol: str, prices: list[DailyPrice]) -> dict | None:
    """CANSLIM score for one symbol using only ThrustWise's existing
    price-history data source (N, L, M — see module docstring for why C,
    A, S, I are unavailable). Returns None when there isn't enough price
    history yet, same "unavailable" convention as every strategy in this
    codebase — never scored from a partial/fabricated price range.
    """
    price_inputs = compute_price_derived_inputs(prices)
    if price_inputs is None:
        return None

    result = compute_canslim_score(**price_inputs)
    result["symbol"] = symbol
    return result


def screen_universe(prices_by_symbol: dict[str, list[DailyPrice]]) -> dict:
    """CANSLIM screener across `prices_by_symbol` (see module docstring —
    ThrustWise's existing ETF universe is used, not individual stocks, so
    the "leadership" and "supply/demand" criteria read differently than
    they would for equities; this is documented, not concealed).

    Results are ranked by score (descending), symbol (ascending) as a
    tiebreaker — filtering by a minimum score is left to the caller
    (router), since which threshold to apply can vary by request; this
    function always returns the full ranked universe plus which symbols
    had insufficient price history.
    """
    results = []
    unavailable_symbols = []
    for symbol, prices in prices_by_symbol.items():
        scored = score_symbol_from_prices(symbol, prices)
        if scored is None:
            unavailable_symbols.append(symbol)
        else:
            results.append(scored)

    results.sort(key=lambda r: (-r["score"], r["symbol"]))

    return {
        "results": results,
        "unavailable_symbols": unavailable_symbols,
        "max_score": CANSLIM_MAX_SCORE,
        "minimum_score": CANSLIM_DEFAULT_MINIMUM_SCORE,
    }