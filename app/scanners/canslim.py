"""
CANSLIM STOCK SCANNER — scoring engine.

THIS IS NOT THE ETF CANSLIM MODULE. See app/services/canslim.py for the
pre-existing, unrelated ETF-universe proxy score (kept for backward
compatibility with GET /api/strategies/canslim). This module implements
William O'Neil's CANSLIM methodology as described in the supplied source
article, for the stock-scanner architecture in app/scanners/.

Every function here is pure (no DB, no HTTP) — same layering convention
as app.services.strategies / app.services.canslim: I/O and cross-symbol
orchestration live in app/scanners/data.py, app/scanners/ingestion.py,
and app/routers/scanners.py.

======================================================================
SOURCE-OF-TRUTH MAPPING (article rule -> implementation), read this
before changing any threshold below
======================================================================

C — Current Quarterly Earnings
    Article: "minimum 25% increase in earnings per share (EPS) in the
    most recent quarter, compared to the same quarter the prior year."
    Implementation: exact — quarter-over-year-ago-quarter EPS growth,
    threshold 25%. Negative/zero year-ago EPS makes a percentage
    meaningless (a company going from -$0.10 to -$0.05 EPS is "growth"
    by a naive formula but is not what the article means), so those
    cases are returned as NOT_APPLICABLE rather than a misleading
    percentage or a silent pass/fail.

A — Annual Earnings Growth
    Article: "annual EPS growth of at least 25% over the past three to
    five years, with strong return on equity (ideally above 17%)."
    Implementation: exact thresholds (25% / 17% ROE). The article's
    phrasing doesn't disambiguate "25% over 3-5 years" as an annualized
    rate vs. total growth across the window — this module computes
    TOTAL growth from the earliest to the latest available annual EPS
    in a 3-5 year window (documented judgment call, not a fabricated
    number; see CANSLIM_A_GROWTH_BASIS below). Requires at least
    CANSLIM_A_MIN_YEARS of annual EPS history; uses up to
    CANSLIM_A_MAX_YEARS when available. Never invents a ROE or an EPS
    value that wasn't supplied.

N — New Products, Services, Management, or Highs
    Article (prose): "buying stocks near 52-week highs... stocks
    breaking out to new highs." No exact percentage is given in the
    article's prose. The article's OWN reference Python script defines
    "near a 52-week high" as current_price / 52w_high >= 0.85 — this
    module reuses THAT number because it is the article's own stated
    interpretation, not because the old ThrustWise ETF module happened
    to reuse the same figure (see app/services/canslim.py — this is a
    coincidence of both being downstream of the same source article,
    not this module copying that module). The "new products / new
    management" component is explicitly qualitative and NOT computed —
    see `qualitative_component_note` below; it is never fabricated.

S — Supply and Demand
    Article: "float... under 25 million shares, ideally." Implementation
    uses exactly that threshold (CANSLIM_S_MAX_FLOAT_SHARES =
    25,000,000), NOT the old ETF module's 500,000,000. The article also
    mentions insider buying and buybacks as supply-reducing signals;
    ThrustWise's EODHD integration does not currently ingest a
    time-series of insider transactions or historical shares-outstanding
    needed to detect "actively repurchasing" — this is documented as a
    known limitation (NOT_APPLICABLE informational field), never
    fabricated as a pass/fail.

L — Leader or Laggard
    Article: "RS Rating of 80 or above — meaning the stock was
    outperforming at least 80% of the market." Implementation: exact
    threshold (80th percentile), but computed as ThrustWise's OWN
    cross-sectional relative-strength percentile (trailing 12-month
    total return, ranked against the rest of the scanned universe) —
    explicitly labeled "Calculated Relative Strength Percentile", never
    called an official IBD RS Rating, since IBD's proprietary RS Rating
    data is not a ThrustWise data source. See CANSLIM_L_LOOKBACK_DAYS.

I — Institutional Sponsorship
    Article (prose only): "at least a few institutional investors...
    should own the stock, and ideally the number of institutional
    owners should be increasing... warned against stocks that are
    over-owned." No exact counts are given anywhere in the article.
    Implementation uses a documented, disclosed interpretation of "a
    few" (CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS, see constant below) as
    the pass/fail gate, computed from EODHD's actual reported
    institutional-holder count — never a fabricated ownership
    percentage. "Increasing over time" and "over-owned" have no
    article-specified numeric threshold and are NOT used to gate
    pass/fail; ownership percentage is still surfaced as an
    informational value.

M — Market Direction
    Article: "if the broad market is in a confirmed downtrend... stay
    on the sidelines" plus the article's own addendum: "replacing the
    200-day SMA proxy with an analysis of the S&P 500 index itself —
    if SPY is below its 200-day SMA, treat the entire market as a
    downtrend." Implementation: exactly that — the S&P 500 INDEX price
    vs. the S&P 500 INDEX's own 200-day SMA, computed ONCE per scan and
    applied identically to every stock in the universe (never a
    per-stock "price vs. its own SMA200", which is what the old ETF
    module does).
======================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.strategies import _round

# --- Status vocabulary (never silently convert UNAVAILABLE to FAIL) -------

STATUS_PASS = "PASS"
STATUS_FAIL = "FAIL"
STATUS_UNAVAILABLE = "UNAVAILABLE"  # required data was not available
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"  # data present but comparison is undefined/misleading

CRITERIA_KEYS = ("c", "a", "n", "s", "l", "i", "m")

CRITERIA_NAMES = {
    "c": "Current Quarterly Earnings",
    "a": "Annual Earnings Growth",
    "n": "New (Price Highs)",
    "s": "Supply and Demand",
    "l": "Leader (Relative Strength)",
    "i": "Institutional Sponsorship",
    "m": "Market Direction",
}

# --- C: Current Quarterly Earnings -----------------------------------------
CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH = 0.25  # article: "minimum 25% increase"

# --- A: Annual Earnings Growth ----------------------------------------------
CANSLIM_A_MIN_ANNUAL_EPS_GROWTH = 0.25  # article: "at least 25%"
CANSLIM_A_MIN_ROE = 0.17  # article: "ideally above 17%"
CANSLIM_A_MIN_YEARS = 3  # article: "three to five years" (lower bound)
CANSLIM_A_MAX_YEARS = 5  # article: "three to five years" (upper bound)
# Documented judgment call (article does not disambiguate annualized vs.
# total growth over the window) — see module docstring, section A.
CANSLIM_A_GROWTH_BASIS = "total_growth_earliest_to_latest_in_window"

# --- N: New (measurable price component only) -------------------------------
# The article's OWN reference script's threshold for "near a 52-week
# high" — see module docstring, section N, for why this is preserved.
CANSLIM_N_NEAR_HIGH_RATIO = 0.85
CANSLIM_N_QUALITATIVE_NOTE = (
    "CANSLIM's 'New' criterion also covers new products, services, or "
    "management — these are qualitative company events that ThrustWise "
    "cannot verify from market/fundamental data and are NOT evaluated "
    "here. Only the measurable price-highs component (proximity to the "
    "52-week high) is scored."
)

# --- S: Supply and Demand -----------------------------------------------------
CANSLIM_S_MAX_FLOAT_SHARES = 25_000_000  # article: "under 25 million shares, ideally"
CANSLIM_S_QUALITATIVE_NOTE = (
    "The article also treats insider buying and active share buybacks as "
    "supply-reducing signals. ThrustWise's EODHD integration does not "
    "ingest a time series of insider transactions or historical shares "
    "outstanding, so these two sub-signals cannot be reliably calculated "
    "and are not part of the pass/fail rule (float shares vs. threshold "
    "only)."
)

# --- L: Leader (relative strength) -------------------------------------------
CANSLIM_L_MIN_PERCENTILE = 80.0  # article: "RS Rating of 80 or above"
# Article gives the threshold but not the lookback window; 252 trading
# days (~12 months) is used here as IBD's own RS Rating is conventionally
# a trailing-~12-month measure — documented choice, not a fabricated
# number pretending to be the article's.
CANSLIM_L_LOOKBACK_TRADING_DAYS = 252
CANSLIM_L_LABEL = "Calculated Relative Strength Percentile"

# --- I: Institutional Sponsorship --------------------------------------------
# The article never gives a number for "a few" institutional investors.
# This is a disclosed interpretive default, not an official CANSLIM
# figure — see module docstring, section I.
CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS = 3

# --- M: Market Direction (evaluated once for the whole universe) ------------
CANSLIM_M_SMA_PERIOD = 200
CANSLIM_M_INDEX_SYMBOL = "GSPC"  # S&P 500 index, per the article's own addendum

CANSLIM_MAX_CRITERIA = 7
CANSLIM_DEFAULT_MIN_CRITERIA_PASSED = 5


def _criterion(
    key: str,
    *,
    status: str,
    value: float | int | None = None,
    threshold: float | int | None = None,
    unit: str | None = None,
    period: str | None = None,
    explanation: str,
    data_source: str,
    as_of: str | None = None,
) -> dict:
    return {
        "criterion": key.upper(),
        "name": CRITERIA_NAMES[key],
        "status": status,
        "value": value,
        "threshold": threshold,
        "unit": unit,
        "period": period,
        "explanation": explanation,
        "data_source": data_source,
        "as_of": as_of,
    }


# ---------------------------------------------------------------------------
# C — Current Quarterly Earnings
# ---------------------------------------------------------------------------


def compute_criterion_c(
    eps_current: float | None,
    eps_current_period: str | None,
    eps_year_ago: float | None,
    eps_year_ago_period: str | None,
) -> dict:
    if eps_current is None or eps_year_ago is None:
        return _criterion(
            "c",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH,
            unit="percent",
            explanation=(
                "Quarterly EPS for the current quarter and/or the "
                "same quarter one year ago is not available from EODHD "
                "for this symbol."
            ),
            data_source="EODHD Fundamentals: Earnings.History",
        )

    if eps_year_ago <= 0:
        return _criterion(
            "c",
            status=STATUS_NOT_APPLICABLE,
            value=None,
            threshold=CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH,
            unit="percent",
            period=eps_current_period,
            explanation=(
                f"Same-quarter-prior-year EPS was {eps_year_ago:.2f} "
                "(zero or negative), so a percentage growth comparison "
                "against it would be undefined or misleading. Not "
                "scored as pass or fail."
            ),
            data_source="EODHD Fundamentals: Earnings.History",
        )

    growth = (eps_current - eps_year_ago) / eps_year_ago
    passed = growth >= CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH

    return _criterion(
        "c",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=_round(growth, 4),
        threshold=CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH,
        unit="percent",
        period=eps_current_period,
        explanation=(
            f"Quarterly EPS grew {growth:.1%} year-over-year "
            f"({eps_current_period or 'latest quarter'}: {eps_current:.2f} vs. "
            f"{eps_year_ago_period or 'year-ago quarter'}: {eps_year_ago:.2f}), "
            f"{'meeting' if passed else 'below'} the {CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH:.0%} minimum."
        ),
        data_source="EODHD Fundamentals: Earnings.History",
    )


# ---------------------------------------------------------------------------
# A — Annual Earnings Growth
# ---------------------------------------------------------------------------


def compute_criterion_a(
    annual_eps_history: list[dict],  # [{"year": int, "eps": float}], ascending by year
    return_on_equity: float | None,
) -> dict:
    years_available = len(annual_eps_history)

    if years_available < CANSLIM_A_MIN_YEARS or return_on_equity is None:
        missing = []
        if years_available < CANSLIM_A_MIN_YEARS:
            missing.append(
                f"only {years_available} year(s) of annual EPS history "
                f"(need at least {CANSLIM_A_MIN_YEARS})"
            )
        if return_on_equity is None:
            missing.append("return on equity (TTM)")
        return _criterion(
            "a",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_A_MIN_ANNUAL_EPS_GROWTH,
            unit="percent",
            explanation="Unavailable: " + "; ".join(missing) + ".",
            data_source="EODHD Fundamentals: Earnings.Annual, Highlights.ReturnOnEquityTTM",
        )

    window = annual_eps_history[-CANSLIM_A_MAX_YEARS:]
    earliest, latest = window[0], window[-1]

    if earliest["eps"] <= 0:
        return _criterion(
            "a",
            status=STATUS_NOT_APPLICABLE,
            threshold=CANSLIM_A_MIN_ANNUAL_EPS_GROWTH,
            unit="percent",
            period=f"{earliest['year']}-{latest['year']}",
            explanation=(
                f"Earliest year ({earliest['year']}) EPS was "
                f"{earliest['eps']:.2f} (zero or negative), so a "
                "percentage growth comparison over the window would be "
                "undefined or misleading. Not scored as pass or fail."
            ),
            data_source="EODHD Fundamentals: Earnings.Annual, Highlights.ReturnOnEquityTTM",
        )

    growth = (latest["eps"] - earliest["eps"]) / earliest["eps"]
    roe_pass = return_on_equity >= CANSLIM_A_MIN_ROE
    growth_pass = growth >= CANSLIM_A_MIN_ANNUAL_EPS_GROWTH
    passed = growth_pass and roe_pass

    return _criterion(
        "a",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=_round(growth, 4),
        threshold=CANSLIM_A_MIN_ANNUAL_EPS_GROWTH,
        unit="percent",
        period=f"{earliest['year']}-{latest['year']}",
        explanation=(
            f"EPS grew {growth:.1%} from {earliest['year']} "
            f"({earliest['eps']:.2f}) to {latest['year']} ({latest['eps']:.2f}); "
            f"ROE (TTM) is {return_on_equity:.1%}. "
            f"{'Meets' if growth_pass else 'Below'} the {CANSLIM_A_MIN_ANNUAL_EPS_GROWTH:.0%} "
            f"EPS-growth threshold and {'meets' if roe_pass else 'below'} the "
            f"{CANSLIM_A_MIN_ROE:.0%} ROE threshold — both are required to pass."
        ),
        data_source="EODHD Fundamentals: Earnings.Annual, Highlights.ReturnOnEquityTTM",
    )


# ---------------------------------------------------------------------------
# N — New (measurable price-highs component)
# ---------------------------------------------------------------------------


def compute_criterion_n(
    current_price: float | None, week_52_high: float | None
) -> dict:
    if current_price is None or week_52_high is None or week_52_high <= 0:
        return _criterion(
            "n",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_N_NEAR_HIGH_RATIO,
            unit="ratio",
            explanation="Insufficient price history to determine the 52-week high. "
            + CANSLIM_N_QUALITATIVE_NOTE,
            data_source="Ingested EODHD daily price history",
        )

    ratio = current_price / week_52_high
    passed = ratio >= CANSLIM_N_NEAR_HIGH_RATIO
    is_new_high = current_price >= week_52_high

    return _criterion(
        "n",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=_round(ratio, 4),
        threshold=CANSLIM_N_NEAR_HIGH_RATIO,
        unit="ratio",
        explanation=(
            f"Price is {ratio:.1%} of its 52-week high"
            f"{' and is making a new 52-week high today' if is_new_high else ''}. "
            f"{'Within' if passed else 'Below'} the "
            f"{CANSLIM_N_NEAR_HIGH_RATIO:.0%} 'near a 52-week high' band. "
            + CANSLIM_N_QUALITATIVE_NOTE
        ),
        data_source="Ingested EODHD daily price history",
    )


# ---------------------------------------------------------------------------
# S — Supply and Demand
# ---------------------------------------------------------------------------


def compute_criterion_s(shares_float: float | None) -> dict:
    if shares_float is None:
        return _criterion(
            "s",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_S_MAX_FLOAT_SHARES,
            unit="shares",
            explanation="Shares float is not available from EODHD for this symbol. "
            + CANSLIM_S_QUALITATIVE_NOTE,
            data_source="EODHD Fundamentals: SharesStats.SharesFloat",
        )

    passed = shares_float < CANSLIM_S_MAX_FLOAT_SHARES
    return _criterion(
        "s",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=shares_float,
        threshold=CANSLIM_S_MAX_FLOAT_SHARES,
        unit="shares",
        explanation=(
            f"Float is {shares_float:,.0f} shares, "
            f"{'under' if passed else 'at or above'} the "
            f"{CANSLIM_S_MAX_FLOAT_SHARES:,} share threshold. "
            + CANSLIM_S_QUALITATIVE_NOTE
        ),
        data_source="EODHD Fundamentals: SharesStats.SharesFloat",
    )


# ---------------------------------------------------------------------------
# L — Leader (cross-sectional relative strength)
# ---------------------------------------------------------------------------


def rank_relative_strength(returns_by_symbol: dict[str, float]) -> dict[str, dict]:
    """Cross-sectional percentile rank of trailing total return.

    `returns_by_symbol` should already be restricted to symbols that had
    enough price history for a valid trailing return (see
    app/scanners/data.py) — this function does not itself decide
    availability. Rank 1 = best performer. Percentile = the fraction of
    the rest of the universe this symbol outperformed, so the best
    performer gets the highest percentile.

    Returns {symbol: {"rank": int, "universe_size": int, "percentile": float}}.
    """
    universe_size = len(returns_by_symbol)
    if universe_size == 0:
        return {}

    ranked = sorted(returns_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    result: dict[str, dict] = {}
    for rank, (symbol, _ret) in enumerate(ranked, start=1):
        percentile = (
            _round((universe_size - rank) / (universe_size - 1) * 100, 1)
            if universe_size > 1
            else 100.0
        )
        result[symbol] = {
            "rank": rank,
            "universe_size": universe_size,
            "percentile": percentile,
        }
    return result


def compute_criterion_l(
    relative_strength: dict | None, lookback_days: int = CANSLIM_L_LOOKBACK_TRADING_DAYS
) -> dict:
    if relative_strength is None:
        return _criterion(
            "l",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_L_MIN_PERCENTILE,
            unit="percentile",
            period=f"trailing {lookback_days} trading days",
            explanation=(
                "Insufficient price history for this symbol, or the "
                "symbol was excluded from the universe's relative-"
                "strength ranking."
            ),
            data_source=f"Ingested EODHD daily price history — {CANSLIM_L_LABEL}",
        )

    percentile = relative_strength["percentile"]
    passed = percentile >= CANSLIM_L_MIN_PERCENTILE

    return _criterion(
        "l",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=percentile,
        threshold=CANSLIM_L_MIN_PERCENTILE,
        unit="percentile",
        period=f"trailing {lookback_days} trading days",
        explanation=(
            f"{CANSLIM_L_LABEL}: rank {relative_strength['rank']} of "
            f"{relative_strength['universe_size']} in the scanned universe "
            f"(percentile {percentile:.1f}). "
            f"{'Meets' if passed else 'Below'} the {CANSLIM_L_MIN_PERCENTILE:.0f}th-"
            "percentile threshold. This is ThrustWise's own calculated "
            "percentile, NOT an official IBD RS Rating."
        ),
        data_source=f"Ingested EODHD daily price history — {CANSLIM_L_LABEL}",
    )


# ---------------------------------------------------------------------------
# I — Institutional Sponsorship
# ---------------------------------------------------------------------------


def compute_criterion_i(
    institutional_holders_count: int | None, percent_institutions: float | None
) -> dict:
    if institutional_holders_count is None:
        return _criterion(
            "i",
            status=STATUS_UNAVAILABLE,
            threshold=CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS,
            unit="holders",
            explanation=(
                "Institutional holder count is not available from EODHD "
                "for this symbol."
            ),
            data_source="EODHD Fundamentals: Holders.Institutions, SharesStats.PercentInstitutions",
        )

    passed = institutional_holders_count >= CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS
    ownership_note = (
        f" Reported institutional ownership is {percent_institutions:.1%}."
        if percent_institutions is not None
        else " Institutional ownership percentage is unavailable."
    )
    return _criterion(
        "i",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=institutional_holders_count,
        threshold=CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS,
        unit="holders",
        explanation=(
            f"{institutional_holders_count} reported institutional holder(s), "
            f"{'meeting' if passed else 'below'} the disclosed minimum of "
            f"{CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS} used to interpret the "
            "article's qualitative 'a few institutional investors' "
            "requirement (the article gives no exact number)."
            + ownership_note
            + " 'Increasing sponsorship over time' and 'over-owned' are not "
            "gated (no article-given threshold) and are informational only."
        ),
        data_source="EODHD Fundamentals: Holders.Institutions, SharesStats.PercentInstitutions",
    )


# ---------------------------------------------------------------------------
# M — Market Direction (computed once per scan, shared across all stocks)
# ---------------------------------------------------------------------------


def compute_market_direction(
    index_price: float | None, index_sma200: float | None
) -> dict:
    if index_price is None or index_sma200 is None:
        return _criterion(
            "m",
            status=STATUS_UNAVAILABLE,
            unit="index_points",
            explanation=(
                f"Insufficient {CANSLIM_M_INDEX_SYMBOL} index price history "
                f"to compute a {CANSLIM_M_SMA_PERIOD}-day SMA."
            ),
            data_source=f"Ingested EODHD index price history ({CANSLIM_M_INDEX_SYMBOL}.INDX)",
        )

    passed = index_price > index_sma200
    return _criterion(
        "m",
        status=STATUS_PASS if passed else STATUS_FAIL,
        value=_round(index_price),
        threshold=_round(index_sma200),
        unit="index_points",
        explanation=(
            f"S&P 500 index is {'above' if passed else 'at or below'} its "
            f"{CANSLIM_M_SMA_PERIOD}-day SMA ({index_price:.2f} vs. "
            f"{index_sma200:.2f}) — market condition is "
            f"{'a confirmed uptrend' if passed else 'a confirmed downtrend'} "
            "under this rule. This one market-direction result is applied "
            "identically to every stock in the scan (it is not computed "
            "per-stock)."
        ),
        data_source=f"Ingested EODHD index price history ({CANSLIM_M_INDEX_SYMBOL}.INDX)",
    )


# ---------------------------------------------------------------------------
# Assembling one stock's full, explainable result
# ---------------------------------------------------------------------------


def assemble_stock_result(
    *,
    symbol: str,
    company_name: str | None,
    sector: str | None,
    industry: str | None,
    universe: str,
    criteria: dict[str, dict],  # {"c": {...}, "a": {...}, ...} — one per CRITERIA_KEYS
    price_date: str | None,
    fundamentals_as_of: str | None,
    fundamentals_fetched_at: str | None,
    data_updated_at: str | None,
    min_criteria_passed: int = CANSLIM_DEFAULT_MIN_CRITERIA_PASSED,
) -> dict:
    passed = sum(1 for c in criteria.values() if c["status"] == STATUS_PASS)
    failed = sum(1 for c in criteria.values() if c["status"] == STATUS_FAIL)
    unavailable = sum(
        1
        for c in criteria.values()
        if c["status"] in (STATUS_UNAVAILABLE, STATUS_NOT_APPLICABLE)
    )
    evaluated = passed + failed

    if evaluated == 0:
        data_quality = "INSUFFICIENT"
    elif unavailable == 0:
        data_quality = "COMPLETE"
    else:
        data_quality = "PARTIAL"

    return {
        "symbol": symbol,
        "company_name": company_name,
        "sector": sector,
        "industry": industry,
        "universe": universe.upper(),
        "criteria": criteria,
        "criteria_passed": passed,
        "criteria_evaluated": evaluated,
        "criteria_failed": failed,
        "criteria_unavailable": unavailable,
        "criteria_total": CANSLIM_MAX_CRITERIA,
        "qualifies": passed >= min_criteria_passed,
        "min_criteria_passed": min_criteria_passed,
        "data_quality": data_quality,
        "price_date": price_date,
        "fundamental_as_of": fundamentals_as_of,
        "fundamental_fetched_at": fundamentals_fetched_at,
        "data_updated_at": data_updated_at,
    }
