"""
Portfolio Analytics engine — VT vs VTI + VXUS Portfolio Comparison.

Conceptually and physically separate from services/strategies.py: that
module computes technical indicators (EMA/RSI/MACD/etc.) for a single
symbol. This module instead grows a hypothetical $-investment across TWO
static, buy-and-hold portfolios built from THREE underlying ETFs (VT, VTI,
VXUS) and reports the value of each portfolio on every comparable trading
date. It is not a technical-analysis strategy, has no indicators, no
crossover/trend flags, and never rebalances after the initial purchase.

Portfolio A: 100% VT
Portfolio B: 60% VTI + 40% VXUS (configurable via the keyword defaults
below; not yet exposed as an API parameter — see module docstring in
routers/strategy.py).

Same shape as the rest of the services layer: pure functions over
`list[DailyPrice]` already loaded from the database (see
routers.deps.load_prices) — no DB or HTTP access here, no external calls.
"""

import datetime
import math

import pandas as pd

from app.models import DailyPrice
from app.services.analytics import to_series

# ---------------------------------------------------------------------------
# Configurable defaults (kept as named constants — not hardcoded inline —
# so a future allocation slider only needs to thread these through as
# optional parameters rather than restructure this module).
# ---------------------------------------------------------------------------

DEFAULT_INITIAL_INVESTMENT = 10_000.0
DEFAULT_VT_ALLOCATION = 1.0  # Portfolio A is 100% VT
DEFAULT_VTI_WEIGHT = 0.60  # Portfolio B: 60% VTI
DEFAULT_VXUS_WEIGHT = 0.40  # Portfolio B: 40% VXUS

# Client's example start date. Not assumed to be a valid trading day —
# see the date-alignment note on compute_vt_vs_vti_vxus below.
DEFAULT_START_DATE = datetime.date(2010, 1, 1)

# Minimum number of comparable trading days (across all three symbols, on
# or after start_date) required before the comparison is considered
# meaningful. Mirrors the MIN_ROWS_* pattern in services/strategies.py.
MIN_ROWS_PORTFOLIO_COMPARISON = 2

STRATEGY_KEY = "vt_vs_vti_vxus"


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def compute_vt_vs_vti_vxus(
    vt_prices: list[DailyPrice],
    vti_prices: list[DailyPrice],
    vxus_prices: list[DailyPrice],
    *,
    initial_investment: float = DEFAULT_INITIAL_INVESTMENT,
    vti_weight: float = DEFAULT_VTI_WEIGHT,
    vxus_weight: float = DEFAULT_VXUS_WEIGHT,
    start_date: datetime.date = DEFAULT_START_DATE,
) -> dict | None:
    """Grow a hypothetical `initial_investment` in two static portfolios:

    Portfolio A — 100% VT.
    Portfolio B — `vti_weight` in VTI + `vxus_weight` in VXUS, purchased
    once at the start date and then left to drift (no periodic
    rebalancing back to the target split — first-version requirement).

    Date alignment: VT, VTI, and VXUS do not necessarily share an
    identical trading calendar (different listing/holiday history), so
    the three price series are inner-joined on date first — only dates
    where ALL THREE symbols have a real, ingested price are kept. No
    missing value is ever interpolated or fabricated. The comparison
    then starts at the first joined date on/after `start_date` (so an
    input like Jan 1 2010, a market holiday, safely resolves to the next
    valid trading day) and ends at the latest joined date available.

    Returns None when any symbol has no price history at all, or when
    fewer than MIN_ROWS_PORTFOLIO_COMPARISON comparable trading days
    exist on/after start_date — callers should treat that as
    "unavailable", the same convention services/strategies.py uses for
    insufficient history.
    """
    vt_series = to_series(vt_prices)
    vti_series = to_series(vti_prices)
    vxus_series = to_series(vxus_prices)

    if vt_series.empty or vti_series.empty or vxus_series.empty:
        return None

    # Inner join: only dates present for VT, VTI, AND VXUS survive. This is
    # the safe alignment behavior called for — never insert a fake price
    # for a symbol missing on a given date.
    combined = pd.concat(
        {"vt": vt_series, "vti": vti_series, "vxus": vxus_series},
        axis=1,
        join="inner",
    ).dropna()

    start_ts = pd.Timestamp(start_date)
    combined = combined[combined.index >= start_ts]

    if len(combined) < MIN_ROWS_PORTFOLIO_COMPARISON:
        return None

    vt_start_price = float(combined["vt"].iloc[0])
    vti_start_price = float(combined["vti"].iloc[0])
    vxus_start_price = float(combined["vxus"].iloc[0])

    if vt_start_price <= 0 or vti_start_price <= 0 or vxus_start_price <= 0:
        return None

    vti_investment = initial_investment * vti_weight
    vxus_investment = initial_investment * vxus_weight

    # Normalized growth: value(t) = investment * (price(t) / price(start)).
    # No rebalancing — VTI and VXUS shares are fixed at the initial split
    # and simply grow independently from here.
    vt_value = initial_investment * (combined["vt"] / vt_start_price)
    vti_value = vti_investment * (combined["vti"] / vti_start_price)
    vxus_value = vxus_investment * (combined["vxus"] / vxus_start_price)
    combined_value = vti_value + vxus_value

    performance = [
        {
            "date": ts.date().isoformat(),
            "vt_value": _round(float(vt_v)),
            "vti_vxus_value": _round(float(combo_v)),
        }
        for ts, vt_v, combo_v in zip(
            combined.index, vt_value.tolist(), combined_value.tolist()
        )
    ]

    return {
        "strategy": STRATEGY_KEY,
        "start_date": combined.index[0].date().isoformat(),
        "end_date": combined.index[-1].date().isoformat(),
        "initial_investment": _round(initial_investment),
        "vt_allocation": DEFAULT_VT_ALLOCATION,
        "vti_allocation": vti_weight,
        "vxus_allocation": vxus_weight,
        "performance": performance,
    }