"""
Set 3 / Quantitative / Asset Allocation - Relative Valuation:
Fed Model Relative Valuation.

Source: "Predicting Stock Market Valuation and Direction Using the Fed Model".

    Earnings yield (%)  = 100 / P/E
    Fed Model spread    = earnings yield (%) - 10-year Treasury yield (%)
                          (percentage points)

Interpretation (relative comparison only):
    spread >  +band   equities appear relatively more attractive than bonds
    spread <  -band   bonds appear relatively more attractive
    otherwise         relative yields are approximately equal

This is NOT a proven trading signal; the module emits no buy / sell / hold.

DATA HONESTY
* The ETF's P/E is the ETF-level figure the existing provider (yfinance) reports -
  a clearly labelled PROXY for the portfolio's aggregate earnings yield. The repo
  holds no constituent fundamentals + weights for these ETFs, so a weighted
  aggregate cannot be built.
* The provider supplies a CURRENT P/E only. No historical earnings-yield series
  exists in this backend, so none is invented (no constant-EPS back-fill): the
  historical earnings-yield / spread series contain only real observations
  (today's snapshot) and say so. The historical 10-year Treasury yield series is
  real provider data.
* Bond / commodity ETFs (SHY, GLD, ...) are ``not_applicable``.
* The optional CPI-vs-spread Granger test needs a historical CPI series AND a
  historical spread series; neither exists here, so it is reported
  ``unavailable`` rather than approximated.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import pandas as pd

from app.strategies.set_3_strategies.base import (
    CATEGORY_QUANTITATIVE,
    StrategyDescriptor,
)
from app.strategies.set_3_strategies.common import downsample_indices
from app.strategies.set_3_strategies.schemas.quantitative import (
    FED_STRATEGY_ID,
    DataAvailability,
    DatePoint,
    FedModelCalculatedValues,
    FedModelChartData,
    FedModelChartPoint,
    FedModelHistoricalSeries,
    FedModelParams,
    FedModelResult,
    GrangerBlock,
    Interpretation,
    LabeledValue,
    TreasuryStats,
)
from app.strategies.set_3_strategies.services import data_service
from app.strategies.set_3_strategies.services.data_service import (
    EarningsYieldInput,
    Set3DataUnavailable,
    TreasuryYieldInput,
)

logger = logging.getLogger(__name__)

CALCULATION_VERSION = "fed-1.0.0"
DESCRIPTOR = StrategyDescriptor(
    strategy_id=FED_STRATEGY_ID,
    name="Fed Model Relative Valuation",
    category=CATEGORY_QUANTITATIVE,
    subcategory="Asset Allocation / Relative Valuation",
    description=(
        "Compares an equity ETF's earnings yield (1 / P/E) with the US 10-year "
        "Treasury yield. A positive Fed Model spread means equities look relatively "
        "cheaper than bonds on this comparison. A relative-valuation heuristic, not "
        "a proven trading signal."
    ),
    calculation_version=CALCULATION_VERSION,
)

MAX_HISTORY_POINTS = 5000
ALIGNMENT_WARNING_DAYS = 5
EARNINGS_HISTORY_NOTE = (
    "The data provider exposes only the current ETF-level P/E, so no historical "
    "earnings-yield series exists in this backend. Only real observations are "
    "listed; earnings yield and spread are not back-filled or estimated."
)
GRANGER_MESSAGE = (
    "Optional Granger-causality test (CPI inflation vs. Fed Model spread, with lags) "
    "is unavailable: it needs a historical CPI series and a historical Fed Model "
    "spread series, and this backend has neither (no CPI source; earnings-yield "
    "history is not stored). Note that correlation, Granger-causality testing and "
    "out-of-sample predictive power are three different things; none is claimed here."
)


@dataclass(frozen=True)
class EtfProfile:
    symbol: str
    name: str
    tags: tuple[str, ...]
    asset_class: str


_NON_EQUITY_TAGS = {"treasury", "gold", "bond", "commodity"}
_NON_EQUITY_WORDS = ("treasur", "bond", "fixed income", "commodit", "gold", "cash")


def check_equity_applicability(profile: EtfProfile) -> tuple[bool, str]:
    """(applicable, reason). Only equity ETFs have a meaningful P/E for this model."""
    tags = {t.lower() for t in profile.tags}
    ac = profile.asset_class.lower()
    is_equity = "equity" in tags or "equity" in ac
    non_equity = bool(tags & _NON_EQUITY_TAGS) or any(w in ac for w in _NON_EQUITY_WORDS)
    if is_equity and not non_equity:
        return True, ""
    if non_equity:
        return False, (
            f"{profile.symbol} ({profile.asset_class}) is not an equity ETF; the Fed Model "
            "compares EQUITY earnings yield with Treasury yields, so it does not apply."
        )
    return False, (
        f"{profile.symbol} could not be identified as an equity ETF "
        f"(asset class: {profile.asset_class}); the Fed Model is not applicable."
    )


def _envelope(
    symbol: str, params: FedModelParams, availability: DataAvailability, status: str, **extra
) -> FedModelResult:
    return FedModelResult(
        strategy_id=DESCRIPTOR.strategy_id,
        strategy_name=DESCRIPTOR.name,
        subcategory=DESCRIPTOR.subcategory,
        symbol=symbol,
        description=DESCRIPTOR.description,
        calculation_version=CALCULATION_VERSION,
        parameters=params,
        data_availability=availability,
        calculation_status=status,  # type: ignore[arg-type]
        **extra,
    )


def unavailable_result(
    symbol: str, params: FedModelParams, message: str, *, status: str = "unavailable",
    error: str | None = None,
) -> FedModelResult:
    return _envelope(
        symbol, params,
        DataAvailability(status="unavailable", message=message),
        status,
        error=error if error is not None else (message if status == "error" else None),
    )


def classify_spread(spread_pct: float, neutral_band_pct: float) -> str:
    if spread_pct > neutral_band_pct:
        return "equities_relatively_more_attractive"
    if spread_pct < -neutral_band_pct:
        return "bonds_relatively_more_attractive"
    return "approximately_equal"


_INTERPRETATION = {
    "equities_relatively_more_attractive": (
        "Positive spread: under the Fed Model comparison, equities appear relatively "
        "more attractive than bonds."
    ),
    "bonds_relatively_more_attractive": (
        "Negative spread: under the Fed Model comparison, bonds appear relatively "
        "more attractive than equities."
    ),
    "approximately_equal": (
        "Near-zero spread: equity earnings yield and the 10-year Treasury yield are "
        "approximately equal."
    ),
}


def _percentile_of(series: pd.Series, value: float) -> float:
    return float((series <= value).mean() * 100.0)


def build_fed_model_result(
    symbol: str,
    params: FedModelParams,
    earnings: EarningsYieldInput,
    treasury: TreasuryYieldInput,
) -> FedModelResult:
    """Pure computation once both real inputs are in hand."""
    ey_pct = 100.0 / earnings.pe_ratio
    series = treasury.series
    y_now = float(series.iloc[-1])
    spread = ey_pct - y_now
    alignment = (earnings.as_of - treasury.as_of).days
    relation = classify_spread(spread, params.neutral_band_pct)

    stats = TreasuryStats(
        lookback_years=params.lookback_years,
        observations=int(len(series)),
        start_date=series.index[0].date().isoformat(),
        end_date=series.index[-1].date().isoformat(),
        min_pct=round(float(series.min()), 4),
        max_pct=round(float(series.max()), 4),
        mean_pct=round(float(series.mean()), 4),
        current_percentile=round(_percentile_of(series, y_now), 2),
    )
    granger = (
        GrangerBlock(status="unavailable", message=GRANGER_MESSAGE)
        if params.include_granger
        else None
    )
    calculated = FedModelCalculatedValues(
        pe_ratio=round(earnings.pe_ratio, 4),
        pe_basis=earnings.pe_basis,
        earnings_yield_pct=round(ey_pct, 4),
        earnings_yield_source=earnings.source,
        earnings_yield_as_of=earnings.as_of.isoformat(),
        earnings_yield_is_proxy=earnings.is_proxy,
        treasury_yield_pct=round(y_now, 4),
        treasury_yield_as_of=treasury.as_of.isoformat(),
        treasury_yield_source=treasury.source,
        fed_model_spread_pct=round(spread, 4),
        date_alignment_days=alignment,
        neutral_band_pct=params.neutral_band_pct,
        relative_valuation=relation,  # type: ignore[arg-type]
        treasury_stats=stats,
        granger_causality=granger,
    )

    # --- historical series: REAL observations only
    t_idx = downsample_indices(len(series), MAX_HISTORY_POINTS)
    treasury_points = [
        DatePoint(date=series.index[i].date().isoformat(), value=round(float(series.iloc[i]), 4))
        for i in t_idx
    ]
    ey_point = DatePoint(date=earnings.as_of.isoformat(), value=round(ey_pct, 4))
    spread_point = DatePoint(date=earnings.as_of.isoformat(), value=round(spread, 4))
    historical = FedModelHistoricalSeries(
        earnings_yield=[ey_point],
        treasury_yield=treasury_points,
        fed_model_spread=[spread_point],
        earnings_yield_history_available=False,
        earnings_yield_history_note=EARNINGS_HISTORY_NOTE,
    )

    c_idx = downsample_indices(len(series), params.chart_max_points)
    rows: dict[str, FedModelChartPoint] = {
        series.index[i].date().isoformat(): FedModelChartPoint(
            date=series.index[i].date().isoformat(),
            treasury_yield_pct=round(float(series.iloc[i]), 4),
        )
        for i in c_idx
    }
    key = earnings.as_of.isoformat()
    row = rows.get(key) or FedModelChartPoint(date=key)
    row.earnings_yield_pct = round(ey_pct, 4)
    row.fed_model_spread_pct = round(spread, 4)
    rows[key] = row
    chart = FedModelChartData(
        series=[rows[k] for k in sorted(rows)],
        latest_comparison=[
            LabeledValue(label="Earnings yield (%)", value=round(ey_pct, 4)),
            LabeledValue(label="10-year Treasury yield (%)", value=round(y_now, 4)),
            LabeledValue(label="Fed Model spread (pp)", value=round(spread, 4)),
        ],
    )

    warnings = [
        "Earnings yield is derived from the ETF-level P/E reported by the data provider "
        "(a proxy for the portfolio's aggregate earnings yield)."
    ]
    if treasury.dropped_observations:
        warnings.append(
            f"{treasury.dropped_observations} Treasury observation(s) outside 0-20% were dropped."
        )
    if abs(alignment) > ALIGNMENT_WARNING_DAYS:
        warnings.append(
            f"The earnings-yield snapshot ({earnings.as_of}) and the newest Treasury "
            f"observation ({treasury.as_of}) are {abs(alignment)} days apart."
        )

    interpretation = Interpretation(
        summary=_INTERPRETATION[relation],
        points=[
            f"Earnings yield {ey_pct:.2f}% (P/E {earnings.pe_ratio:.2f}, {earnings.pe_basis}) "
            f"vs 10-year Treasury {y_now:.2f}% -> spread {spread:+.2f} percentage points.",
            f"The current Treasury yield sits at the {stats.current_percentile:.0f}th "
            f"percentile of the last {params.lookback_years} year(s).",
            "The Fed Model is a relative-valuation heuristic; it is not a proven market-timing "
            "signal and says nothing about absolute valuation.",
        ],
        classification=relation,
    )
    availability = DataAvailability(
        status="partial",  # current spread yes, historical earnings yield no
        observations=int(len(series)),
        start_date=stats.start_date,
        end_date=stats.end_date,
        message=EARNINGS_HISTORY_NOTE,
        components={
            "earnings_yield_current": "available",
            "earnings_yield_history": "unavailable: provider supplies the current P/E only",
            "treasury_yield_history": "available",
            "fed_model_spread_current": "available",
            "granger_causality": "unavailable" if params.include_granger else "not_requested",
        },
    )
    return _envelope(
        symbol, params, availability, "ok",
        calculated_values=calculated, historical_series=historical, chart_data=chart,
        interpretation=interpretation, warnings=warnings,
        notes=[
            "Earnings yield = 100 / P/E; Fed Model spread = earnings yield - 10-year Treasury "
            "yield, both in percent, spread in percentage points.",
            "Not a proven trading signal; no buy / sell / hold output is produced.",
        ],
    )


async def run_fed_model_relative_valuation(
    symbol: str,
    profile: EtfProfile,
    params: FedModelParams,
    *,
    fetch_earnings=None,
    fetch_treasury=None,
) -> FedModelResult:
    """Applicability check -> fetch both inputs concurrently -> compute."""
    applicable, reason = check_equity_applicability(profile)
    if not applicable:
        return _envelope(
            symbol, params,
            DataAvailability(status="not_applicable", message=reason),
            "not_applicable",
        )

    fetch_earnings = fetch_earnings or data_service.fetch_etf_earnings_yield
    fetch_treasury = fetch_treasury or data_service.fetch_treasury_yield_history
    ey_res, ty_res = await asyncio.gather(
        fetch_earnings(symbol, params.pe_basis),
        fetch_treasury(params.lookback_years),
        return_exceptions=True,
    )

    def message(res: BaseException, what: str) -> str:
        if isinstance(res, Set3DataUnavailable):
            return str(res)
        logger.error("Set 3 Fed Model %s fetch failed", what, exc_info=res)
        return f"The {what} could not be retrieved because of a data-provider error."

    components: dict[str, str] = {}
    ey_msg = ty_msg = None
    if isinstance(ey_res, BaseException):
        ey_msg = message(ey_res, "earnings yield")
        components["earnings_yield_current"] = f"unavailable: {ey_msg}"
    if isinstance(ty_res, BaseException):
        ty_msg = message(ty_res, "10-year Treasury yield")
        components["treasury_yield_history"] = f"unavailable: {ty_msg}"

    if ey_msg or ty_msg:
        msg = " ".join(m for m in (ey_msg, ty_msg) if m)
        hist = None
        if ty_msg is None and not isinstance(ty_res, BaseException):
            s = ty_res.series
            idx = downsample_indices(len(s), MAX_HISTORY_POINTS)
            hist = FedModelHistoricalSeries(
                treasury_yield=[
                    DatePoint(date=s.index[i].date().isoformat(), value=round(float(s.iloc[i]), 4))
                    for i in idx
                ],
                earnings_yield_history_note=EARNINGS_HISTORY_NOTE,
            )
            components["treasury_yield_history"] = "available"
        return _envelope(
            symbol, params,
            DataAvailability(
                status="partial" if hist else "unavailable", message=msg, components=components
            ),
            "partial" if hist else "unavailable",
            historical_series=hist,
            warnings=[msg],
        )
    return build_fed_model_result(symbol, params, ey_res, ty_res)
