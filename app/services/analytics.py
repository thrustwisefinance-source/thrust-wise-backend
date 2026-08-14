"""
Rule-based metrics engine.

Everything here is computed from stored daily adjusted-close prices —
no external calls. Functions take DailyPrice rows sorted ascending by date.
"""

import math

import numpy as np
import pandas as pd

from app.config import settings
from app.constants import ETF_EDUCATION, ETF_HOLDINGS, EtfStaticMeta, MarketIndexMeta
from app.models import DailyPrice

TRADING_DAYS_PER_YEAR = 252
SPARKLINE_POINTS = 30
MAX_CHART_POINTS = 30

# Chart ranges: calendar months back from the latest data point,
# plus the date-label format for that range.
RANGE_CONFIG: dict[str, tuple[int, str]] = {
    "1M": (1, "%b %d"),
    "3M": (3, "%b %d"),
    "6M": (6, "%b %d"),
    "1Y": (12, "%b %Y"),
    "5Y": (60, "%b %Y"),
}


def to_series(prices: list[DailyPrice]) -> pd.Series:
    """Adjusted-close series indexed by date, ascending."""
    if not prices:
        return pd.Series(dtype=float)
    return pd.Series(
        [p.adjusted_close for p in prices],
        index=pd.DatetimeIndex([p.date for p in prices]),
        dtype=float,
    ).sort_index()


def _round(value: float, digits: int = 2) -> float:
    if value is None or math.isnan(value) or math.isinf(value):
        return 0.0
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# Snapshot / quote
# ---------------------------------------------------------------------------


def compute_snapshot(meta: EtfStaticMeta, prices: list[DailyPrice]) -> dict:
    series = to_series(prices)
    price = float(series.iloc[-1]) if len(series) else 0.0
    prev = float(series.iloc[-2]) if len(series) > 1 else price
    change = price - prev
    sparkline = [_round(v) for v in series.tail(SPARKLINE_POINTS).tolist()]

    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "category": meta.category,
        "issuer": meta.issuer,
        "price": _round(price),
        "change_amount": _round(change),
        "change_percent": _round(change / prev * 100 if prev else 0.0),
        "expense_ratio": meta.expense_ratio,
        "risk_level": meta.risk_level,
        "sparkline": sparkline,
    }


def compute_quote(meta: EtfStaticMeta, prices: list[DailyPrice]) -> dict:
    last = prices[-1]
    prev_close = prices[-2].adjusted_close if len(prices) > 1 else last.adjusted_close
    price = last.adjusted_close
    change = price - prev_close

    return {
        "symbol": meta.symbol,
        "price": _round(price),
        "change_amount": _round(change),
        "change_percent": _round(change / prev_close * 100 if prev_close else 0.0),
        "open": _round(last.open),
        "high": _round(last.high),
        "low": _round(last.low),
        "previous_close": _round(prev_close),
        "volume": int(last.volume or 0),
        "as_of": last.date.isoformat(),
    }


# ---------------------------------------------------------------------------
# Performance chart series
# ---------------------------------------------------------------------------


def compute_performance_series(prices: list[DailyPrice]) -> dict[str, list[dict]]:
    """Record<"1M"|"3M"|"6M"|"1Y"|"5Y", {label, value}[]> for Recharts."""
    series = to_series(prices)
    result: dict[str, list[dict]] = {}
    if series.empty:
        return {key: [] for key in RANGE_CONFIG}

    end = series.index[-1]
    for key, (months, label_fmt) in RANGE_CONFIG.items():
        window = series[series.index >= end - pd.DateOffset(months=months)]
        points = _downsample(window)
        result[key] = [
            {"label": ts.strftime(label_fmt), "value": _round(val)}
            for ts, val in points.items()
        ]
    return result


def _downsample(series: pd.Series, max_points: int = MAX_CHART_POINTS) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = math.ceil(len(series) / max_points)
    sampled = series.iloc[::step]
    # Always keep the most recent point so the chart ends "today"
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return sampled


# ---------------------------------------------------------------------------
# Performance stats & risk metrics
# ---------------------------------------------------------------------------


def _return_since(series: pd.Series, start: pd.Timestamp) -> float:
    """Total return (%) from the last observation at/before `start` to now."""
    window = series[series.index >= start]
    if window.empty:
        window = series
    base_idx = series.index.searchsorted(start)
    # Prefer the observation just before the window so a full period is covered
    base = series.iloc[max(base_idx - 1, 0)]
    end = series.iloc[-1]
    return (end / base - 1.0) * 100 if base else 0.0


def compute_performance_stats(prices: list[DailyPrice]) -> dict:
    series = to_series(prices)
    if series.empty:
        return {
            k: 0.0 for k in ("ytd", "one_year", "three_year", "five_year", "annualized")
        }

    end = series.index[-1]
    year_start = pd.Timestamp(year=end.year, month=1, day=1)

    five_year_start = end - pd.DateOffset(years=5)
    window = series[series.index >= five_year_start]
    base = float(window.iloc[0])
    years = max((end - window.index[0]).days / 365.25, 1e-6)
    annualized = (
        ((float(series.iloc[-1]) / base) ** (1.0 / years) - 1.0) * 100 if base else 0.0
    )

    return {
        "ytd": _round(_return_since(series, year_start)),
        "one_year": _round(_return_since(series, end - pd.DateOffset(years=1))),
        "three_year": _round(_return_since(series, end - pd.DateOffset(years=3))),
        "five_year": _round(_return_since(series, five_year_start)),
        "annualized": _round(annualized),
    }


def _risk_score(volatility_pct: float) -> float:
    """Map annualized volatility (%) onto a 1-10 rule-based score."""
    bands = [3, 6, 9, 12, 15, 18, 22, 27, 33]
    for score, upper in enumerate(bands, start=1):
        if volatility_pct < upper:
            return float(score)
    return 10.0


def compute_max_drawdown(prices: list[DailyPrice]) -> float:
    """Maximum peak-to-trough decline (%) over the full price history."""
    series = to_series(prices)
    if len(series) < 2:
        return 0.0
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return _round(float(drawdown.min()) * 100)


def compute_risk_metrics(
    prices: list[DailyPrice],
    benchmark_prices: list[DailyPrice],
    is_benchmark: bool = False,
) -> dict:
    series = to_series(prices)
    if len(series) < 2:
        return {
            "risk_score": 1.0,
            "volatility": 0.0,
            "beta": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    end = series.index[-1]
    one_year = series[series.index >= end - pd.DateOffset(years=1)]
    returns = one_year.pct_change().dropna()

    volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))  # decimal

    # Beta vs benchmark over the trailing year
    if is_benchmark:
        beta = 1.0
    else:
        bench = to_series(benchmark_prices)
        bench_returns = bench.pct_change().dropna()
        aligned = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
        if len(aligned) > 20 and float(aligned.iloc[:, 1].var()) > 0:
            beta = float(aligned.cov().iloc[0, 1] / aligned.iloc[:, 1].var())
        else:
            beta = 0.0

    # Sharpe: trailing-year annualized return over annualized volatility
    if len(one_year) > 1 and float(one_year.iloc[0]) > 0:
        annual_return = float(one_year.iloc[-1]) / float(one_year.iloc[0]) - 1.0
    else:
        annual_return = 0.0
    sharpe = (
        (annual_return - settings.risk_free_rate) / volatility if volatility else 0.0
    )

    volatility_pct = volatility * 100
    max_dd = compute_max_drawdown(prices)

    return {
        "risk_score": _risk_score(volatility_pct),
        "volatility": _round(volatility_pct),
        "beta": _round(beta),
        "sharpe_ratio": _round(sharpe),
        "max_drawdown": max_dd,
    }


def compute_index_snapshot(meta: MarketIndexMeta, prices: list[DailyPrice]) -> dict:
    """Dashboard widget payload for a market index (level, not price)."""
    series = to_series(prices)
    level = float(series.iloc[-1]) if len(series) else 0.0
    prev = float(series.iloc[-2]) if len(series) > 1 else level
    change = level - prev

    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "level": _round(level),
        "change_amount": _round(change),
        "change_percent": _round(change / prev * 100 if prev else 0.0),
        "sparkline": [_round(v) for v in series.tail(SPARKLINE_POINTS).tolist()],
    }


def compute_allocation(
    symbols: list[str],
    weights: list[float],
    all_prices: list[list[DailyPrice]],
) -> dict:
    """
    Blended portfolio metrics for an allocation mix (daily-rebalanced,
    rule-based). Weights are normalized to sum to 100%.
    """
    total = sum(weights)
    norm = [w / total for w in weights]

    series_list = [to_series(p) for p in all_prices]
    df = pd.concat(series_list, axis=1, join="inner").dropna()
    df.columns = symbols

    empty_metrics = {
        "annualized_return": 0.0,
        "volatility": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "risk_score": 1.0,
    }
    allocations = [
        {"symbol": s, "weight": _round(w * 100)} for s, w in zip(symbols, norm)
    ]
    if len(df) < 30:
        return {
            "allocations": allocations,
            "metrics": empty_metrics,
            "correlation_matrix": compute_correlation_matrix(all_prices),
        }

    returns = df.pct_change().dropna()
    blend_returns = (returns * norm).sum(axis=1)
    growth = (1.0 + blend_returns).cumprod()

    years = max((growth.index[-1] - growth.index[0]).days / 365.25, 1e-6)
    cagr = float(growth.iloc[-1]) ** (1.0 / years) - 1.0
    volatility = float(blend_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = (cagr - settings.risk_free_rate) / volatility if volatility else 0.0

    cumulative_max = growth.cummax()
    max_dd = float(((growth - cumulative_max) / cumulative_max).min())

    volatility_pct = volatility * 100
    return {
        "allocations": allocations,
        "metrics": {
            "annualized_return": _round(cagr * 100),
            "volatility": _round(volatility_pct),
            "sharpe_ratio": _round(sharpe),
            "max_drawdown": _round(max_dd * 100),
            "risk_score": _risk_score(volatility_pct),
        },
        "correlation_matrix": compute_correlation_matrix(all_prices),
    }


def compute_correlation_matrix(
    all_prices: list[list[DailyPrice]],
) -> list[list[float]]:
    """Compute correlation matrix across multiple ETFs.
    Returns a 2D list of correlations aligned to input order."""
    if len(all_prices) < 2:
        return [[1.0]]

    all_series = []
    for prices in all_prices:
        s = to_series(prices)
        if not s.empty:
            all_series.append(s.pct_change().dropna())
        else:
            all_series.append(pd.Series(dtype=float))

    # Align all return series on common dates
    df = pd.concat(all_series, axis=1, join="inner").dropna()
    if len(df) < 10:
        n = len(all_prices)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    corr = df.corr()
    return [
        [_round(corr.iloc[i, j]) for j in range(len(corr))] for i in range(len(corr))
    ]


# ---------------------------------------------------------------------------
# Full details payload
# ---------------------------------------------------------------------------


def compute_details(
    meta: EtfStaticMeta,
    prices: list[DailyPrice],
    benchmark_prices: list[DailyPrice],
) -> dict:
    snapshot = compute_snapshot(meta, prices)
    return {
        "symbol": meta.symbol,
        "name": meta.name,
        "category": meta.category,
        "issuer": meta.issuer,
        "price": snapshot["price"],
        "change_amount": snapshot["change_amount"],
        "change_percent": snapshot["change_percent"],
        "expense_ratio": meta.expense_ratio,
        "risk_level": meta.risk_level,
        "fund_objective": meta.fund_objective,
        "fund_type": meta.fund_type,
        "asset_class": meta.asset_class,
        "inception_date": meta.inception_date,
        "aum": meta.aum_usd,
        "dividend_yield": meta.dividend_yield,
        "holdings": ETF_HOLDINGS.get(meta.symbol, []),
        "performance_stats": compute_performance_stats(prices),
        "risk_metrics": compute_risk_metrics(
            prices, benchmark_prices, is_benchmark=meta.symbol == "SPY"
        ),
        "education": ETF_EDUCATION.get(meta.symbol, ""),
        "related_symbols": meta.related_symbols,
    }
