"""
Strategy Analytics engine.

Analytical / visualization only — this module computes indicator values
(EMA, RSI, MACD, Bollinger Bands, SMA, daily-range offsets) and descriptive
flags (trend, pullback, crossover, breakout) for pre-defined technical
strategies. It NEVER produces buy/sell/hold signals or recommendations,
and contains no ML. All indicators are computed manually with pandas/numpy
— no external TA libraries.

Same shape as services/analytics.py: pure functions over `list[DailyPrice]`
already loaded from the database (see routers.deps.load_prices), no DB or
HTTP access here, no external calls.
"""

import dataclasses
import datetime
import math
import numbers

import numpy as np
import pandas as pd

from app.constants import ETF_REGISTRY
from app.models import DailyPrice

# Chart payload: last N trading days of indicator history for the frontend
CHART_LOOKBACK_DAYS = 180

EMA50_PERIOD = 50
RSI_PERIOD = 14
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# Minimum closes required before indicator values are considered stable
# enough to report (period length plus a warm-up buffer for the
# exponential smoothing to converge).
MIN_ROWS_EMA50_RSI = EMA50_PERIOD + RSI_PERIOD + 10
MIN_ROWS_EMA8_21 = EMA21_PERIOD + 10

PULLBACK_THRESHOLD_EMA50_RSI = 2.0  # percent
PULLBACK_THRESHOLD_EMA8_21 = 1.0  # percent

MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
MIN_ROWS_MACD = MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD + 10

BOLLINGER_PERIOD = 20
BOLLINGER_STD_MULTIPLIER = 2
MIN_ROWS_BOLLINGER = BOLLINGER_PERIOD + 10
BAND_NEAR_THRESHOLD_PERCENT = 1.0  # percent distance to band counted as "near"

BETTER_BREAKOUT_OFFSET_LOOKBACK = 3
BETTER_BREAKOUT_SMA_PERIOD = 40
MIN_ROWS_BETTER_BREAKOUT = BETTER_BREAKOUT_SMA_PERIOD + BETTER_BREAKOUT_OFFSET_LOOKBACK + 10

SMA_TREND_PERIODS = (20, 50, 100, 200)
MIN_ROWS_SMA_TREND = max(SMA_TREND_PERIODS) + 10

# Triple-MA Pullback: MA1 = SMA(close), MA2 = SMA(MA1), MA3 = SMA(MA2), all
# using the same period ("nested" moving averages). The reference material
# describes the nesting mechanism but does not specify a period, so 20 is
# used here as the documented default (a standard trend-length lookback
# consistent with the other SMA-based strategies in this module, e.g. the
# 20/50/100/200 set used by SMA Trend above). ATR uses the conventional
# Wilder-smoothed 14-period True Range.
TRIPLE_MA_PERIOD = 20
TRIPLE_MA_MEAN_PERIOD = 5
TRIPLE_MA_ATR_PERIOD = 14
TRIPLE_MA_ENTRY_ATR_MULTIPLIER = 1.0
TRIPLE_MA_EXIT_ATR_MULTIPLIER = 0.5
# MA3 needs MA2 to have warmed up, which needs MA1 to have warmed up first
# (SMA-of-SMA-of-SMA), i.e. ~3x the period before MA3 is stable, plus ATR's
# own warm-up and a small buffer.
MIN_ROWS_TRIPLE_MA = 3 * TRIPLE_MA_PERIOD + TRIPLE_MA_ATR_PERIOD + 10

# TLT Monthly Cycle: calendar/seasonality strategy, not indicator-based.
# Trading-day-of-month windows (business days, i.e. weekdays only — this is
# an approximation since no market-holiday calendar is available; see
# compute_tlt_monthly_cycle docstring for the limitation). Values are
# documented, configurable defaults implementing the described pattern
# ("short near month start, held a few days; long shortly before month end,
# exited around month end") — they are not backtested/optimized figures
# from the reference material, since no such figures were supplied.
TLT_SHORT_HOLD_TRADING_DAYS = 3  # short position open on trading days 1..3
TLT_LONG_WINDOW_TRADING_DAYS = 5  # long position open for the last 5 trading days
# Minimum history required: this strategy only needs the most recent price
# point (for the "current price" readout) plus enough days to safely
# resolve a calendar month — a small, fixed floor is sufficient.
MIN_ROWS_TLT_MONTHLY_CYCLE = 5


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def to_close_series(prices: list[DailyPrice]) -> pd.Series:
    """Close-price series indexed by date, ascending.

    Uses raw `close` (not `adjusted_close`) since EMA/RSI trend-following
    strategies are conventionally computed on unadjusted close price.
    """
    if not prices:
        return pd.Series(dtype=float)
    return pd.Series(
        [p.close for p in prices],
        index=pd.DatetimeIndex([p.date for p in prices]),
        dtype=float,
    ).sort_index()


def to_ohlc_frame(prices: list[DailyPrice]) -> pd.DataFrame:
    """Open/High/Low/Close frame indexed by date, ascending.

    Used by strategies (e.g. Better Breakout) that need the daily range,
    not just the close series. `open` is additive (candle-colour based
    strategies need it); existing callers only select high/low/close.
    """
    if not prices:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    return pd.DataFrame(
        {
            "open": [p.open for p in prices],
            "high": [p.high for p in prices],
            "low": [p.low for p in prices],
            "close": [p.close for p in prices],
        },
        index=pd.DatetimeIndex([p.date for p in prices]),
    ).sort_index()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def _rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder-smoothed RSI (standard 14-period RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # No losses in the lookback window -> maximally overbought reading
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def _chart_rows(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Last CHART_LOOKBACK_DAYS rows of `df` as JSON-ready dicts."""
    tail = df.tail(CHART_LOOKBACK_DAYS)
    rows = []
    for ts, row in tail.iterrows():
        entry: dict = {"date": ts.date().isoformat()}
        for col in columns:
            entry[col] = _round(row[col])
        rows.append(entry)
    return rows


# ---------------------------------------------------------------------------
# Strategy 1: EMA50 + RSI Pullback
# ---------------------------------------------------------------------------


def compute_ema50_rsi_pullback(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    EMA50/RSI14 reading — callers should treat that as "unavailable",
    never as a zeroed-out result."""
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_EMA50_RSI:
        return None

    ema50 = _ema(series, EMA50_PERIOD)
    rsi = _rsi(series, RSI_PERIOD)

    df = pd.DataFrame({"close": series, "ema50": ema50, "rsi": rsi}).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    ema50_val = float(last["ema50"])
    rsi_val = float(last["rsi"])

    price_above_ema50 = price > ema50_val
    trend = "Bullish" if price_above_ema50 else "Bearish"
    distance_pct = ((price - ema50_val) / ema50_val) * 100 if ema50_val else 0.0

    rsi_turning_up = False
    if len(df) > 1:
        prev_rsi = float(df.iloc[-2]["rsi"])
        rsi_turning_up = rsi_val > prev_rsi

    # "Momentum" per the spec (RSI > 40) is listed as a calculation input
    # but is not one of the published Return fields, so it isn't included
    # in the response dict. Pullback is defined purely from
    # distance-from-EMA50, matching the spec's Pullback Detection rule.
    pullback_active = price_above_ema50 and abs(distance_pct) <= PULLBACK_THRESHOLD_EMA50_RSI

    return {
        "price": _round(price),
        "ema50": _round(ema50_val),
        "rsi": _round(rsi_val),
        "trend": trend,
        "price_above_ema50": price_above_ema50,
        "distance_from_ema50_percent": _round(distance_pct),
        "rsi_turning_up": rsi_turning_up,
        "pullback_active": pullback_active,
        "chart_data": _chart_rows(df, ["close", "ema50", "rsi"]),
    }


# ---------------------------------------------------------------------------
# Strategy 2: EMA8 / EMA21 Pullback
# ---------------------------------------------------------------------------


def compute_ema8_21_pullback(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    EMA8/EMA21 reading."""
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_EMA8_21:
        return None

    ema8 = _ema(series, EMA8_PERIOD)
    ema21 = _ema(series, EMA21_PERIOD)

    df = pd.DataFrame({"close": series, "ema8": ema8, "ema21": ema21}).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    ema8_val = float(last["ema8"])
    ema21_val = float(last["ema21"])

    trend = "Bullish" if ema8_val > ema21_val else "Bearish"
    price_above_both = price > ema8_val and price > ema21_val
    distance_to_ema8_pct = ((price - ema8_val) / ema8_val) * 100 if ema8_val else 0.0
    pullback_active = (
        price_above_both and abs(distance_to_ema8_pct) <= PULLBACK_THRESHOLD_EMA8_21
    )

    bullish_cross = False
    bearish_cross = False
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_ema8, prev_ema21 = float(prev["ema8"]), float(prev["ema21"])
        bullish_cross = prev_ema8 <= prev_ema21 and ema8_val > ema21_val
        bearish_cross = prev_ema8 >= prev_ema21 and ema8_val < ema21_val

    return {
        "price": _round(price),
        "ema8": _round(ema8_val),
        "ema21": _round(ema21_val),
        "trend": trend,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "price_above_both_emas": price_above_both,
        "distance_to_ema8_percent": _round(distance_to_ema8_pct),
        "pullback_active": pullback_active,
        "chart_data": _chart_rows(df, ["close", "ema8", "ema21"]),
    }


# ---------------------------------------------------------------------------
# Strategy 3: MACD (Moving Average Convergence Divergence)
# ---------------------------------------------------------------------------


def compute_macd(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    EMA12/EMA26/Signal(9) reading."""
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_MACD:
        return None

    ema12 = _ema(series, MACD_FAST_PERIOD)
    ema26 = _ema(series, MACD_SLOW_PERIOD)
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(
        span=MACD_SIGNAL_PERIOD, adjust=False, min_periods=MACD_SIGNAL_PERIOD
    ).mean()
    histogram = macd_line - signal_line

    df = pd.DataFrame(
        {
            "close": series,
            "ema12": ema12,
            "ema26": ema26,
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    ema12_val = float(last["ema12"])
    ema26_val = float(last["ema26"])
    macd_val = float(last["macd"])
    signal_val = float(last["signal"])
    histogram_val = float(last["histogram"])

    trend = "Bullish" if macd_val > signal_val else "Bearish"

    bullish_crossover = False
    bearish_crossover = False
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_macd, prev_signal = float(prev["macd"]), float(prev["signal"])
        bullish_crossover = prev_macd <= prev_signal and macd_val > signal_val
        bearish_crossover = prev_macd >= prev_signal and macd_val < signal_val

    return {
        "price": _round(price),
        "ema12": _round(ema12_val),
        "ema26": _round(ema26_val),
        "macd_line": _round(macd_val),
        "signal_line": _round(signal_val),
        "histogram": _round(histogram_val),
        "trend": trend,
        "bullish_crossover": bullish_crossover,
        "bearish_crossover": bearish_crossover,
        "chart_data": _chart_rows(df, ["close", "macd", "signal", "histogram"]),
    }


# ---------------------------------------------------------------------------
# Strategy 4: Bollinger Bands
# ---------------------------------------------------------------------------


def compute_bollinger_bands(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    SMA20 / standard-deviation reading."""
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_BOLLINGER:
        return None

    sma20 = series.rolling(window=BOLLINGER_PERIOD).mean()
    std20 = series.rolling(window=BOLLINGER_PERIOD).std()
    upper_band = sma20 + BOLLINGER_STD_MULTIPLIER * std20
    lower_band = sma20 - BOLLINGER_STD_MULTIPLIER * std20

    df = pd.DataFrame(
        {
            "close": series,
            "sma20": sma20,
            "upper_band": upper_band,
            "lower_band": lower_band,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    sma20_val = float(last["sma20"])
    upper_val = float(last["upper_band"])
    lower_val = float(last["lower_band"])

    dist_to_upper_pct = (
        abs((price - upper_val) / upper_val) * 100 if upper_val else float("inf")
    )
    dist_to_lower_pct = (
        abs((price - lower_val) / lower_val) * 100 if lower_val else float("inf")
    )
    price_near_upper_band = dist_to_upper_pct <= BAND_NEAR_THRESHOLD_PERCENT
    price_near_lower_band = dist_to_lower_pct <= BAND_NEAR_THRESHOLD_PERCENT

    if price > upper_val:
        breakout_status = "Above Upper Band"
    elif price < lower_val:
        breakout_status = "Below Lower Band"
    else:
        breakout_status = "Within Bands"

    band_width_percent = ((upper_val - lower_val) / sma20_val) * 100 if sma20_val else 0.0

    return {
        "price": _round(price),
        "sma20": _round(sma20_val),
        "upper_band": _round(upper_val),
        "lower_band": _round(lower_val),
        "price_near_upper_band": price_near_upper_band,
        "price_near_lower_band": price_near_lower_band,
        "breakout_status": breakout_status,
        "band_width_percent": _round(band_width_percent),
        "chart_data": _chart_rows(df, ["close", "sma20", "upper_band", "lower_band"]),
    }


# ---------------------------------------------------------------------------
# Strategy 5: Better Breakout
# ---------------------------------------------------------------------------


def compute_better_breakout(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    SMA40 / 3-day offset-average reading.

    Daily range offsets (High-Close and Close-Low) are computed per day
    from that day's own OHLC. The breakout price for a given day is
    yesterday's close plus the average of the previous 3 days' max
    offsets — i.e. everything used to project the breakout level is
    known as of yesterday's close.
    """
    frame = to_ohlc_frame(prices)
    if len(frame) < MIN_ROWS_BETTER_BREAKOUT:
        return None

    high_offset = frame["high"] - frame["close"]
    low_offset = frame["close"] - frame["low"]
    max_offset = pd.concat([high_offset, low_offset], axis=1).max(axis=1)

    average_offset = (
        max_offset.shift(1).rolling(window=BETTER_BREAKOUT_OFFSET_LOOKBACK).mean()
    )
    breakout_price = frame["close"].shift(1) + average_offset
    sma40 = frame["close"].rolling(window=BETTER_BREAKOUT_SMA_PERIOD).mean()
    momentum = frame["close"] - sma40

    df = pd.DataFrame(
        {
            "close": frame["close"],
            "high_offset": high_offset,
            "low_offset": low_offset,
            "max_offset": max_offset,
            "average_offset": average_offset,
            "breakout_price": breakout_price,
            "sma40": sma40,
            "momentum": momentum,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    breakout_price_val = float(last["breakout_price"])

    return {
        "price": _round(price),
        "high_offset": _round(float(last["high_offset"])),
        "low_offset": _round(float(last["low_offset"])),
        "max_offset": _round(float(last["max_offset"])),
        "average_offset": _round(float(last["average_offset"])),
        "breakout_price": _round(breakout_price_val),
        "sma40": _round(float(last["sma40"])),
        "momentum": _round(float(last["momentum"])),
        "breakout_active": price > breakout_price_val,
        "chart_data": _chart_rows(df, ["close", "breakout_price", "sma40"]),
    }


# ---------------------------------------------------------------------------
# Strategy 6: SMA Trend
# ---------------------------------------------------------------------------


def compute_sma_trend(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    SMA200 reading (the longest of the four SMA periods)."""
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_SMA_TREND:
        return None

    sma20 = series.rolling(window=20).mean()
    sma50 = series.rolling(window=50).mean()
    sma100 = series.rolling(window=100).mean()
    sma200 = series.rolling(window=200).mean()

    df = pd.DataFrame(
        {
            "close": series,
            "sma20": sma20,
            "sma50": sma50,
            "sma100": sma100,
            "sma200": sma200,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    sma20_val = float(last["sma20"])
    sma50_val = float(last["sma50"])
    sma100_val = float(last["sma100"])
    sma200_val = float(last["sma200"])

    price_above_sma200 = price > sma200_val
    if price > sma200_val and sma50_val > sma200_val:
        trend = "Bullish"
    elif price < sma200_val and sma50_val < sma200_val:
        trend = "Bearish"
    else:
        trend = "Neutral"

    golden_cross = False
    death_cross = False
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_sma50, prev_sma200 = float(prev["sma50"]), float(prev["sma200"])
        golden_cross = prev_sma50 <= prev_sma200 and sma50_val > sma200_val
        death_cross = prev_sma50 >= prev_sma200 and sma50_val < sma200_val

    return {
        "price": _round(price),
        "sma20": _round(sma20_val),
        "sma50": _round(sma50_val),
        "sma100": _round(sma100_val),
        "sma200": _round(sma200_val),
        "trend": trend,
        "golden_cross": golden_cross,
        "death_cross": death_cross,
        "price_above_sma200": price_above_sma200,
        "chart_data": _chart_rows(df, ["close", "sma20", "sma50", "sma100", "sma200"]),
    }


# ---------------------------------------------------------------------------
# Strategy 7: EVaR Risk / Position Sizing
#
# Entropic Value at Risk (EVaR) generalizes Gaussian VaR to better capture
# fat-tail / extreme-event risk, using ideas from Tsallis (q-exponential)
# entropy: the further a return distribution's tails deviate from a normal
# distribution, the more a naive Gaussian VaR underestimates real downside
# risk. This is a quantitative risk-sizing signal — NOT a crash predictor
# and NOT a guarantee of any future outcome.
#
# IMPLEMENTATION NOTE (read before changing constants below):
# The source article describes the concept (Tsallis/q-exponential EVaR
# used for dynamic position sizing) but not a fully reproducible formula.
# There is no attempt here to replicate a specific author's TradingView
# script. Instead this is a defensible, documented, from-first-principles
# implementation:
#
#   1. Over a trailing window of daily log returns, estimate the mean (mu),
#      standard deviation (sigma), and Fisher excess kurtosis (kappa).
#      Kurtosis is the standard, model-free way to detect fat tails.
#   2. Map kappa to a Tsallis entropic index q via
#         q = 1 + kappa_clamped / (kappa_clamped + 3)
#      This is a simple, monotonic, bounded (q in [1, ~1.8]) function we
#      define for this purpose: q = 1 (Gaussian, no tail adjustment) when
#      kappa <= 0, and q rises toward the bound as kappa grows. It is a
#      practical proxy for "how much fatter than Gaussian are the tails",
#      not a maximum-likelihood fit of a q-Gaussian distribution (that fit
#      is numerically unstable on rolling windows of daily ETF data and is
#      out of scope here).
#   3. The base risk quantile uses a fixed z-score for a 95% one-sided
#      confidence level (EVAR_CONFIDENCE_LEVEL / EVAR_Z_SCORE below), then
#      inflates it by the q-derived tail-fatness via a linear amplification
#      factor (EVAR_TAIL_AMPLIFICATION). This is a heuristic, not a closed-
#      form q-Gaussian quantile — documented as such.
#   4. EVaR is reported as a 1-day potential downside move, in percent of
#      price, at the (tail-adjusted) confidence level.
#   5. Risk classification and suggested exposure are both derived from
#      where today's EVaR sits in ITS OWN trailing distribution (a
#      percentile rank), not from a fixed universal threshold — this keeps
#      the classification meaningful across very different ETFs (e.g. GLD
#      vs QQQ) without hand-picking per-asset cutoffs.
# ---------------------------------------------------------------------------

# Trailing window (trading days) used to estimate the return distribution
# (mean / stdev / kurtosis) for each day's EVaR reading.
EVAR_RETURN_WINDOW = 252  # ~1 trading year
EVAR_MIN_RETURN_WINDOW = 60  # allow the rolling window to warm up gradually

# Trailing number of daily EVaR readings used to rank today's EVaR value
# as a percentile (0 = lowest tail-risk day in the window, 1 = highest).
EVAR_PERCENTILE_WINDOW = 120

# Total price history required before a result is returned at all.
MIN_ROWS_EVAR = EVAR_MIN_RETURN_WINDOW + EVAR_PERCENTILE_WINDOW  # 180 trading days

EVAR_CONFIDENCE_LEVEL = 95.0  # percent, one-sided
EVAR_Z_SCORE = 1.645  # standard normal one-sided 95% quantile

# Heuristic amplification of the base Gaussian z-score as the Tsallis q
# parameter departs from 1 (Gaussian). See module note above.
EVAR_TAIL_AMPLIFICATION = 1.5
EVAR_MAX_Q = 1.8

# Suggested exposure is bounded — EVaR informs sizing, it never suggests
# fully exiting a position based on a single statistical read.
EVAR_MIN_EXPOSURE_PERCENT = 20.0
EVAR_MAX_EXPOSURE_PERCENT = 100.0
# suggested_exposure = MAX_EXPOSURE - EXPOSURE_SENSITIVITY * percentile
EVAR_EXPOSURE_SENSITIVITY = 80.0


def _evar_series(series: pd.Series) -> pd.DataFrame:
    """Rolling EVaR (%) and its trailing percentile rank for every day
    that has enough trailing history. Vectorized via pandas .rolling()
    so this is cheap even over a symbol's full price history.
    """
    log_returns = np.log(series / series.shift(1))

    roll = log_returns.rolling(
        window=EVAR_RETURN_WINDOW, min_periods=EVAR_MIN_RETURN_WINDOW
    )
    mu = roll.mean()
    sigma = roll.std()
    kappa = roll.kurt()  # pandas .kurt() is Fisher (excess) kurtosis

    kappa_clamped = kappa.clip(lower=0)
    q = 1 + kappa_clamped / (kappa_clamped + 3)
    q = q.clip(upper=EVAR_MAX_Q)

    z_adjusted = EVAR_Z_SCORE * (1 + EVAR_TAIL_AMPLIFICATION * (q - 1))
    evar_pct = ((z_adjusted * sigma) - mu) * 100
    evar_pct = evar_pct.clip(lower=0)

    percentile = evar_pct.rolling(
        window=EVAR_PERCENTILE_WINDOW, min_periods=EVAR_PERCENTILE_WINDOW
    ).rank(pct=True)

    return pd.DataFrame(
        {
            "close": series,
            "evar_percent": evar_pct,
            "tsallis_q": q,
            "tail_risk_percentile": percentile * 100,
        }
    )


def _evar_classification(percentile: float) -> tuple[str, str]:
    """(tail_risk_level, risk_regime) from a 0-100 percentile rank of
    today's EVaR within its own trailing history."""
    if percentile < 33:
        return "Low", "Calm"
    if percentile < 66:
        return "Moderate", "Normal"
    if percentile < 85:
        return "Elevated", "Elevated"
    return "High", "Stressed"


def compute_evar_risk(prices: list[DailyPrice]) -> dict | None:
    """EVaR Risk / Position Sizing strategy.

    Returns None when there isn't enough price history for a stable
    rolling return-distribution estimate plus a meaningful percentile
    ranking (see MIN_ROWS_EVAR) — same "unavailable" convention as every
    other strategy in this module. Never returns a zeroed-out result for
    insufficient data.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_EVAR:
        return None

    df = _evar_series(series).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    evar_pct = float(last["evar_percent"])
    tsallis_q = float(last["tsallis_q"])
    percentile = float(last["tail_risk_percentile"])

    tail_risk_level, risk_regime = _evar_classification(percentile)

    suggested_exposure = EVAR_MAX_EXPOSURE_PERCENT - EVAR_EXPOSURE_SENSITIVITY * (
        percentile / 100
    )
    suggested_exposure = max(
        EVAR_MIN_EXPOSURE_PERCENT, min(EVAR_MAX_EXPOSURE_PERCENT, suggested_exposure)
    )

    return {
        "price": _round(price),
        "evar_percent": _round(evar_pct),
        "tsallis_q": _round(tsallis_q, 3),
        "confidence_level": EVAR_CONFIDENCE_LEVEL,
        "lookback_days": min(len(series), EVAR_RETURN_WINDOW),
        "tail_risk_percentile": _round(percentile),
        "tail_risk_level": tail_risk_level,
        "risk_regime": risk_regime,
        "suggested_exposure_percent": _round(suggested_exposure),
        "chart_data": _chart_rows(
            df, ["close", "evar_percent", "tail_risk_percentile"]
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 8: Triple-MA Pullback
#
# Trend-following + mean-reversion strategy:
#   1. Confirm an uptrend via three nested SMAs: MA1 = SMA(close), MA2 =
#      SMA(MA1), MA3 = SMA(MA2). Uptrend is confirmed when MA1 > MA2 > MA3
#      and close > MA3.
#   2. Only once an uptrend is confirmed, look for a pullback: the 5-day
#      mean (SMA5 of close) and ATR(14) define an entry zone at
#      mean - 1.0*ATR and an exit zone at mean + 0.5*ATR.
#
# Same "analytical only" convention as every other strategy in this
# module: booleans/status strings describe conditions (uptrend confirmed,
# pullback zone reached, exit zone reached) — this is not a buy/sell
# recommendation.
# ---------------------------------------------------------------------------


def _atr(frame: pd.DataFrame, period: int = TRIPLE_MA_ATR_PERIOD) -> pd.Series:
    """Wilder-smoothed Average True Range from a high/low/close frame."""
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_triple_ma_pullback(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there isn't enough price history for a stable
    triple-nested-SMA / ATR(14) reading — same "unavailable" convention as
    every other strategy in this module."""
    frame = to_ohlc_frame(prices)
    if len(frame) < MIN_ROWS_TRIPLE_MA:
        return None

    close = frame["close"]
    ma1 = close.rolling(window=TRIPLE_MA_PERIOD).mean()
    ma2 = ma1.rolling(window=TRIPLE_MA_PERIOD).mean()
    ma3 = ma2.rolling(window=TRIPLE_MA_PERIOD).mean()
    mean5 = close.rolling(window=TRIPLE_MA_MEAN_PERIOD).mean()
    atr14 = _atr(frame, TRIPLE_MA_ATR_PERIOD)

    entry_level = mean5 - TRIPLE_MA_ENTRY_ATR_MULTIPLIER * atr14
    exit_level = mean5 + TRIPLE_MA_EXIT_ATR_MULTIPLIER * atr14

    df = pd.DataFrame(
        {
            "close": close,
            "ma1": ma1,
            "ma2": ma2,
            "ma3": ma3,
            "mean5": mean5,
            "atr14": atr14,
            "entry_level": entry_level,
            "exit_level": exit_level,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    ma1_val = float(last["ma1"])
    ma2_val = float(last["ma2"])
    ma3_val = float(last["ma3"])
    mean5_val = float(last["mean5"])
    atr14_val = float(last["atr14"])
    entry_level_val = float(last["entry_level"])
    exit_level_val = float(last["exit_level"])

    uptrend_confirmed = ma1_val > ma2_val and ma2_val > ma3_val and price > ma3_val
    trend = "Uptrend" if uptrend_confirmed else "No Confirmed Uptrend"

    pullback_active = uptrend_confirmed and price < entry_level_val
    exit_zone_active = price > exit_level_val

    if pullback_active:
        current_signal = "Pullback Entry Zone"
    elif uptrend_confirmed and exit_zone_active:
        current_signal = "Uptrend — Exit Zone"
    elif uptrend_confirmed:
        current_signal = "Uptrend — No Pullback"
    else:
        current_signal = "No Confirmed Uptrend"

    return {
        "price": _round(price),
        "ma1": _round(ma1_val),
        "ma2": _round(ma2_val),
        "ma3": _round(ma3_val),
        "mean5": _round(mean5_val),
        "atr14": _round(atr14_val),
        "entry_level": _round(entry_level_val),
        "exit_level": _round(exit_level_val),
        "trend": trend,
        "uptrend_confirmed": uptrend_confirmed,
        "pullback_active": pullback_active,
        "exit_zone_active": exit_zone_active,
        "current_signal": current_signal,
        "chart_data": _chart_rows(
            df, ["close", "ma1", "ma2", "ma3", "entry_level", "exit_level"]
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 9: TLT Monthly Cycle
#
# Calendar/seasonality strategy — NOT indicator-based (no EMA, RSI, MACD,
# Bollinger, or moving averages are used here). The described cycle:
#   - Short near the start of the month, exit the short after a few
#     trading days.
#   - Go long shortly before month-end, exit the long around month-end.
#   - Repeat monthly.
#
# Trading-day-of-month position is approximated using business days
# (Mon-Fri) within the calendar month of the most recent available price,
# since no exchange-holiday calendar is available in this codebase — see
# the "limitations" note in the response docstring (schemas.strategy).
# This is a descriptive calendar readout only (current phase / position /
# key dates), not a buy/sell recommendation, matching every other strategy
# in this module.
# ---------------------------------------------------------------------------


def _business_days_of_month(year: int, month: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(0)
    return pd.bdate_range(start=start, end=end)


def _tlt_cycle_dates(bdays: pd.DatetimeIndex) -> dict[str, datetime.date]:
    num_days = len(bdays)
    short_entry_idx = 1
    short_exit_idx = min(TLT_SHORT_HOLD_TRADING_DAYS, num_days)
    long_entry_idx = max(num_days - TLT_LONG_WINDOW_TRADING_DAYS + 1, short_exit_idx + 1)
    long_entry_idx = min(long_entry_idx, num_days)
    long_exit_idx = num_days

    return {
        "short_entry_date": bdays[short_entry_idx - 1].date(),
        "short_exit_date": bdays[short_exit_idx - 1].date(),
        "long_entry_date": bdays[long_entry_idx - 1].date(),
        "long_exit_date": bdays[long_exit_idx - 1].date(),
    }


def compute_tlt_monthly_cycle(prices: list[DailyPrice]) -> dict | None:
    """Returns None when there's no TLT price history yet — same
    "unavailable" convention as every other strategy in this module.

    `prices` is TLT's own daily-bar history (loaded the same way VT/VXUS
    are loaded directly by symbol for the VT-vs-VTI+VXUS strategy — see
    routers/strategy.py), independent of whichever symbol's page the
    caller is viewing.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_TLT_MONTHLY_CYCLE:
        return None

    last_date = series.index[-1].date()
    price = float(series.iloc[-1])

    this_month_bdays = _business_days_of_month(last_date.year, last_date.month)
    this_month_dates = _tlt_cycle_dates(this_month_bdays)

    # 1-based trading-day-of-month index for the most recent price date.
    day_positions = np.searchsorted(this_month_bdays.date, last_date)
    today_idx = int(day_positions) + 1
    num_days = len(this_month_bdays)

    short_entry_idx = 1
    short_exit_idx = min(TLT_SHORT_HOLD_TRADING_DAYS, num_days)
    long_entry_idx = max(
        num_days - TLT_LONG_WINDOW_TRADING_DAYS + 1, short_exit_idx + 1
    )
    long_entry_idx = min(long_entry_idx, num_days)
    long_exit_idx = num_days

    if today_idx <= short_exit_idx:
        current_position = "Short" if today_idx >= short_entry_idx else "Flat"
        current_phase = "Short Window (Start of Month)"
    elif today_idx < long_entry_idx:
        current_position = "Flat"
        current_phase = "Mid-Month (No Position)"
    elif today_idx <= long_exit_idx:
        current_position = "Long"
        current_phase = "Long Window (End of Month)"
    else:
        current_position = "Flat"
        current_phase = "Post Long-Exit (No Position)"

    # Next expected action/date: the next upcoming key date strictly after
    # today, rolling into next month once this month's cycle is done.
    events = [
        (short_entry_idx, "Enter Short", this_month_dates["short_entry_date"]),
        (short_exit_idx, "Exit Short", this_month_dates["short_exit_date"]),
        (long_entry_idx, "Enter Long", this_month_dates["long_entry_date"]),
        (long_exit_idx, "Exit Long", this_month_dates["long_exit_date"]),
    ]
    upcoming = [e for e in events if e[0] > today_idx]
    if upcoming:
        _, next_action, next_date = min(upcoming, key=lambda e: e[0])
    else:
        next_month = last_date.month + 1
        next_year = last_date.year
        if next_month > 12:
            next_month = 1
            next_year += 1
        next_month_dates = _tlt_cycle_dates(
            _business_days_of_month(next_year, next_month)
        )
        next_action = "Enter Short"
        next_date = next_month_dates["short_entry_date"]

    df = pd.DataFrame({"close": series}).dropna()

    return {
        "price": _round(price),
        "month": last_date.strftime("%Y-%m"),
        "current_phase": current_phase,
        "current_position": current_position,
        "short_entry_date": this_month_dates["short_entry_date"].isoformat(),
        "short_exit_date": this_month_dates["short_exit_date"].isoformat(),
        "long_entry_date": this_month_dates["long_entry_date"].isoformat(),
        "long_exit_date": this_month_dates["long_exit_date"].isoformat(),
        "next_expected_action": next_action,
        "next_expected_date": next_date.isoformat(),
        "chart_data": _chart_rows(df, ["close"]),
    }


# ---------------------------------------------------------------------------
# Strategy 11: TQQQ / TMF / IEF Rebalancing
#
# Multi-ETF portfolio-allocation and crash-defense strategy — conceptually
# closer to services/portfolio_comparison.py (which grows a hypothetical
# $-investment across a portfolio built from several underlying ETFs) than
# to the single-symbol technical indicators above, but implemented here
# alongside every other pre-defined strategy for consistency, following
# the exact same "pure function over list[DailyPrice], returns None when
# there isn't enough history" convention as Strategy 9 (TLT Monthly
# Cycle) just above.
#
# Unlike the buy-and-hold VT vs VTI+VXUS comparison, this strategy is
# path-dependent (it rebalances periodically and switches allocations
# based on a crash filter), so it is simulated day-by-day over the full
# joined price history rather than computed as a single vectorized
# snapshot. Every decision at day t only ever looks at prices up to and
# including day t — no future price is used to decide the current day's
# state (no look-ahead bias).
#
# STATE MACHINE
# --------------
# NORMAL (50% TQQQ / 50% TMF):
#   - Rebalanced back to 50/50 on the last available trading day of every
#     second calendar month.
#   - Monitored daily for the crash filter.
#
# Crash filter (NORMAL -> DEFENSIVE): if TQQQ's close-to-close return on
# a single trading day is <= TQQQ_TMF_IEF_CRASH_THRESHOLD_PERCENT, exit
# TQQQ and TMF entirely and move 100% into IEF. The TQQQ close from the
# trading day *before* the crash is stored as the pre-crash reference
# price.
#
# DEFENSIVE (100% IEF): held until TQQQ's close exceeds the stored
# pre-crash reference price (not a moving average, not a fixed recovery
# percentage — literally the pre-crash price). On recovery, the portfolio
# returns to 50% TQQQ / 50% TMF and a fresh two-month rebalancing cycle
# begins from the recovery date.
#
# This is a research/backtesting framework, not investment advice. TQQQ
# and TMF are 3x leveraged ETFs and carry substantial risk; the crash
# filter reduces exposure after a large single-day decline but does not
# guarantee protection from losses. See TQQQ_TMF_IEF_DISCLAIMER below.
# ---------------------------------------------------------------------------

TQQQ_TMF_IEF_STRATEGY_KEY = "tqqq_tmf_ief_rebalancing"

TQQQ_TMF_IEF_NORMAL_TQQQ_WEIGHT = 0.5
TQQQ_TMF_IEF_NORMAL_TMF_WEIGHT = 0.5

TQQQ_TMF_IEF_REBALANCE_FREQUENCY_MONTHS = 2

# Single-trading-day TQQQ decline that trips the crash filter. Deliberately
# NOT a drawdown-from-peak, weekly, or monthly measure — see module note
# above.
TQQQ_TMF_IEF_CRASH_THRESHOLD_PERCENT = -20.0

# Backtest defaults. $100,000 mirrors the amount used to describe the
# strategy concept; the resulting performance figures are ThrustWise's own
# calculation from ThrustWise's own ingested price history — NOT the
# source article's reported backtest (see TQQQ_TMF_IEF_BACKTEST_LABEL).
TQQQ_TMF_IEF_DEFAULT_INITIAL_INVESTMENT = 100_000.0

# Per-leg execution cost applied to every dollar of turnover (bought or
# sold) at each rebalance / crash / recovery event — a simple, documented
# slippage assumption, not a claim of matching any third party's reported
# results.
TQQQ_TMF_IEF_SLIPPAGE_RATE = 0.0025  # 0.25%

# Minimum number of joined (TQQQ ^ TMF ^ IEF) trading days required before
# a result is considered meaningful — same "unavailable" convention as
# every MIN_ROWS_* constant above.
MIN_ROWS_TQQQ_TMF_IEF = 10

TQQQ_TMF_IEF_BACKTEST_LABEL = "ThrustWise calculated backtest"

TQQQ_TMF_IEF_DISCLAIMER = (
    "This strategy is a research and backtesting framework involving "
    "leveraged ETFs. Historical results do not guarantee future "
    "performance, and the crash filter does not eliminate market or "
    "execution risk."
)


def _tqqq_tmf_ief_rebalance_to_target(
    current_values: dict[str, float],
    target_weights: dict[str, float],
    total_value: float,
    slippage_rate: float = TQQQ_TMF_IEF_SLIPPAGE_RATE,
) -> dict[str, float]:
    """Move a portfolio from `current_values` (mark-to-market $ per leg) to
    `target_weights` (fraction of `total_value` per leg), charging
    `slippage_rate` against the total dollar turnover (every dollar bought
    plus every dollar sold).

    Used for every allocation change in the simulation below: the initial
    purchase, periodic 50/50 rebalances, the crash exit into IEF, and the
    recovery exit back into 50/50 TQQQ/TMF. Reused rather than duplicated
    per event type because the trade-cost mechanics are identical in each
    case — only the target weights differ.
    """
    target_values = {leg: total_value * weight for leg, weight in target_weights.items()}
    legs = set(current_values) | set(target_values)
    turnover = sum(
        abs(target_values.get(leg, 0.0) - current_values.get(leg, 0.0)) for leg in legs
    )
    cost = turnover * slippage_rate
    net_value = max(total_value - cost, 0.0)
    scale = (net_value / total_value) if total_value > 0 else 0.0
    return {leg: value * scale for leg, value in target_values.items()}


def _tqqq_tmf_ief_join_prices(
    tqqq_prices: list[DailyPrice],
    tmf_prices: list[DailyPrice],
    ief_prices: list[DailyPrice],
) -> pd.DataFrame:
    """Inner-join TQQQ/TMF/IEF close prices on date — only trading days
    where all three symbols have a real, ingested price are kept. Same
    alignment convention as portfolio_comparison.compute_vt_vs_vti_vxus:
    no missing value is ever interpolated or fabricated.

    Uses adjusted_close (via services.analytics.to_series), same series
    convention as the VT vs VTI+VXUS portfolio comparison, since this is
    a portfolio-growth simulation rather than a raw-close indicator
    reading.
    """
    from app.services.analytics import to_series  # local import: avoids a

    # module-level cycle between services.strategies and services.analytics
    tqqq_series = to_series(tqqq_prices)
    tmf_series = to_series(tmf_prices)
    ief_series = to_series(ief_prices)

    if tqqq_series.empty or tmf_series.empty or ief_series.empty:
        return pd.DataFrame(columns=["tqqq", "tmf", "ief"])

    combined = pd.concat(
        {"tqqq": tqqq_series, "tmf": tmf_series, "ief": ief_series},
        axis=1,
        join="inner",
    ).dropna()
    return combined


def _tqqq_tmf_ief_month_end_flags(index: pd.DatetimeIndex) -> list[bool]:
    """True for the last available trading day of each calendar month
    within `index` (rebalancing uses trading days, not arbitrary calendar
    dates, so "month end" means the last joined trading day actually
    observed in that month — not a fixed calendar date that might not be
    a trading day at all)."""
    n = len(index)
    flags = [False] * n
    for i in range(n):
        if i == n - 1:
            flags[i] = True
        else:
            cur, nxt = index[i], index[i + 1]
            flags[i] = cur.month != nxt.month or cur.year != nxt.year
    return flags


def compute_tqqq_tmf_ief_rebalancing(
    tqqq_prices: list[DailyPrice],
    tmf_prices: list[DailyPrice],
    ief_prices: list[DailyPrice],
    *,
    initial_investment: float = TQQQ_TMF_IEF_DEFAULT_INITIAL_INVESTMENT,
) -> dict | None:
    """Simulate the TQQQ / TMF / IEF Rebalancing strategy day-by-day over
    the full joined TQQQ/TMF/IEF price history and return the current
    strategy state plus a ThrustWise-calculated backtest equity curve.

    Returns None when any of the three symbols has no price history yet,
    or when fewer than MIN_ROWS_TQQQ_TMF_IEF joined trading days exist —
    callers should treat that as "unavailable", the same convention used
    by every other compute_* function in this module.
    """
    df = _tqqq_tmf_ief_join_prices(tqqq_prices, tmf_prices, ief_prices)
    if len(df) < MIN_ROWS_TQQQ_TMF_IEF:
        return None

    index = df.index
    tqqq = df["tqqq"].tolist()
    tmf = df["tmf"].tolist()
    ief = df["ief"].tolist()
    n = len(df)

    month_end = _tqqq_tmf_ief_month_end_flags(index)

    # --- Day 0: initial purchase, 50% TQQQ / 50% TMF -----------------------
    values = _tqqq_tmf_ief_rebalance_to_target(
        current_values={},
        target_weights={
            "tqqq": TQQQ_TMF_IEF_NORMAL_TQQQ_WEIGHT,
            "tmf": TQQQ_TMF_IEF_NORMAL_TMF_WEIGHT,
        },
        total_value=initial_investment,
    )
    shares = {
        "tqqq": values["tqqq"] / tqqq[0] if tqqq[0] else 0.0,
        "tmf": values["tmf"] / tmf[0] if tmf[0] else 0.0,
        "ief": 0.0,
    }

    state = "Normal"
    last_rebalance_date: datetime.date = index[0].date()
    next_rebalance_period = (
        pd.Period(index[0], freq="M") + TQQQ_TMF_IEF_REBALANCE_FREQUENCY_MONTHS
    )

    crash_trigger_date: datetime.date | None = None
    pre_crash_tqqq_price: float | None = None
    recovered_since_crash = True  # no crash yet == "recovered" (inactive)

    chart_rows: list[dict] = []
    equity_curve: list[float] = []

    def _mark_to_market(i: int) -> dict[str, float]:
        return {
            "tqqq": shares["tqqq"] * tqqq[i],
            "tmf": shares["tmf"] * tmf[i],
            "ief": shares["ief"] * ief[i],
        }

    for i in range(n):
        is_crash_event = False
        is_recovery_event = False
        is_rebalance_event = i == 0  # initial purchase counts as the first rebalance

        # 1) Crash filter — evaluated on consecutive trading-day closes only.
        if i > 0 and state == "Normal":
            tqqq_return_pct = (
                ((tqqq[i] - tqqq[i - 1]) / tqqq[i - 1]) * 100 if tqqq[i - 1] else 0.0
            )
            if tqqq_return_pct <= TQQQ_TMF_IEF_CRASH_THRESHOLD_PERCENT:
                mtm = _mark_to_market(i)
                total = sum(mtm.values())
                new_values = _tqqq_tmf_ief_rebalance_to_target(
                    current_values=mtm, target_weights={"ief": 1.0}, total_value=total
                )
                shares = {
                    "tqqq": 0.0,
                    "tmf": 0.0,
                    "ief": (new_values["ief"] / ief[i]) if ief[i] else 0.0,
                }
                state = "Defensive"
                pre_crash_tqqq_price = tqqq[i - 1]
                crash_trigger_date = index[i].date()
                recovered_since_crash = False
                is_crash_event = True

        # 2) Recovery — exit IEF once TQQQ's close exceeds the pre-crash price.
        elif state == "Defensive" and pre_crash_tqqq_price is not None:
            if tqqq[i] > pre_crash_tqqq_price:
                mtm = _mark_to_market(i)
                total = sum(mtm.values())
                new_values = _tqqq_tmf_ief_rebalance_to_target(
                    current_values=mtm,
                    target_weights={
                        "tqqq": TQQQ_TMF_IEF_NORMAL_TQQQ_WEIGHT,
                        "tmf": TQQQ_TMF_IEF_NORMAL_TMF_WEIGHT,
                    },
                    total_value=total,
                )
                shares = {
                    "tqqq": (new_values["tqqq"] / tqqq[i]) if tqqq[i] else 0.0,
                    "tmf": (new_values["tmf"] / tmf[i]) if tmf[i] else 0.0,
                    "ief": 0.0,
                }
                state = "Normal"
                last_rebalance_date = index[i].date()
                next_rebalance_period = (
                    pd.Period(index[i], freq="M")
                    + TQQQ_TMF_IEF_REBALANCE_FREQUENCY_MONTHS
                )
                recovered_since_crash = True
                is_recovery_event = True
                is_rebalance_event = True

        # 3) Periodic rebalance — only while Normal, on the last available
        #    trading day of every second calendar month, and never on the
        #    same day a crash/recovery just fired above.
        if (
            state == "Normal"
            and not is_crash_event
            and not is_recovery_event
            and i > 0
            and month_end[i]
            and pd.Period(index[i], freq="M") >= next_rebalance_period
        ):
            mtm = _mark_to_market(i)
            total = sum(mtm.values())
            new_values = _tqqq_tmf_ief_rebalance_to_target(
                current_values=mtm,
                target_weights={
                    "tqqq": TQQQ_TMF_IEF_NORMAL_TQQQ_WEIGHT,
                    "tmf": TQQQ_TMF_IEF_NORMAL_TMF_WEIGHT,
                },
                total_value=total,
            )
            shares = {
                "tqqq": (new_values["tqqq"] / tqqq[i]) if tqqq[i] else 0.0,
                "tmf": (new_values["tmf"] / tmf[i]) if tmf[i] else 0.0,
                "ief": 0.0,
            }
            last_rebalance_date = index[i].date()
            next_rebalance_period = (
                pd.Period(index[i], freq="M") + TQQQ_TMF_IEF_REBALANCE_FREQUENCY_MONTHS
            )
            is_rebalance_event = True

        mtm = _mark_to_market(i)
        total_value = sum(mtm.values())
        equity_curve.append(total_value)

        chart_rows.append(
            {
                "date": index[i].date().isoformat(),
                "tqqq": _round(tqqq[i]),
                "tmf": _round(tmf[i]),
                "ief": _round(ief[i]),
                "portfolio_value": _round(total_value),
                "state": state,
                "is_rebalance_event": is_rebalance_event,
                "is_crash_event": is_crash_event,
                "is_recovery_event": is_recovery_event,
            }
        )

    # --- Current state (as of the latest joined trading day) ---------------
    last_i = n - 1
    last_mtm = _mark_to_market(last_i)
    last_total = sum(last_mtm.values())
    tqqq_alloc_pct = (last_mtm["tqqq"] / last_total * 100) if last_total else 0.0
    tmf_alloc_pct = (last_mtm["tmf"] / last_total * 100) if last_total else 0.0
    ief_alloc_pct = (last_mtm["ief"] / last_total * 100) if last_total else 0.0

    tqqq_daily_return_pct = (
        ((tqqq[last_i] - tqqq[last_i - 1]) / tqqq[last_i - 1]) * 100
        if last_i > 0 and tqqq[last_i - 1]
        else 0.0
    )

    # Next scheduled rebalance date shown to the user: the last joined
    # trading day actually observed in the target month, when we have data
    # that far; otherwise the calendar month-end as a forward estimate
    # (never used to make a decision above — decisions only ever look at
    # data already observed, this is display-only).
    target_month_rows = [
        ts for ts in index if pd.Period(ts, freq="M") == next_rebalance_period
    ]
    if target_month_rows:
        next_rebalance_date = target_month_rows[-1].date().isoformat()
    else:
        next_rebalance_date = next_rebalance_period.end_time.date().isoformat()

    if state == "Normal":
        crash_filter_status = "Inactive"
        recovery_status = None
        next_expected_action = f"Rebalance TQQQ/TMF to 50/50 on {next_rebalance_date}"
    else:
        crash_filter_status = "Triggered"
        recovery_status = "Waiting for TQQQ to exceed pre-crash price"
        next_expected_action = (
            "Exit IEF and return to 50% TQQQ / 50% TMF once TQQQ closes above "
            f"its pre-crash price of {_round(pre_crash_tqqq_price)}"
        )

    # --- ThrustWise-calculated backtest summary (own data, own numbers —
    # never the source article's reported figures; see
    # TQQQ_TMF_IEF_BACKTEST_LABEL). ---
    final_value = equity_curve[-1]
    total_return_pct = (
        ((final_value - initial_investment) / initial_investment) * 100
        if initial_investment
        else 0.0
    )
    years = max((index[-1] - index[0]).days / 365.25, 1e-9)
    cagr_pct = (
        ((final_value / initial_investment) ** (1 / years) - 1) * 100
        if initial_investment > 0 and final_value > 0
        else 0.0
    )

    equity_series = pd.Series(equity_curve)
    running_peak = equity_series.cummax()
    drawdown_pct = ((equity_series - running_peak) / running_peak) * 100
    max_drawdown_pct = float(drawdown_pct.min()) if not drawdown_pct.empty else 0.0

    return {
        "strategy": TQQQ_TMF_IEF_STRATEGY_KEY,
        "state": "Normal" if state == "Normal" else "Defensive / Crash",
        "tqqq_allocation_percent": _round(tqqq_alloc_pct),
        "tmf_allocation_percent": _round(tmf_alloc_pct),
        "ief_allocation_percent": _round(ief_alloc_pct),
        "tqqq_price": _round(tqqq[last_i]),
        "tmf_price": _round(tmf[last_i]),
        "ief_price": _round(ief[last_i]),
        "tqqq_daily_return_percent": _round(tqqq_daily_return_pct),
        "last_rebalance_date": last_rebalance_date.isoformat(),
        "next_rebalance_date": next_rebalance_date,
        "rebalance_frequency_months": TQQQ_TMF_IEF_REBALANCE_FREQUENCY_MONTHS,
        "crash_filter_status": crash_filter_status,
        "crash_filter_threshold_percent": TQQQ_TMF_IEF_CRASH_THRESHOLD_PERCENT,
        "crash_trigger_date": (
            crash_trigger_date.isoformat()
            if crash_trigger_date and not recovered_since_crash
            else None
        ),
        "pre_crash_tqqq_price": (
            _round(pre_crash_tqqq_price) if state == "Defensive" else None
        ),
        "recovery_status": recovery_status,
        "next_expected_action": next_expected_action,
        "backtest_start_date": index[0].date().isoformat(),
        "backtest_end_date": index[-1].date().isoformat(),
        "backtest_initial_investment": _round(initial_investment),
        "backtest_final_value": _round(final_value),
        "backtest_total_return_percent": _round(total_return_pct),
        "backtest_cagr_percent": _round(cagr_pct),
        "backtest_max_drawdown_percent": _round(max_drawdown_pct),
        "backtest_label": TQQQ_TMF_IEF_BACKTEST_LABEL,
        "disclaimer": TQQQ_TMF_IEF_DISCLAIMER,
        "chart_data": chart_rows,
    }


# ---------------------------------------------------------------------------
# Strategy 12: Mswing Momentum
#
# Single-symbol momentum strategy — same "pure function over
# list[DailyPrice], returns None when there isn't enough history"
# convention as every other strategy in this module (most similar in
# shape to Strategy 8, Triple-MA Pullback: a momentum/trend readout with
# a documented default entry/exit condition, not a signal or
# recommendation).
#
#   Mswing = pct_change(close, 20) + pct_change(close, 50)     (percent)
#   Mswing EMA9  = EMA(Mswing, 9)
#   SMA50        = SMA(close, 50)
#
# STATE (exactly four, per the source article — no additional states are
# invented):
#   Strong bullish momentum      Mswing > 0 and Mswing > EMA9
#   Weakening bullish momentum   Mswing > 0 and Mswing < EMA9
#   Recovering bearish momentum  Mswing < 0 and Mswing > EMA9
#   Bearish momentum             Mswing < 0 and Mswing < EMA9
#
# The source material does not define a fifth "Mswing == 0" case. Since a
# reading of exactly zero represents no net momentum rather than positive
# momentum, it is treated as satisfying the "Mswing < 0" bucket for state
# classification only (a documented default, same spirit as the
# TRIPLE_MA_PERIOD assumption above) — this only affects the boundary
# value itself, never values strictly above or below zero.
#
# DEFAULT STRATEGY CONDITIONS (descriptive/analytical only — no order is
# ever placed, matching the "no trading execution" convention used by
# every strategy in this codebase):
#   Bullish:      Mswing > 0 and Mswing > EMA9 and Close > SMA50
#   Bearish/Exit: Mswing < 0 or Close < SMA50 or Mswing >= 4
#
# RELATIVE STRENGTH: Mswing(stock) - Mswing(index). QQQ is used as the
# index because it is already part of the existing ETF universe/price
# infrastructure (app.constants.ETF_REGISTRY) — no new external data
# source is introduced. When the index's own Mswing value is unavailable
# (insufficient history), relative_strength is null rather than
# fabricated, same "unavailable" convention as everywhere else in this
# module. See routers/strategy.py for how the index value is loaded and
# cached (cross-cutting, once per request cycle — same treatment as TLT
# Monthly Cycle and TQQQ/TMF/IEF above).
# ---------------------------------------------------------------------------

MSWING_SHORT_LENGTH = 20
MSWING_LONG_LENGTH = 50
MSWING_EMA_LENGTH = 9
MSWING_SMA_LENGTH = 50

# Bearish/exit condition also fires on an overextended reading, per the
# source article's default strategy conditions.
MSWING_OVEREXTENDED_THRESHOLD = 4.0

# Needs close[t-50] for the long-length pct-change leg, plus EMA(9)
# warm-up on the resulting Mswing series, plus a small buffer — same
# period-plus-buffer sizing convention as every other MIN_ROWS_* constant
# in this module (e.g. MIN_ROWS_EMA50_RSI).
MIN_ROWS_MSWING = MSWING_LONG_LENGTH + MSWING_EMA_LENGTH + 10

MSWING_INDEX_SYMBOL = "QQQ"


def _mswing_series(series: pd.Series) -> pd.DataFrame:
    """Mswing, its EMA9, and SMA50 of close — vectorized over the full
    close-price series. Shared by compute_mswing (full result) and
    compute_mswing_index_value (index-only raw value for relative
    strength) so the calculation is defined in exactly one place.
    """
    short_pct = (series / series.shift(MSWING_SHORT_LENGTH) - 1) * 100
    long_pct = (series / series.shift(MSWING_LONG_LENGTH) - 1) * 100
    mswing = short_pct + long_pct
    mswing_ema9 = _ema(mswing, MSWING_EMA_LENGTH)
    sma50 = series.rolling(window=MSWING_SMA_LENGTH).mean()

    return pd.DataFrame(
        {
            "close": series,
            "mswing": mswing,
            "mswing_ema9": mswing_ema9,
            "sma50": sma50,
        }
    )


def compute_mswing_index_value(prices: list[DailyPrice]) -> float | None:
    """The index's (QQQ's) own latest Mswing reading, used only as the
    subtrahend for relative_strength in compute_mswing below. Returns
    None when there isn't enough index price history — never fabricated.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_MSWING:
        return None
    df = _mswing_series(series).dropna(subset=["mswing"])
    if df.empty:
        return None
    return float(df.iloc[-1]["mswing"])


def compute_mswing(
    prices: list[DailyPrice],
    *,
    index_mswing: float | None = None,
    index_symbol: str = MSWING_INDEX_SYMBOL,
) -> dict | None:
    """Mswing Momentum strategy.

    Returns None when there isn't enough price history for a stable
    Mswing/EMA9/SMA50 reading (see MIN_ROWS_MSWING) — same "unavailable"
    convention as every other strategy in this module.

    `index_mswing` is the index's (QQQ's) own latest Mswing value,
    computed separately via compute_mswing_index_value and passed in by
    the caller (see routers/strategy.py) — this function never loads
    index data itself, staying a pure function over its own `prices`
    like every other strategy here.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_MSWING:
        return None

    df = _mswing_series(series).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    mswing_val = float(last["mswing"])
    mswing_ema9_val = float(last["mswing_ema9"])
    sma50_val = float(last["sma50"])

    mswing_above_zero = mswing_val > 0
    mswing_above_ema = mswing_val > mswing_ema9_val
    price_above_sma50 = price > sma50_val

    # See module note above: an exact-zero Mswing reading is bucketed
    # with "< 0" for state classification (no fifth state is invented).
    if mswing_above_zero:
        mswing_state = (
            "Strong bullish momentum"
            if mswing_above_ema
            else "Weakening bullish momentum"
        )
    else:
        mswing_state = (
            "Recovering bearish momentum"
            if mswing_above_ema
            else "Bearish momentum"
        )

    zero_line_bullish_cross = False
    zero_line_bearish_cross = False
    ema_bullish_cross = False
    ema_bearish_cross = False
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_mswing = float(prev["mswing"])
        prev_ema9 = float(prev["mswing_ema9"])
        zero_line_bullish_cross = prev_mswing <= 0 and mswing_val > 0
        zero_line_bearish_cross = prev_mswing >= 0 and mswing_val < 0
        ema_bullish_cross = prev_mswing <= prev_ema9 and mswing_val > mswing_ema9_val
        ema_bearish_cross = prev_mswing >= prev_ema9 and mswing_val < mswing_ema9_val

    bullish_condition_active = (
        mswing_above_zero and mswing_above_ema and price_above_sma50
    )
    bearish_exit_condition_active = (
        not mswing_above_zero
        or not price_above_sma50
        or mswing_val >= MSWING_OVEREXTENDED_THRESHOLD
    )

    relative_strength = (
        _round(mswing_val - index_mswing) if index_mswing is not None else None
    )

    return {
        "price": _round(price),
        "mswing": _round(mswing_val),
        "mswing_ema9": _round(mswing_ema9_val),
        "sma50": _round(sma50_val),
        "mswing_state": mswing_state,
        "mswing_above_zero": mswing_above_zero,
        "mswing_above_ema": mswing_above_ema,
        "price_above_sma50": price_above_sma50,
        "zero_line_bullish_cross": zero_line_bullish_cross,
        "zero_line_bearish_cross": zero_line_bearish_cross,
        "ema_bullish_cross": ema_bullish_cross,
        "ema_bearish_cross": ema_bearish_cross,
        "bullish_condition_active": bullish_condition_active,
        "bearish_exit_condition_active": bearish_exit_condition_active,
        "relative_strength": relative_strength,
        "relative_strength_index_symbol": index_symbol,
        "chart_data": _chart_rows(df, ["close", "mswing", "mswing_ema9", "sma50"]),
    }


# ---------------------------------------------------------------------------
# Strategy 13: Momentum Reversal
#
# Single-symbol contrarian strategy — same "pure function over
# list[DailyPrice], returns None when there isn't enough history"
# convention as every other strategy in this module.
#
# SOURCE MATERIAL NOTE: the reference article demonstrates this strategy
# on 1-minute TSLA bars:
#
#   returns = close.pct_change()
#   if returns < -0.001: go long   (BUY)
#   if returns >  0.001: go short  (SELL)
#   otherwise: no new signal, retain the previous position
#
# The article's own backtest is on a single symbol (TSLA) at 1-minute
# resolution and is not treated here as evidence the rule is profitable
# on any other symbol or timeframe.
#
# DATA LIMITATION: this codebase's ingestion pipeline (services/
# ingestion.py) only stores end-of-day bars (see app.models.DailyPrice —
# one row per trading day, no intraday timestamps), the same data source
# every other strategy in this module already uses. No new data pipeline
# or external data source is introduced to chase 1-minute bars; instead
# the identical threshold rule is applied to daily close-to-close returns
# (`to_close_series(prices).pct_change()`), i.e. the closest existing
# data resolution actually available. Because daily returns are far
# larger in magnitude than 1-minute returns, the same 0.1% thresholds
# fire on essentially every trading day at daily resolution — this is a
# faithful, undiluted application of the article's stated rule, not a
# retuned/optimized variant, and is documented here rather than silently
# rescaled.
# ---------------------------------------------------------------------------

# Exact thresholds from the source article (0.1%), applied to
# `close.pct_change()` — see module note above for the daily-vs-1-minute
# data-resolution caveat.
MOMENTUM_REVERSAL_BUY_THRESHOLD = -0.001  # return below this => bullish reversal
MOMENTUM_REVERSAL_SELL_THRESHOLD = 0.001  # return above this => bearish reversal

# Small fixed floor: the rule itself only needs two consecutive closes,
# but a short buffer is required so the position-retention chart is
# meaningful rather than a single point — same "small fixed floor"
# convention as MIN_ROWS_TLT_MONTHLY_CYCLE above.
MIN_ROWS_MOMENTUM_REVERSAL = 15


def compute_momentum_reversal(prices: list[DailyPrice]) -> dict | None:
    """Momentum Reversal strategy.

    Returns None when there isn't enough price history for even a short
    position-retention readout (see MIN_ROWS_MOMENTUM_REVERSAL) — same
    "unavailable" convention as every other strategy in this module.
    Never returns a zeroed-out result for insufficient data.

    Analytical/descriptive only: `reversal_signal` and `current_position`
    describe what the rule would flag today and the position it would be
    holding given the "retain previous position" rule — this is not a
    buy/sell recommendation or an executed trade, matching every other
    strategy in this module.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_MOMENTUM_REVERSAL:
        return None

    returns = series.pct_change()
    df = pd.DataFrame({"close": series, "return": returns}).dropna()
    if df.empty:
        return None

    signal = np.where(
        df["return"] < MOMENTUM_REVERSAL_BUY_THRESHOLD,
        "Bullish Reversal",
        np.where(
            df["return"] > MOMENTUM_REVERSAL_SELL_THRESHOLD,
            "Bearish Reversal",
            "No Signal",
        ),
    )
    df["signal"] = signal

    # Position retention: "Otherwise ... retain previous position" per
    # the article's rule — a signal-less day never resets the position
    # back to flat.
    position: list[str] = []
    current_position = "Flat"
    for s in df["signal"]:
        if s == "Bullish Reversal":
            current_position = "Long"
        elif s == "Bearish Reversal":
            current_position = "Short"
        position.append(current_position)
    df["position"] = position

    last = df.iloc[-1]
    price = float(last["close"])
    return_pct = float(last["return"]) * 100
    reversal_signal = str(last["signal"])
    current_position_val = str(last["position"])

    position_changed_today = False
    if len(df) > 1:
        position_changed_today = str(df.iloc[-2]["position"]) != current_position_val

    tail = df.tail(CHART_LOOKBACK_DAYS)
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "daily_return_percent": _round(row["return"] * 100, 3),
            "position": row["position"],
        }
        for ts, row in tail.iterrows()
    ]

    return {
        "price": _round(price),
        "daily_return_percent": _round(return_pct, 3),
        "buy_threshold_percent": _round(MOMENTUM_REVERSAL_BUY_THRESHOLD * 100, 3),
        "sell_threshold_percent": _round(MOMENTUM_REVERSAL_SELL_THRESHOLD * 100, 3),
        "reversal_signal": reversal_signal,
        "current_position": current_position_val,
        "position_changed_today": position_changed_today,
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 14: One-Month Factor Momentum
#
# Cross-ETF strategy — same cross-cutting treatment as Risk-On/Risk-Off
# (services/regime.py): computed once across the whole ETF universe and
# cached separately (see routers/strategy.py), then sliced per symbol via
# extract_factor_momentum_view below, rather than recomputed per symbol.
#
# SOURCE MATERIAL: the reference article's "factor momentum" measures the
# *previous month's* factor performance (rather than the traditional
# 12-month lookback) and rotates toward whichever factors led over that
# window.
#
# FACTOR-DATA LIMITATION (read before changing this section): this
# codebase does not ingest or compute genuine style-factor index data —
# there are no Value/Quality/Low-Volatility/Momentum factor series (e.g.
# no MTUM/VLUE/QUAL-style holdings or factor-return time series exist
# anywhere in app.constants or app.services). Fabricating one would mean
# inventing factor-return data that was never actually observed, which
# this implementation deliberately does not do.
#
# What DOES already exist in the codebase is app.constants.ETF_REGISTRY's
# `tags` field — the same category labels ("Equity", "Index",
# "Technology", "Gold", "Treasury") already used to drive the ETF
# Explorer's filter chips (app.constants.FILTER_CATEGORIES). This
# strategy reuses that existing categorical grouping as the "factor"
# grouping: a "factor's" one-month performance is the average one-month
# price return of every ETF in the existing universe that already carries
# that tag. This is a documented adaptation built entirely from data
# already present in the codebase — NOT literal Fama-French/MSCI style
# factors, and it is reported as such (see `factor_data_basis` in the
# response) so it is never mistaken for genuine factor-index data.
# ---------------------------------------------------------------------------

# ~1 trading month, matching the article's "previous 1-month" lookback
# (as opposed to the traditional 12-month factor-momentum lookback).
FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS = 21

# Needs the lookback window plus one extra day (t and t-21) plus a small
# buffer, same "period plus buffer" convention as every other MIN_ROWS_*
# constant in this module.
MIN_ROWS_FACTOR_MOMENTUM = FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS + 5

# Suggested exposure is bounded, same spirit (and same bounded-sizing
# convention) as EVAR_MIN/MAX_EXPOSURE_PERCENT above — this informs
# sizing, it never suggests fully exiting a position from a single
# cross-sectional read.
FACTOR_MOMENTUM_MIN_EXPOSURE_PERCENT = 30.0
FACTOR_MOMENTUM_MAX_EXPOSURE_PERCENT = 100.0

FACTOR_MOMENTUM_DATA_BASIS = (
    "Existing ETF_REGISTRY category tags (Equity, Index, Technology, "
    "Gold, Treasury) used as a factor-grouping proxy — not standalone "
    "style-factor (Value/Quality/Low-Vol) index data, which this "
    "codebase does not ingest."
)


def _one_month_return_percent(series: pd.Series) -> float | None:
    """Close-to-close return over FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS
    trading days, in percent. None when there isn't enough history or
    the lookback price is zero/unusable — never fabricated."""
    if len(series) < MIN_ROWS_FACTOR_MOMENTUM:
        return None
    recent = series.iloc[-1]
    lookback = series.iloc[-(FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS + 1)]
    if lookback in (0, None) or pd.isna(lookback) or pd.isna(recent):
        return None
    return float((recent / lookback - 1) * 100)


def compute_factor_momentum(
    prices_by_symbol: dict[str, list[DailyPrice]],
) -> dict | None:
    """One-Month Factor Momentum — cross-ETF computation shared by every
    symbol's response (see module note above and routers/strategy.py).

    `prices_by_symbol` is every ETF_REGISTRY symbol's own daily-bar
    history (loaded the same way as the Risk-On/Risk-Off universe load —
    see routers/strategy.py `_load_factor_momentum`). Returns None when
    fewer than two ETFs have enough history for a one-month return, since
    a meaningful factor-group ranking needs at least two ETFs.
    """
    returns_by_symbol: dict[str, float] = {}
    for symbol, prices in prices_by_symbol.items():
        series = to_close_series(prices)
        one_month_return = _one_month_return_percent(series)
        if one_month_return is not None:
            returns_by_symbol[symbol] = one_month_return

    if len(returns_by_symbol) < 2:
        return None

    tag_returns: dict[str, list[float]] = {}
    for symbol, one_month_return in returns_by_symbol.items():
        meta = ETF_REGISTRY.get(symbol)
        if meta is None:
            continue
        for tag in meta.tags:
            tag_returns.setdefault(tag, []).append(one_month_return)

    if not tag_returns:
        return None

    factor_performance = {
        tag: sum(vals) / len(vals) for tag, vals in tag_returns.items()
    }
    ranked_factors = sorted(
        factor_performance.items(), key=lambda kv: kv[1], reverse=True
    )
    factor_rankings = [
        {
            "factor": tag,
            "one_month_return_percent": _round(ret, 2),
            "rank": i + 1,
        }
        for i, (tag, ret) in enumerate(ranked_factors)
    ]
    leading_factor = ranked_factors[0][0]
    lagging_factor = ranked_factors[-1][0]

    # Symbol-level factor-momentum score: equal-weighted average of the
    # factor performance of every tag THIS symbol itself carries (a
    # symbol with no recognized tags gets no score, same "unavailable"
    # convention as everywhere else).
    symbol_scores: dict[str, float] = {}
    for symbol in returns_by_symbol:
        meta = ETF_REGISTRY.get(symbol)
        if meta is None or not meta.tags:
            continue
        tag_vals = [
            factor_performance[tag] for tag in meta.tags if tag in factor_performance
        ]
        if tag_vals:
            symbol_scores[symbol] = sum(tag_vals) / len(tag_vals)

    return {
        "lookback_trading_days": FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS,
        "factor_rankings": factor_rankings,
        "leading_factor": leading_factor,
        "lagging_factor": lagging_factor,
        "etf_universe": sorted(returns_by_symbol.keys()),
        "symbol_returns": returns_by_symbol,
        "symbol_scores": symbol_scores,
        "factor_data_basis": FACTOR_MOMENTUM_DATA_BASIS,
    }


def extract_factor_momentum_view(
    factor_momentum: dict | None,
    symbol: str,
    own_prices: list[DailyPrice],
) -> dict | None:
    """Per-symbol response slice from the shared cross-ETF computation
    (see compute_factor_momentum above), plus a symbol-own chart series —
    same pattern as regime.extract_symbol_view for Risk-On/Risk-Off.

    Returns None if the cross-ETF computation is unavailable, or if
    `symbol` specifically didn't have a usable one-month return or any
    recognized ETF_REGISTRY tags.
    """
    if factor_momentum is None:
        return None

    symbol_return = factor_momentum["symbol_returns"].get(symbol)
    symbol_score = factor_momentum["symbol_scores"].get(symbol)
    meta = ETF_REGISTRY.get(symbol)
    if symbol_return is None or symbol_score is None or meta is None:
        return None

    symbol_tags = meta.tags
    scores = list(factor_momentum["symbol_scores"].values())
    if len(scores) > 1:
        percentile = sum(1 for s in scores if s <= symbol_score) / len(scores) * 100
    else:
        percentile = 100.0

    suggested_exposure = FACTOR_MOMENTUM_MIN_EXPOSURE_PERCENT + (
        FACTOR_MOMENTUM_MAX_EXPOSURE_PERCENT - FACTOR_MOMENTUM_MIN_EXPOSURE_PERCENT
    ) * (percentile / 100)

    is_aligned_with_leading_factor = factor_momentum["leading_factor"] in symbol_tags
    is_aligned_with_lagging_factor = factor_momentum["lagging_factor"] in symbol_tags

    if is_aligned_with_leading_factor:
        # "Maintain/increase exposure to stronger factors" per the
        # source article's conceptual rule.
        exposure_guidance = "Maintain / Increase Exposure"
    elif is_aligned_with_lagging_factor:
        exposure_guidance = "Reduce Exposure"
    else:
        exposure_guidance = "Neutral"

    series = to_close_series(own_prices)
    rolling_return_pct = (
        series.pct_change(periods=FACTOR_MOMENTUM_LOOKBACK_TRADING_DAYS) * 100
    )
    chart_df = pd.DataFrame(
        {"close": series, "one_month_return_percent": rolling_return_pct}
    ).dropna()
    chart_data = (
        _chart_rows(chart_df, ["close", "one_month_return_percent"])
        if not chart_df.empty
        else []
    )

    return {
        "symbol": symbol,
        "symbol_tags": symbol_tags,
        "one_month_return_percent": _round(symbol_return, 2),
        "factor_momentum_score": _round(symbol_score, 2),
        "factor_rankings": factor_momentum["factor_rankings"],
        "leading_factor": factor_momentum["leading_factor"],
        "lagging_factor": factor_momentum["lagging_factor"],
        "is_aligned_with_leading_factor": is_aligned_with_leading_factor,
        "exposure_guidance": exposure_guidance,
        "suggested_exposure_percent": _round(suggested_exposure, 2),
        "etf_universe": factor_momentum["etf_universe"],
        "lookback_trading_days": factor_momentum["lookback_trading_days"],
        "factor_data_basis": factor_momentum["factor_data_basis"],
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 15: HMM Regime-Switching
#
# Single-symbol strategy — same "pure function over list[DailyPrice],
# returns None when there isn't enough history" convention as every other
# strategy in this module.
#
# SOURCE MATERIAL: the reference article uses a Hidden Markov Model to
# classify market regimes (e.g. bull/trending vs bear/high-volatility)
# and uses that regime classification to switch between a Momentum
# approach and a Mean-Reversion approach.
#
# NO DEPENDENCY CHANGE: no HMM library (e.g. hmmlearn) is a dependency of
# this project (see requirements.txt) and none is added here. A minimal,
# 2-state Gaussian Hidden Markov Model is instead implemented directly
# with numpy — Baum-Welch (EM) parameter estimation via the standard
# forward-backward algorithm — the same "implemented directly with
# pandas/numpy, no external library" convention already used by
# services/regime.py for its own DTW/clustering step (see that module's
# docstring) and by every indicator in this module (EMA/RSI/MACD/ATR are
# all hand-rolled above, not pulled from a TA library).
#
# MODEL: 2 hidden states over the daily log-return series (the same
# close-price data every other single-symbol strategy in this module
# uses). The discriminating feature between the two fitted states is
# volatility (variance of returns) — the lower-variance state is reported
# as "Trending / Momentum-Favorable", the higher-variance state as
# "High-Volatility / Mean-Reversion-Favorable", the standard practical
# reading of a 2-state Gaussian return-regime model. `recommended_approach`
# ("Momentum" | "Mean Reversion") reflects that regime-to-approach
# switch, same descriptive/analytical convention as every other strategy
# in this module — it is not a buy/sell recommendation or an executed
# trade.
# ---------------------------------------------------------------------------

HMM_N_STATES = 2
HMM_MAX_ITER = 100
HMM_TOLERANCE = 1e-4

# Trailing window of daily log returns used to fit the HMM. ~1 trading
# year — long enough to let both a "calm" and a "stressed" regime show up
# in-sample, short enough that the fit reflects the recent regime mix
# rather than the symbol's entire history.
HMM_RETURN_LOOKBACK_DAYS = 252

# Minimum daily bars required before a 2-state fit is considered
# meaningful — same "unavailable rather than zeroed-out" convention as
# every MIN_ROWS_* constant in this module.
MIN_ROWS_HMM_REGIME = 120


def _hmm_fit_gaussian_2state(x: np.ndarray) -> dict:
    """Minimal Baum-Welch (EM) fit of a 2-state Gaussian HMM on a 1-D
    observation series, implemented directly with numpy (see module note
    above for why no HMM library is used).

    Initialization is deterministic (median split of the observations,
    not a random draw), so this stays a pure function of its input like
    every other strategy in this module — the same result is produced
    for the same price history every time.
    """
    n = len(x)
    k = HMM_N_STATES

    median = np.median(x)
    low_obs = x[x <= median]
    high_obs = x[x > median]
    if len(low_obs) < 2 or len(high_obs) < 2:
        # Degenerate split (e.g. many repeated values) — fall back to a
        # simple positional split so the fit still has two distinct
        # starting points.
        half = max(n // 2, 1)
        low_obs = x[:half] if half >= 2 else x
        high_obs = x[half:] if (n - half) >= 2 else x

    means = np.array([float(low_obs.mean()), float(high_obs.mean())])
    variances = np.array(
        [max(float(low_obs.var()), 1e-10), max(float(high_obs.var()), 1e-10)]
    )
    # Order states by mean return (ascending) as a stable, deterministic
    # starting point — final state identity is reassigned by volatility
    # after fitting (see compute_hmm_regime_switching below).
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]

    # Sticky prior (regimes persist day-to-day far more often than they
    # switch) — a standard, documented starting point for daily-return
    # regime models, refined by EM below.
    transition = np.full((k, k), 0.05)
    np.fill_diagonal(transition, 0.95)
    initial = np.full(k, 1.0 / k)

    log_likelihood_prev = -np.inf
    for _ in range(HMM_MAX_ITER):
        emission = np.zeros((n, k))
        for state in range(k):
            var_state = max(float(variances[state]), 1e-10)
            emission[:, state] = np.exp(
                -0.5 * (x - means[state]) ** 2 / var_state
            ) / np.sqrt(2 * np.pi * var_state)
        emission = np.clip(emission, 1e-300, None)

        # Forward pass (scaled, to avoid numerical underflow over long
        # observation windows).
        alpha = np.zeros((n, k))
        scale = np.zeros(n)
        alpha[0] = initial * emission[0]
        scale[0] = alpha[0].sum() or 1e-300
        alpha[0] /= scale[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ transition) * emission[t]
            scale[t] = alpha[t].sum() or 1e-300
            alpha[t] /= scale[t]

        # Backward pass (scaled with the same factors as the forward pass).
        beta = np.zeros((n, k))
        beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (transition @ (emission[t + 1] * beta[t + 1])) / scale[t + 1]

        gamma = alpha * beta
        gamma_row_sums = gamma.sum(axis=1, keepdims=True)
        gamma_row_sums[gamma_row_sums == 0] = 1e-300
        gamma /= gamma_row_sums

        xi_sum = np.zeros((k, k))
        for t in range(n - 1):
            xi_t = (
                alpha[t][:, None]
                * transition
                * emission[t + 1][None, :]
                * beta[t + 1][None, :]
            ) / (scale[t + 1] or 1e-300)
            xi_sum += xi_t

        initial = gamma[0]
        gamma_sum_excl_last = gamma[:-1].sum(axis=0)
        gamma_sum_excl_last[gamma_sum_excl_last == 0] = 1e-300
        transition = xi_sum / gamma_sum_excl_last[:, None]

        gamma_total = gamma.sum(axis=0)
        gamma_total[gamma_total == 0] = 1e-300
        means = (gamma * x[:, None]).sum(axis=0) / gamma_total
        variances = (gamma * (x[:, None] - means[None, :]) ** 2).sum(
            axis=0
        ) / gamma_total
        variances = np.clip(variances, 1e-10, None)

        log_likelihood = float(np.sum(np.log(scale)))
        if abs(log_likelihood - log_likelihood_prev) < HMM_TOLERANCE:
            break
        log_likelihood_prev = log_likelihood

    return {
        "means": means,
        "variances": variances,
        "transition": transition,
        "gamma": gamma,
        "log_likelihood": log_likelihood_prev,
    }


def compute_hmm_regime_switching(prices: list[DailyPrice]) -> dict | None:
    """HMM Regime-Switching strategy.

    Returns None when there isn't enough daily price history for a
    meaningful 2-state fit (see MIN_ROWS_HMM_REGIME) — same "unavailable"
    convention as every other strategy in this module.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_HMM_REGIME:
        return None

    log_returns = np.log(series / series.shift(1)).dropna()
    log_returns = log_returns.tail(HMM_RETURN_LOOKBACK_DAYS)
    if len(log_returns) < MIN_ROWS_HMM_REGIME:
        return None

    x = log_returns.to_numpy(dtype=float)
    fit = _hmm_fit_gaussian_2state(x)
    means, variances = fit["means"], fit["variances"]
    transition, gamma = fit["transition"], fit["gamma"]

    trending_state = int(np.argmin(variances))
    high_vol_state = 1 - trending_state

    current_probs = gamma[-1]
    current_state = int(np.argmax(current_probs))
    current_regime = (
        "Trending / Momentum-Favorable"
        if current_state == trending_state
        else "High-Volatility / Mean-Reversion-Favorable"
    )
    recommended_approach = (
        "Momentum" if current_state == trending_state else "Mean Reversion"
    )

    regime_changed_today = False
    if len(gamma) > 1:
        previous_state = int(np.argmax(gamma[-2]))
        regime_changed_today = previous_state != current_state

    dates = log_returns.index
    state_path = np.argmax(gamma, axis=1)
    regime_labels = pd.Series(state_path, index=dates).map(
        {trending_state: "Trending", high_vol_state: "High-Volatility"}
    )
    chart_df = pd.DataFrame(
        {
            "close": series.reindex(dates),
            "log_return_percent": log_returns * 100,
            "trending_probability_percent": gamma[:, trending_state] * 100,
        }
    )
    tail = chart_df.tail(CHART_LOOKBACK_DAYS)
    regime_tail = regime_labels.tail(CHART_LOOKBACK_DAYS)
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "log_return_percent": _round(row["log_return_percent"], 3),
            "trending_probability_percent": _round(
                row["trending_probability_percent"], 2
            ),
            "regime": regime_tail.loc[ts],
        }
        for ts, row in tail.iterrows()
    ]

    return {
        "price": _round(float(series.iloc[-1])),
        "current_regime": current_regime,
        "recommended_approach": recommended_approach,
        "trending_probability_percent": _round(
            float(current_probs[trending_state]) * 100, 2
        ),
        "high_volatility_probability_percent": _round(
            float(current_probs[high_vol_state]) * 100, 2
        ),
        "regime_persistence_probability_percent": _round(
            float(transition[current_state, current_state]) * 100, 2
        ),
        "regime_changed_today": regime_changed_today,
        "trending_state_mean_return_percent": _round(
            float(means[trending_state]) * 100, 4
        ),
        "trending_state_volatility_percent": _round(
            float(np.sqrt(variances[trending_state])) * 100, 4
        ),
        "high_vol_state_mean_return_percent": _round(
            float(means[high_vol_state]) * 100, 4
        ),
        "high_vol_state_volatility_percent": _round(
            float(np.sqrt(variances[high_vol_state])) * 100, 4
        ),
        "lookback_days": len(log_returns),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 16: Squeeze Momentum Indicator
#
# Source article: "I Found a New Indicator on Substack — Then Backtested
# It on 50 Random Stocks". Combines Bollinger Bands and Keltner Channels
# to detect a volatility "squeeze" (Bollinger Bands compress inside the
# Keltner Channel = low-volatility coiling) and its "release" (Bollinger
# Bands expand back outside the Keltner Channel), plus a rolling
# linear-regression momentum oscillator.
#
# IMPORTANT: `momentum` below is the LAST FITTED VALUE of the rolling
# linear regression line over the window (matching Pine's
# ta.linreg(source, length, 0)) — i.e. intercept + slope * (length - 1)
# — NOT the regression slope by itself. See _linreg_last_value.
#
# Descriptive/analytical only: the article's own basic strategy logic
# (squeeze_off => entry, squeeze_on => exit) is surfaced here purely as
# `squeeze_state` / `momentum_direction` / `momentum_state` fields to
# read, never as an executed or recommended trade — same convention as
# every other strategy in this module.
# ---------------------------------------------------------------------------

SMI_LENGTH = 20
SMI_MULT_BB = 2.0
SMI_MULT_KC = 1.5
MIN_ROWS_SQUEEZE_MOMENTUM = SMI_LENGTH + 20


def _true_range(frame: pd.DataFrame) -> pd.Series:
    """Wilder's True Range from an OHLC frame (see to_ohlc_frame)."""
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            (frame["high"] - frame["low"]).abs(),
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _linreg_last_value(series: pd.Series, length: int) -> pd.Series:
    """Rolling linear-regression LAST FITTED VALUE (matches Pine's
    ta.linreg(source, length, 0)): for each trailing window of `length`
    points, fit y = a + b*x over x = 0..length-1 (oldest to newest) via
    ordinary least squares and return a + b*(length-1) — the fitted
    line's value at the most recent point in the window — NOT the slope
    b alone. The source article specifically warns this is an easy
    implementation mistake.
    """
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _fit(window: np.ndarray) -> float:
        y_mean = window.mean()
        slope = ((x - x_mean) * (window - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        return intercept + slope * x[-1]

    return series.rolling(window=length).apply(_fit, raw=True)


def compute_squeeze_momentum(prices: list[DailyPrice]) -> dict | None:
    """Squeeze Momentum Indicator strategy.

    Returns None when there isn't enough OHLC history for a stable
    Bollinger/Keltner/linear-regression reading (see
    MIN_ROWS_SQUEEZE_MOMENTUM) — same "unavailable" convention as every
    other strategy in this module. Uses the existing OHLC price
    infrastructure (to_ohlc_frame) — no new data source.
    """
    frame = to_ohlc_frame(prices)
    if len(frame) < MIN_ROWS_SQUEEZE_MOMENTUM:
        return None

    close = frame["close"]
    basis = close.rolling(SMI_LENGTH).mean()
    dev = SMI_MULT_BB * close.rolling(SMI_LENGTH).std(ddof=0)
    upper_bb = basis + dev
    lower_bb = basis - dev

    kc_ma = close.rolling(SMI_LENGTH).mean()
    true_range = _true_range(frame)
    range_ma = true_range.rolling(SMI_LENGTH).mean()
    upper_kc = kc_ma + range_ma * SMI_MULT_KC
    lower_kc = kc_ma - range_ma * SMI_MULT_KC

    squeeze_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    squeeze_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

    highest_high = frame["high"].rolling(SMI_LENGTH).max()
    lowest_low = frame["low"].rolling(SMI_LENGTH).min()
    midline = ((highest_high + lowest_low) / 2 + kc_ma) / 2
    momentum_source = close - midline
    momentum = _linreg_last_value(momentum_source, SMI_LENGTH)

    df = pd.DataFrame(
        {
            "close": close,
            "upper_bb": upper_bb,
            "lower_bb": lower_bb,
            "upper_kc": upper_kc,
            "lower_kc": lower_kc,
            "squeeze_on": squeeze_on,
            "squeeze_off": squeeze_off,
            "momentum": momentum,
        }
    ).dropna(subset=["upper_bb", "lower_bb", "upper_kc", "lower_kc", "momentum"])
    if df.empty:
        return None

    df["squeeze_state"] = np.where(
        df["squeeze_on"], "on", np.where(df["squeeze_off"], "off", "none")
    )

    last = df.iloc[-1]
    momentum_val = float(last["momentum"])
    squeeze_on_val = bool(last["squeeze_on"])
    squeeze_off_val = bool(last["squeeze_off"])
    squeeze_state = (
        "Squeeze On"
        if squeeze_on_val
        else ("Squeeze Released" if squeeze_off_val else "No Squeeze")
    )

    momentum_increasing = False
    momentum_direction = "Flat"
    squeeze_released_today = False
    if len(df) > 1:
        prev = df.iloc[-2]
        prev_momentum = float(prev["momentum"])
        momentum_increasing = momentum_val > prev_momentum
        momentum_direction = (
            "Rising"
            if momentum_increasing
            else ("Falling" if momentum_val < prev_momentum else "Flat")
        )
        squeeze_released_today = bool(prev["squeeze_on"]) and squeeze_off_val

    if momentum_val > 0:
        momentum_state = (
            "Bullish Momentum (Strengthening)"
            if momentum_increasing
            else "Bullish Momentum (Weakening)"
        )
    else:
        momentum_state = (
            "Bearish Momentum (Weakening)"
            if momentum_increasing
            else "Bearish Momentum (Strengthening)"
        )

    tail = df.tail(CHART_LOOKBACK_DAYS)
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "upper_bb": _round(row["upper_bb"]),
            "lower_bb": _round(row["lower_bb"]),
            "upper_kc": _round(row["upper_kc"]),
            "lower_kc": _round(row["lower_kc"]),
            "momentum": _round(row["momentum"]),
            "squeeze_state": row["squeeze_state"],
        }
        for ts, row in tail.iterrows()
    ]

    return {
        "price": _round(float(last["close"])),
        "upper_bb": _round(float(last["upper_bb"])),
        "lower_bb": _round(float(last["lower_bb"])),
        "upper_kc": _round(float(last["upper_kc"])),
        "lower_kc": _round(float(last["lower_kc"])),
        "momentum": _round(momentum_val),
        "momentum_direction": momentum_direction,
        "momentum_state": momentum_state,
        "squeeze_state": squeeze_state,
        "squeeze_on": squeeze_on_val,
        "squeeze_off": squeeze_off_val,
        "squeeze_released_today": squeeze_released_today,
        "bb_length": SMI_LENGTH,
        "bb_mult": SMI_MULT_BB,
        "kc_mult": SMI_MULT_KC,
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 17: Self-Organized Criticality — Avalanche Distribution
#
# Econophysics-style read: treats drawdown episodes ("avalanches" — a
# cascade of losses from a local price peak down to the next new peak)
# the way Self-Organized-Criticality models (e.g. the Bak-Tang-Wiesenfeld
# sandpile) treat avalanche sizes, and asks whether their size
# distribution follows a power law — a signature of a system operating
# near a critical point.
#
# METHODOLOGY NOTE — the source article describes the concept (SOC,
# power-law avalanche sizes, an alpha exponent, logarithmic binning, and
# the four alpha-range regimes below) but does not specify exactly how
# an "avalanche" is delimited in price data. Rather than inventing a
# hidden methodology, this implementation documents its own explicit,
# transparent definition:
#
#   1. Track the running all-time-high closing price (cummax) — no
#      look-ahead, uses only past/current data.
#   2. An "avalanche" is every maximal run of consecutive trading days
#      where the close stays below that running peak.
#   3. Its "size" is the run's maximum peak-to-trough percentage
#      drawdown.
#
# alpha is then estimated via a log-log linear least-squares fit of a
# logarithmically-binned histogram of avalanche sizes ("logarithmic
# binning" per the article), implemented with numpy only (no scipy
# dependency in this project).
#
# This does NOT claim to predict crashes — it is a descriptive read of
# how heavy-tailed the current drawdown-size distribution is.
# ---------------------------------------------------------------------------

SOC_MIN_TRADING_DAYS = 300
SOC_MIN_AVALANCHES = 15  # need enough events for a meaningful alpha fit
SOC_LOG_BINS = 12


def _extract_avalanches(series: pd.Series) -> pd.Series:
    """Percentage peak-to-trough drawdown size of every maximal run of
    consecutive days spent below the running all-time-high close so
    far. Only past/current prices are used (running max via .cummax()),
    so this carries no look-ahead bias.
    """
    running_peak = series.cummax()
    drawdown_pct = (series - running_peak) / running_peak * 100  # <= 0

    in_drawdown = drawdown_pct < 0
    run_id = (in_drawdown != in_drawdown.shift(1)).cumsum()
    sizes = (
        drawdown_pct[in_drawdown]
        .groupby(run_id[in_drawdown])
        .min()  # most negative = largest drawdown reached in the run
        .abs()
    )
    return sizes  # positive magnitudes, percent


def _fit_power_law_alpha(sizes: pd.Series) -> dict | None:
    """Log-log linear regression on a logarithmically-binned histogram
    of avalanche sizes -> alpha exponent. Returns None if there isn't
    enough distinct, positive-sized data for a meaningful fit — never
    fabricates alpha from an unreliable sample.
    """
    sizes = sizes[sizes > 0]
    if len(sizes) < SOC_MIN_AVALANCHES:
        return None

    log_min, log_max = math.log10(float(sizes.min())), math.log10(float(sizes.max()))
    if not math.isfinite(log_min) or not math.isfinite(log_max) or log_max <= log_min:
        return None

    bin_edges = np.logspace(log_min, log_max, SOC_LOG_BINS + 1)
    counts, edges = np.histogram(sizes.to_numpy(), bins=bin_edges)
    bin_widths = np.diff(edges)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])  # geometric bin center

    with np.errstate(divide="ignore", invalid="ignore"):
        density = counts / bin_widths
    mask = counts > 0
    if mask.sum() < 3:  # need at least 3 populated bins for a stable fit
        return None

    log_x = np.log10(bin_centers[mask])
    log_y = np.log10(density[mask])
    slope, intercept = np.polyfit(log_x, log_y, 1)
    alpha = -slope  # P(x) ~ x^-alpha  =>  log P = -alpha * log x + const

    fitted_log_y = slope * log_x + intercept
    residual = log_y - fitted_log_y
    ss_res = float((residual**2).sum())
    ss_tot = float(((log_y - log_y.mean()) ** 2).sum())
    r_squared = (1 - ss_res / ss_tot) if ss_tot > 0 else None

    return {
        "alpha": float(alpha),
        "r_squared": r_squared,
        "bin_centers": bin_centers[mask].tolist(),
        "bin_density": density[mask].tolist(),
        "sample_size": int(len(sizes)),
    }


def _soc_classification(alpha: float) -> str:
    """Regime classification exactly per the article's alpha ranges."""
    if alpha >= 2.8:
        return "Gaussian"
    if alpha >= 1.8:
        return "Transitional"
    if alpha >= 1.0:
        return "Critical"
    return "Super-critical"


def compute_soc_avalanche(prices: list[DailyPrice]) -> dict | None:
    """Self-Organized Criticality — Avalanche Distribution strategy.

    Returns None when there isn't enough price history, or too few
    distinct drawdown "avalanches", for a meaningful power-law fit (see
    SOC_MIN_TRADING_DAYS / SOC_MIN_AVALANCHES). Never fabricates alpha
    from an unreliable sample; this is a deterministic, pure function of
    `prices` like every other strategy in this module.
    """
    series = to_close_series(prices)
    if len(series) < SOC_MIN_TRADING_DAYS:
        return None

    sizes = _extract_avalanches(series)
    fit = _fit_power_law_alpha(sizes)
    if fit is None:
        return None

    alpha = fit["alpha"]
    regime = _soc_classification(alpha)

    running_peak = series.cummax()
    current_drawdown_pct = float(
        (series.iloc[-1] - running_peak.iloc[-1]) / running_peak.iloc[-1] * 100
    )

    chart_data = [
        {"size": _round(c, 4), "density": _round(d, 6)}
        for c, d in zip(fit["bin_centers"], fit["bin_density"])
    ]
    avalanche_tail = sizes.tail(CHART_LOOKBACK_DAYS)
    avalanche_history = [
        {"avalanche_index": int(i), "size_percent": _round(float(v))}
        for i, v in enumerate(avalanche_tail.tolist())
    ]

    return {
        "price": _round(float(series.iloc[-1])),
        "alpha": _round(alpha, 3),
        "criticality_regime": regime,
        "r_squared": (
            _round(fit["r_squared"], 3) if fit["r_squared"] is not None else None
        ),
        "avalanche_count": fit["sample_size"],
        "mean_avalanche_size_percent": _round(float(sizes.mean())),
        "max_avalanche_size_percent": _round(float(sizes.max())),
        "current_drawdown_percent": _round(current_drawdown_pct),
        "log_bins_used": SOC_LOG_BINS,
        "chart_data": chart_data,
        "avalanche_history": avalanche_history,
        "methodology_note": (
            "Avalanches are runs of consecutive trading days below the "
            "running all-time-high close; alpha is fit via a log-log "
            "linear regression on a logarithmically-binned histogram of "
            "avalanche sizes. This is a documented, transparent "
            "approximation (see module docstring above), not a "
            "reproduction of an undisclosed methodology, and it does "
            "not predict crashes."
        ),
    }


# ---------------------------------------------------------------------------
# Strategy 18: Adaptive Causal Wavelet Trend Filter
#
# Causal (no look-ahead) approximation of a Mexican Hat / Ricker wavelet
# trend filter: psi(t) = (1 - t^2) * e^(-t^2 / 2). The textbook Ricker
# wavelet is symmetric around t=0 (it uses both past AND future data),
# which would introduce look-ahead bias in a live indicator. This
# implementation uses only the CAUSAL half of the kernel (t >= 0, i.e.
# "now" and the past), convolved with daily log returns via a rolling
# dot product — every value at day t is a function of day t and earlier
# only (see _causal_wavelet_filter).
#
# MULTI-RESOLUTION: the same causal kernel shape is evaluated at three
# time scales (short/medium/long trading-day windows), each acting as a
# band-pass-ish filter that smooths noise while remaining responsive to
# the underlying trend.
#
# ADAPTIVE VOLATILITY ADJUSTMENT: each scale's raw filtered value is
# divided by the trailing standard deviation of returns over that same
# window, so the trend reading is comparable across calm and turbulent
# periods rather than simply scaling with raw price volatility.
# ---------------------------------------------------------------------------

WAVELET_SCALES: dict[str, int] = {"short": 5, "medium": 10, "long": 20}
WAVELET_KERNEL_WIDTH_MULTIPLIER = 3  # kernel window length = scale * multiplier
MIN_ROWS_WAVELET = max(WAVELET_SCALES.values()) * WAVELET_KERNEL_WIDTH_MULTIPLIER + 30


def _causal_ricker_kernel(
    scale: int, width_multiplier: int = WAVELET_KERNEL_WIDTH_MULTIPLIER
) -> np.ndarray:
    """Causal half (t >= 0, "now" back into the past) of the Mexican
    Hat/Ricker wavelet, sampled at unit lags and scaled by `scale`.
    kernel[0] is the weight for today (lag 0), kernel[k] the weight for
    k trading days ago.
    """
    length = scale * width_multiplier
    t = np.arange(length, dtype=float) / scale
    return (1 - t**2) * np.exp(-(t**2) / 2)


def _causal_wavelet_filter(returns: pd.Series, scale: int) -> pd.Series:
    """Rolling causal convolution of `returns` with the scaled causal
    Ricker kernel — every output value depends only on the current and
    past `length` return observations, never future ones."""
    kernel = _causal_ricker_kernel(scale)
    length = len(kernel)

    def _apply(window: np.ndarray) -> float:
        # `window` arrives oldest -> newest (pandas rolling convention);
        # reverse it so index 0 aligns with "today" (lag 0), matching
        # kernel[0].
        return float(np.dot(window[::-1], kernel))

    return returns.rolling(window=length).apply(_apply, raw=True)


def compute_wavelet_trend_filter(prices: list[DailyPrice]) -> dict | None:
    """Adaptive Causal Wavelet Trend Filter strategy.

    Returns None when there isn't enough price history for the longest
    scale's kernel plus a volatility warm-up window (see
    MIN_ROWS_WAVELET). Never returns a zeroed-out result for
    insufficient data, and never uses future price data (see module
    note above — every filtered value is a rolling, backward-looking
    function of `prices` only).
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_WAVELET:
        return None

    log_returns = np.log(series / series.shift(1))

    trend_columns: dict[str, pd.Series] = {}
    for label, scale in WAVELET_SCALES.items():
        raw = _causal_wavelet_filter(log_returns, scale)
        window = scale * WAVELET_KERNEL_WIDTH_MULTIPLIER
        vol = log_returns.rolling(window=window).std()
        trend_columns[f"trend_{label}"] = raw / vol.replace(0, np.nan)

    df = pd.DataFrame({"close": series, **trend_columns}).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    trend_short = float(last["trend_short"])
    trend_medium = float(last["trend_medium"])
    trend_long = float(last["trend_long"])

    signs = [trend_short > 0, trend_medium > 0, trend_long > 0]
    if all(signs):
        current_trend_state = "Strong Uptrend"
    elif not any(signs):
        current_trend_state = "Strong Downtrend"
    elif trend_medium > 0:
        current_trend_state = "Uptrend (Mixed Confirmation)"
    elif trend_medium < 0:
        current_trend_state = "Downtrend (Mixed Confirmation)"
    else:
        current_trend_state = "Transitional / Mixed"

    medium_window = WAVELET_SCALES["medium"] * WAVELET_KERNEL_WIDTH_MULTIPLIER
    medium_vol = log_returns.rolling(window=medium_window).std().dropna()
    volatility_adjustment_percent = (
        _round(float(medium_vol.iloc[-1]) * 100, 3) if not medium_vol.empty else None
    )

    tail = df.tail(CHART_LOOKBACK_DAYS)
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "trend_short": _round(row["trend_short"], 4),
            "trend_medium": _round(row["trend_medium"], 4),
            "trend_long": _round(row["trend_long"], 4),
        }
        for ts, row in tail.iterrows()
    ]

    return {
        "price": _round(float(last["close"])),
        "current_trend_state": current_trend_state,
        "wavelet_trend_value": _round(trend_medium, 4),
        "trend_short": _round(trend_short, 4),
        "trend_medium": _round(trend_medium, 4),
        "trend_long": _round(trend_long, 4),
        "volatility_adjustment_percent": volatility_adjustment_percent,
        "scales_days": dict(WAVELET_SCALES),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 19: First Passage Time Distribution Analysis
#
# Estimates, under a Brownian-motion-with-drift approximation of daily
# log returns (mu, sigma fit from trailing history), the probability
# that cumulative log-returns reach a given upside or downside target
# within a fixed horizon, and the expected time to do so. This uses the
# standard closed-form first-passage-time (inverse-Gaussian) result for
# drifted Brownian motion, implemented directly with numpy/math — no
# scipy dependency.
#
# ASSUMPTIONS (documented, not hidden):
#   - Daily log returns are treated as i.i.d. Normal(mu, sigma^2) — the
#     textbook Brownian-motion-with-drift approximation. Real markets
#     have fatter tails than this (see the EVaR strategy above, which
#     explicitly models tail-fatness); this strategy intentionally uses
#     the simpler, analytically tractable Normal/Brownian assumption
#     because the closed-form first-passage-time formulas rely on it.
#   - Default target levels (+/-10%) and horizon (252 trading days,
#     ~1 year) are documented defaults for this analytics surface, not
#     values taken from the source article (which discusses the concept
#     without prescribing universal defaults) and not backtested/
#     optimized figures.
#   - Probabilities and expected times are outputs of the model given
#     its stated assumptions — not guarantees about actual future price
#     paths.
# ---------------------------------------------------------------------------

FPT_LOOKBACK_DAYS = 252
MIN_ROWS_FIRST_PASSAGE_TIME = FPT_LOOKBACK_DAYS // 2 + 30
FPT_MIN_RETURN_OBSERVATIONS = 30
FPT_UPSIDE_TARGET_PERCENT = 10.0
FPT_DOWNSIDE_TARGET_PERCENT = -10.0
FPT_HORIZON_DAYS = 252


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF, implemented via math.erf (no scipy
    dependency in this project)."""
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def _first_passage_cdf(
    mu: float, sigma: float, level: float, horizon_days: int
) -> np.ndarray | None:
    """P(a Brownian motion X_t = mu*t + sigma*W_t has reached `level` > 0
    by time t), evaluated for every t = 1..horizon_days — the standard
    closed-form inverse-Gaussian first-passage-time CDF. Returns None
    for degenerate inputs (sigma <= 0, level <= 0, horizon <= 0).
    """
    if sigma <= 0 or level <= 0 or horizon_days <= 0:
        return None
    t = np.arange(1, horizon_days + 1, dtype=float)
    sqrt_t = np.sqrt(t)
    term1 = _norm_cdf((mu * t - level) / (sigma * sqrt_t))
    # exp(2*mu*level/sigma^2) can overflow for a large level/mu ratio;
    # clip the exponent defensively rather than silently producing inf.
    exponent = np.clip(2 * mu * level / (sigma**2), -700, 700)
    term2 = np.exp(exponent) * _norm_cdf((-mu * t - level) / (sigma * sqrt_t))
    return term1 + term2


def compute_first_passage_time(prices: list[DailyPrice]) -> dict | None:
    """First Passage Time Distribution Analysis strategy.

    Returns None when there isn't enough price history for a stable
    mu/sigma estimate of daily log returns (see
    MIN_ROWS_FIRST_PASSAGE_TIME) — same "unavailable" convention as
    every other strategy in this module.
    """
    series = to_close_series(prices)
    if len(series) < MIN_ROWS_FIRST_PASSAGE_TIME:
        return None

    log_returns = np.log(series / series.shift(1)).dropna().tail(FPT_LOOKBACK_DAYS)
    if len(log_returns) < FPT_MIN_RETURN_OBSERVATIONS:
        return None

    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    if sigma <= 0 or not math.isfinite(mu) or not math.isfinite(sigma):
        return None

    price = float(series.iloc[-1])
    upside_level = math.log(1 + FPT_UPSIDE_TARGET_PERCENT / 100)
    downside_level = -math.log(1 + FPT_DOWNSIDE_TARGET_PERCENT / 100)  # positive

    # Downside case computed by symmetry: reaching -downside_level under
    # drift mu is the same as the mirrored process (negated) reaching
    # +downside_level under drift -mu.
    upside_cdf = _first_passage_cdf(mu, sigma, upside_level, FPT_HORIZON_DAYS)
    downside_cdf = _first_passage_cdf(-mu, sigma, downside_level, FPT_HORIZON_DAYS)

    upside_probability_percent = (
        _round(float(upside_cdf[-1]) * 100) if upside_cdf is not None else None
    )
    downside_probability_percent = (
        _round(float(downside_cdf[-1]) * 100) if downside_cdf is not None else None
    )

    # E[first passage time] = level / mu, and is only finite when the
    # drift actually points toward that barrier; otherwise (in a pure
    # Brownian model) expected time to reach it is infinite, so this is
    # left null rather than fabricated — see assumptions_note below.
    expected_days_upside = _round(upside_level / mu, 1) if mu > 0 else None
    expected_days_downside = _round(downside_level / -mu, 1) if mu < 0 else None

    upside_target_price = _round(price * math.exp(upside_level))
    downside_target_price = _round(price * math.exp(-downside_level))

    last_date = series.index[-1]
    future_dates = pd.bdate_range(start=last_date, periods=FPT_HORIZON_DAYS + 1)[1:]
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "trading_day": i + 1,
            "upside_hit_probability_percent": (
                _round(float(upside_cdf[i]) * 100) if upside_cdf is not None else None
            ),
            "downside_hit_probability_percent": (
                _round(float(downside_cdf[i]) * 100)
                if downside_cdf is not None
                else None
            ),
        }
        for i, ts in enumerate(future_dates)
    ]

    return {
        "price": _round(price),
        "mean_daily_log_return_percent": _round(mu * 100, 4),
        "daily_volatility_percent": _round(sigma * 100, 4),
        "horizon_trading_days": FPT_HORIZON_DAYS,
        "upside_target_percent": FPT_UPSIDE_TARGET_PERCENT,
        "downside_target_percent": FPT_DOWNSIDE_TARGET_PERCENT,
        "upside_target_price": upside_target_price,
        "downside_target_price": downside_target_price,
        "upside_target_probability_percent": upside_probability_percent,
        "downside_target_probability_percent": downside_probability_percent,
        "expected_days_to_upside_target": expected_days_upside,
        "expected_days_to_downside_target": expected_days_downside,
        "lookback_days": len(log_returns),
        "assumptions_note": (
            "Assumes daily log returns are i.i.d. Normal(mu, sigma) "
            "(Brownian motion with drift) fit from trailing history; "
            "expected time to a target is null when the estimated "
            "drift points away from that target (expected first-"
            "passage time in that direction is infinite under a pure "
            "Brownian model). Not a guarantee about actual future price "
            "paths."
        ),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Portfolio Optimization: Kelly Criterion + Mean-Variance Optimization
#
# Genuinely multi-asset (unlike every strategy above, which is a pure
# function of ONE symbol's own price history): given a basket of
# symbols, estimates annualized expected returns and a covariance matrix
# from daily log returns, then reports:
#   - the global minimum-volatility portfolio
#   - the maximum-Sharpe-ratio ("tangency") portfolio
#   - an efficient frontier (closed-form Markowitz Lagrangian solution
#     swept across a range of target returns)
#   - per-asset and portfolio-level Kelly Criterion sizing information
#
# Implemented with closed-form linear algebra (numpy) only — this
# project has no scipy.optimize dependency, so there is no general
# quadratic-programming solver here. The unconstrained Markowitz
# closed-form solution can produce negative (short) weights; this
# implementation floors negative weights at 0 and renormalizes to sum
# to 100% as a documented LONG-ONLY APPROXIMATION of the true
# box-constrained problem — see _long_only_projection.
#
# Treated as Portfolio Optimization (see routers/strategy.py's dedicated
# POST /portfolio/optimization endpoint), NOT as a single-symbol
# /etfs/{symbol}/strategies indicator, per the reorg spec.
#
# Descriptive/analytical only: NOT a guaranteed-optimal allocation, and
# NOT investment advice.
# ---------------------------------------------------------------------------

PORTFOLIO_OPT_TRADING_DAYS_PER_YEAR = 252
PORTFOLIO_OPT_MIN_ROWS = 60  # minimum overlapping daily observations
PORTFOLIO_OPT_MIN_ASSETS = 2
PORTFOLIO_OPT_MAX_ASSETS = 8
PORTFOLIO_OPT_FRONTIER_POINTS = 15
# Kelly fraction is capped purely for sane display — this is not a
# leverage recommendation, just keeps a near-zero-variance asset from
# producing an absurd raw f* = mu/sigma^2 value.
PORTFOLIO_OPT_KELLY_DISPLAY_CAP = 3.0


def _long_only_projection(weights: np.ndarray) -> np.ndarray:
    """Floor negative weights at 0 and renormalize to sum to 1.

    Documented simplification: this is NOT a true long-only
    mean-variance QP solve (which would re-optimize subject to the
    constraint) — just a floor-and-renormalize projection of the
    unconstrained closed-form solution (see module note above). If
    every weight is non-positive (degenerate), falls back to equal
    weighting rather than fabricating a direction.
    """
    floored = np.clip(weights, 0, None)
    total = floored.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return floored / total


def _portfolio_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free_rate: float
) -> tuple[float, float, float]:
    expected_return = float(weights @ mu)
    variance = float(weights @ cov @ weights)
    volatility = math.sqrt(max(variance, 0.0))
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 0 else 0.0
    return expected_return, volatility, sharpe


def _weights_payload(symbols: list[str], weights: np.ndarray) -> list[dict]:
    return [
        {"symbol": symbol, "weight_percent": _round(float(w) * 100)}
        for symbol, w in zip(symbols, weights)
    ]


def compute_portfolio_optimization(
    prices_by_symbol: dict[str, list[DailyPrice]],
    *,
    risk_free_rate_percent: float = 0.0,
) -> dict | None:
    """Kelly Criterion + Mean-Variance Optimization across a basket of
    symbols.

    `prices_by_symbol` values are each symbol's own daily price history
    (already loaded by the caller — see routers/strategy.py's dedicated
    portfolio-optimization endpoint). Returns None when fewer than
    PORTFOLIO_OPT_MIN_ASSETS symbols have enough overlapping trading-day
    history, or the resulting covariance matrix is degenerate.
    """
    series_by_symbol = {
        symbol: to_close_series(prices)
        for symbol, prices in prices_by_symbol.items()
        if prices
    }
    series_by_symbol = {k: v for k, v in series_by_symbol.items() if not v.empty}
    if len(series_by_symbol) < PORTFOLIO_OPT_MIN_ASSETS:
        return None

    # Inner join on date, same alignment convention as
    # portfolio_comparison.compute_vt_vs_vti_vxus — never interpolate or
    # fabricate a missing price.
    combined = pd.concat(series_by_symbol, axis=1, join="inner").dropna()
    if len(combined) < PORTFOLIO_OPT_MIN_ROWS:
        return None

    symbols = list(combined.columns)
    n = len(symbols)

    log_returns = np.log(combined / combined.shift(1)).dropna()
    if len(log_returns) < PORTFOLIO_OPT_MIN_ROWS - 1:
        return None

    mu_daily = log_returns.mean().to_numpy()
    cov_daily = log_returns.cov().to_numpy()

    mu = mu_daily * PORTFOLIO_OPT_TRADING_DAYS_PER_YEAR
    cov = cov_daily * PORTFOLIO_OPT_TRADING_DAYS_PER_YEAR

    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return None

    ones = np.ones(n)
    risk_free_rate = risk_free_rate_percent / 100

    denom_minvol = float(ones @ cov_inv @ ones)
    if denom_minvol == 0:
        return None
    w_minvol_raw = (cov_inv @ ones) / denom_minvol
    w_minvol = _long_only_projection(w_minvol_raw)

    excess_mu = mu - risk_free_rate
    denom_sharpe = float(ones @ cov_inv @ excess_mu)
    w_maxsharpe_raw = (
        (cov_inv @ excess_mu) / denom_sharpe if abs(denom_sharpe) >= 1e-12 else w_minvol_raw
    )
    w_maxsharpe = _long_only_projection(w_maxsharpe_raw)

    minvol_return, minvol_vol, minvol_sharpe = _portfolio_stats(
        w_minvol, mu, cov, risk_free_rate
    )
    maxsharpe_return, maxsharpe_vol, maxsharpe_sharpe = _portfolio_stats(
        w_maxsharpe, mu, cov, risk_free_rate
    )

    # Closed-form efficient frontier (classic two-fund/Lagrangian
    # Markowitz solution), swept across target returns spanning the
    # asset universe's own return range, then long-only projected (see
    # module note above — realized return after projection may differ
    # slightly from the target).
    A = float(ones @ cov_inv @ ones)
    B = float(ones @ cov_inv @ mu)
    C = float(mu @ cov_inv @ mu)
    D = A * C - B**2

    efficient_frontier = []
    if abs(D) > 1e-12:
        target_low, target_high = float(mu.min()), float(mu.max())
        if target_high > target_low:
            targets = np.linspace(
                target_low, target_high, PORTFOLIO_OPT_FRONTIER_POINTS
            )
            for target in targets:
                w_raw = (
                    (C - B * target) * (cov_inv @ ones)
                    + (A * target - B) * (cov_inv @ mu)
                ) / D
                w = _long_only_projection(w_raw)
                ret, vol, sharpe = _portfolio_stats(w, mu, cov, risk_free_rate)
                efficient_frontier.append(
                    {
                        "target_return_percent": _round(target * 100),
                        "expected_return_percent": _round(ret * 100),
                        "volatility_percent": _round(vol * 100),
                        "sharpe_ratio": _round(sharpe, 3),
                        "weights": _weights_payload(symbols, w),
                    }
                )

    # Kelly sizing: single-asset continuous Kelly fraction f* = mu/sigma^2
    # per symbol (descriptive only), plus the portfolio-level Kelly
    # direction — proportional to Sigma^-1 * excess_mu, the same
    # direction as the max-Sharpe/tangency portfolio (a standard Kelly/
    # Markowitz equivalence result under Normal returns) — shown at
    # full-Kelly and a conservative half-Kelly scaling.
    variances = np.diag(cov)
    kelly_sizing = []
    for i, symbol in enumerate(symbols):
        single_kelly = float(mu[i] / variances[i]) if variances[i] > 0 else 0.0
        single_kelly = float(
            np.clip(
                single_kelly,
                -PORTFOLIO_OPT_KELLY_DISPLAY_CAP,
                PORTFOLIO_OPT_KELLY_DISPLAY_CAP,
            )
        )
        kelly_sizing.append(
            {
                "symbol": symbol,
                "single_asset_kelly_fraction_percent": _round(single_kelly * 100),
                "portfolio_kelly_weight_percent": _round(
                    float(w_maxsharpe[i]) * 100
                ),
                "half_kelly_weight_percent": _round(float(w_maxsharpe[i]) * 50),
            }
        )

    return {
        "symbols": symbols,
        "lookback_trading_days": len(log_returns),
        "risk_free_rate_percent": risk_free_rate_percent,
        "min_volatility_portfolio": {
            "weights": _weights_payload(symbols, w_minvol),
            "expected_return_percent": _round(minvol_return * 100),
            "volatility_percent": _round(minvol_vol * 100),
            "sharpe_ratio": _round(minvol_sharpe, 3),
        },
        "max_sharpe_portfolio": {
            "weights": _weights_payload(symbols, w_maxsharpe),
            "expected_return_percent": _round(maxsharpe_return * 100),
            "volatility_percent": _round(maxsharpe_vol * 100),
            "sharpe_ratio": _round(maxsharpe_sharpe, 3),
        },
        "efficient_frontier": efficient_frontier,
        "kelly_sizing": kelly_sizing,
        "long_only_note": (
            "Weights are floored at 0% and renormalized to 100% as a "
            "long-only approximation of the unconstrained closed-form "
            "Markowitz solution (see _long_only_projection) — not a "
            "full box-constrained optimization solve."
        ),
        "disclaimer": (
            "Analytical output only, derived from historical price "
            "history — not a guaranteed-optimal allocation, not "
            "investment advice, and not a prediction of future returns."
        ),
    }


# ===========================================================================
# Strategies 20-26: additional Technical Strategies
#
#   20. TD Setup / TD 9                      (Tom DeMark, Sequential Part I)
#   21. Supertrend Confirmation              (Supertrend + two-candle filter)
#   22. UFX + Vervoort Heiken Ashi           (STAND-IN components - see below)
#   23. Trend Signals + Trend Tide           (proprietary parts NOT reproducible)
#   24. Pivot Levels                         (classic floor-trader pivots)
#   25. CPR (Central Pivot Range)
#   26. TA Candles / Trend Health            (ThrustWise's own transparent build)
#
# All seven belong to the existing "Technical Strategies" category and
# follow the same conventions as every strategy above: pure functions over
# `list[DailyPrice]`, `None` when history is insufficient, `chart_data`
# built from the last CHART_LOOKBACK_DAYS rows, and ANALYTICAL output only
# (state / condition flags - never an action field).
#
# Data basis: this codebase only ingests DAILY bars (app.models.DailyPrice).
# Every rule below is applied to daily candles, exactly like Momentum
# Reversal (Strategy 13) does; no intraday pipeline was introduced.
#
# EXACTNESS SUMMARY (also repeated in each schema docstring):
#   exact           - TD Setup / TD 9, Supertrend Confirmation rules,
#                     Pivot Levels, CPR formulas
#   partial/stand-in- UFX + Vervoort HA (UFX Pro and Vervoort HA Long Term are
#                     proprietary TradingView scripts; public-formula
#                     stand-ins are used and clearly labelled as such)
#   not reproducible- Trend Signals + Trend Tide (no formula or parameters in
#                     the source), TA Candles / MasterPiece (same)
# ===========================================================================

TD_COMPARISON_LAG = 4  # Close[t] vs Close[t-4]
TD_SETUP_LENGTH = 9
MIN_ROWS_TD_SETUP = TD_COMPARISON_LAG + TD_SETUP_LENGTH

# The source for Supertrend Confirmation does not state ATR period /
# multiplier. 10 / 3.0 are the conventional Supertrend defaults and are
# documented as an assumption (exposed in the response so nothing is hidden).
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
SUPERTREND_CONFIRMATION_CANDLES = 2
MIN_ROWS_SUPERTREND_CONFIRMATION = SUPERTREND_ATR_PERIOD + 20

# UFX + Vervoort HA: settings quoted by the source article.
UFX_ATR_PERIOD = 6
UFX_FACTOR = 2.0
VERVOORT_TIMER_PERIOD = 11  # quoted by the source; NOT mappable (see below)
VERVOORT_EMA_PERIOD = 44  # quoted by the source; NOT mappable (see below)
UFX_VERVOORT_RISK_REWARD = 1.5  # 1 : 1.5
MIN_ROWS_UFX_VERVOORT = 50

TREND_SIGNALS_RISK_REWARD = 1.5  # 1 : 1.5
MIN_ROWS_TREND_SIGNALS_TREND_TIDE = 20

MIN_ROWS_PIVOT_LEVELS = 5

CPR_WIDTH_LOOKBACK = 60
CPR_NARROW_PERCENTILE = 0.25  # mid-rank percentile of width <= this -> Narrow
CPR_WIDE_PERCENTILE = 0.75  # mid-rank percentile of width >= this -> Wide
CPR_WIDTH_EQUALITY_TOLERANCE = 1e-9
MIN_ROWS_CPR = CPR_WIDTH_LOOKBACK + 1

TA_EMA_FAST = 21
TA_EMA_SLOW = 50
TA_TREND_MIN_SEPARATION_PERCENT = 0.25  # |EMA21-EMA50| / EMA50, else "Mixed"
TA_ROC_PERIOD = 10
TA_ATR_PERIOD = 14
TA_STRENGTH_WINDOW = 10
TA_VOLATILITY_BASELINE_WINDOW = 50
TA_VOLATILITY_EXPANDING_RATIO = 1.25
TA_VOLATILITY_CONTRACTING_RATIO = 0.75
TA_ALIGNED_CANDLE_RATIO = 0.6
TA_WEAKENING_CANDLE_RATIO = 0.5
MIN_ROWS_TA_CANDLES = TA_EMA_SLOW + TA_ATR_PERIOD + 10


def _candle_colour(frame: pd.DataFrame) -> pd.Series:
    """'Green' (close > open), 'Red' (close < open) or 'Neutral' (doji or
    missing open). Missing data can therefore never confirm a setup."""
    return pd.Series(
        np.where(
            frame["close"] > frame["open"],
            "Green",
            np.where(frame["close"] < frame["open"], "Red", "Neutral"),
        ),
        index=frame.index,
    )


def _ohlc_clean(prices: list[DailyPrice]) -> pd.DataFrame:
    """OHLC frame with rows lacking high/low/close removed (they cannot be
    used by any ATR / pivot calculation). A missing `open` is KEPT as NaN -
    it simply yields a 'Neutral' candle (see _candle_colour)."""
    frame = to_ohlc_frame(prices)
    if frame.empty:
        return frame
    return frame.dropna(subset=["high", "low", "close"])


# ---------------------------------------------------------------------------
# Strategy 20: TD Setup / TD 9 (Tom DeMark Sequential - Part I only)
#
# BEARISH TD SETUP: Close[t] >  Close[t-4] for N consecutive candles.
# BULLISH TD SETUP: Close[t] <  Close[t-4] for N consecutive candles.
# A completed count of 9 marks a POTENTIAL exhaustion area (upside for the
# bearish setup, downside for the bullish setup) - it is NOT a buy/sell
# instruction. The TD Sequential COUNTDOWN phase is intentionally not
# implemented (it is not part of the supplied source).
#
# Interruption: any candle that fails the comparison (including equal
# closes or a missing close) resets that setup's count to 0.
# ---------------------------------------------------------------------------


def _consecutive_run_lengths(hits: np.ndarray) -> np.ndarray:
    runs = np.zeros(len(hits), dtype=int)
    run = 0
    for i, hit in enumerate(hits):
        run = run + 1 if hit else 0
        runs[i] = run
    return runs


def compute_td_setup(prices: list[DailyPrice]) -> dict | None:
    """TD Setup / TD 9. Returns None when there are fewer than
    MIN_ROWS_TD_SETUP closes or the latest close is missing."""
    close = to_close_series(prices)
    if len(close) < MIN_ROWS_TD_SETUP or pd.isna(close.iloc[-1]):
        return None

    reference = close.shift(TD_COMPARISON_LAG)
    # NaN compares False, so a missing close interrupts the count.
    bearish_runs = _consecutive_run_lengths((close > reference).to_numpy())
    bullish_runs = _consecutive_run_lengths((close < reference).to_numpy())

    bearish_run = int(bearish_runs[-1])
    bullish_run = int(bullish_runs[-1])
    if bearish_run > 0:
        setup_type, run = "Bearish", bearish_run
    elif bullish_run > 0:
        setup_type, run = "Bullish", bullish_run
    else:
        setup_type, run = "None", 0

    setup_completed = run >= TD_SETUP_LENGTH
    if setup_completed and setup_type == "Bearish":
        exhaustion_state = "Potential Upside Exhaustion"
    elif setup_completed and setup_type == "Bullish":
        exhaustion_state = "Potential Downside Exhaustion"
    elif run > 0:
        exhaustion_state = "Setup In Progress"
    else:
        exhaustion_state = "No Active Setup"

    # Most recent bar on which either setup first reached a count of 9.
    last_completed_type: str | None = None
    last_completed_date: str | None = None
    latest_idx = -1
    for label, runs in (("Bearish", bearish_runs), ("Bullish", bullish_runs)):
        hits = np.flatnonzero(runs == TD_SETUP_LENGTH)
        if len(hits) and hits[-1] > latest_idx:
            latest_idx = int(hits[-1])
            last_completed_type = label
            last_completed_date = close.index[latest_idx].date().isoformat()

    df = pd.DataFrame(
        {
            "close": close,
            "bearish_setup_count": np.minimum(bearish_runs, TD_SETUP_LENGTH),
            "bullish_setup_count": np.minimum(bullish_runs, TD_SETUP_LENGTH),
        }
    )
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "bearish_setup_count": int(row["bearish_setup_count"]),
            "bullish_setup_count": int(row["bullish_setup_count"]),
        }
        for ts, row in df.tail(CHART_LOOKBACK_DAYS).iterrows()
    ]

    return {
        "price": _round(float(close.iloc[-1])),
        "setup_type": setup_type,
        "setup_count": min(run, TD_SETUP_LENGTH),
        "setup_completed": setup_completed,
        "exhaustion_state": exhaustion_state,
        "consecutive_qualifying_candles": run,
        "bars_since_completion": (run - TD_SETUP_LENGTH) if setup_completed else None,
        "comparison_lag_candles": TD_COMPARISON_LAG,
        "setup_length": TD_SETUP_LENGTH,
        "last_completed_setup_type": last_completed_type,
        "last_completed_setup_date": last_completed_date,
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Supertrend (standard, publicly documented ATR-band algorithm)
#
# Reuses the module's existing Wilder `_atr`. Returns the Supertrend line
# plus direction (+1 bullish / -1 bearish) and the active final bands.
# ---------------------------------------------------------------------------


def _supertrend(
    frame: pd.DataFrame, period: int, multiplier: float
) -> pd.DataFrame:
    hl2 = (frame["high"] + frame["low"]) / 2
    atr = _atr(frame, period)
    basic_upper = (hl2 + multiplier * atr).to_numpy()
    basic_lower = (hl2 - multiplier * atr).to_numpy()
    close = frame["close"].to_numpy()

    n = len(frame)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    line = np.full(n, np.nan)

    start = next((i for i in range(n) if not np.isnan(basic_upper[i])), None)
    if start is not None:
        final_upper[start] = basic_upper[start]
        final_lower[start] = basic_lower[start]
        direction[start] = 1 if close[start] >= hl2.iloc[start] else -1
        line[start] = final_lower[start] if direction[start] == 1 else final_upper[start]
        for i in range(start + 1, n):
            final_upper[i] = (
                basic_upper[i]
                if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]
                else final_upper[i - 1]
            )
            final_lower[i] = (
                basic_lower[i]
                if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]
                else final_lower[i - 1]
            )
            if direction[i - 1] == -1 and close[i] > final_upper[i]:
                direction[i] = 1
            elif direction[i - 1] == 1 and close[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]
            line[i] = final_lower[i] if direction[i] == 1 else final_upper[i]

    out = pd.DataFrame(
        {
            "supertrend": line,
            "direction": direction.astype(float),
            "final_upper": final_upper,
            "final_lower": final_lower,
        },
        index=frame.index,
    )
    # Warm-up rows (no valid ATR yet) have no direction.
    out.loc[out["direction"] == 0, :] = np.nan
    return out


# ---------------------------------------------------------------------------
# Strategy 21: Supertrend Confirmation
#
# BUY  condition: the two candles immediately BEFORE the signal candle are
#                 GREEN, close > Supertrend, Supertrend is bullish (green).
# SELL condition: the two candles immediately BEFORE the signal candle are
#                 RED,   close < Supertrend, Supertrend is bearish (red).
# If either preceding candle is the wrong colour (or a doji / missing open)
# the condition is NOT met - a bare Supertrend direction change is never
# reported as a condition on its own. `supertrend_flipped_today` is exposed
# separately so a consumer can see whether the trend just changed.
# Candle colour: green = close > open, red = close < open.
# ---------------------------------------------------------------------------


def compute_supertrend_confirmation(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_SUPERTREND_CONFIRMATION:
        return None

    st = _supertrend(frame, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
    colour = _candle_colour(frame)
    green = colour == "Green"
    red = colour == "Red"
    n = SUPERTREND_CONFIRMATION_CANDLES
    prev_green = pd.Series(True, index=frame.index)
    prev_red = pd.Series(True, index=frame.index)
    for k in range(1, n + 1):
        # fill_value=False: rows without k preceding candles never confirm.
        prev_green &= green.shift(k, fill_value=False)
        prev_red &= red.shift(k, fill_value=False)

    close = frame["close"]
    buy = prev_green & (close > st["supertrend"]) & (st["direction"] == 1)
    sell = prev_red & (close < st["supertrend"]) & (st["direction"] == -1)

    df = pd.DataFrame(
        {
            "close": close,
            "supertrend": st["supertrend"],
            "direction": st["direction"],
            "buy": buy,
            "sell": sell,
        }
    ).dropna(subset=["supertrend", "direction"])
    if df.empty:
        return None
    df["confirmation_state"] = np.where(
        df["buy"], "Buy Confirmed", np.where(df["sell"], "Sell Confirmed", "No Confirmation")
    )

    last = df.iloc[-1]
    buy_active, sell_active = bool(last["buy"]), bool(last["sell"])
    flipped = len(df) > 1 and int(df["direction"].iloc[-2]) != int(last["direction"])

    def _last_date(mask: pd.Series) -> str | None:
        hits = mask[mask]
        return hits.index[-1].date().isoformat() if len(hits) else None

    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "supertrend": _round(row["supertrend"]),
            "confirmation_state": row["confirmation_state"],
        }
        for ts, row in df.tail(CHART_LOOKBACK_DAYS).iterrows()
    ]

    return {
        "price": _round(float(last["close"])),
        "supertrend": _round(float(last["supertrend"])),
        "supertrend_trend": "Bullish" if last["direction"] == 1 else "Bearish",
        "price_above_supertrend": bool(last["close"] > last["supertrend"]),
        "supertrend_flipped_today": bool(flipped),
        "previous_two_candles": [
            colour.iloc[-1 - k] if len(colour) > k else "Neutral" for k in (2, 1)
        ],
        "buy_condition_active": buy_active,
        "sell_condition_active": sell_active,
        "confirmation_state": str(last["confirmation_state"]),
        "last_buy_condition_date": _last_date(df["buy"]),
        "last_sell_condition_date": _last_date(df["sell"]),
        "confirmation_candles": n,
        "atr_period": SUPERTREND_ATR_PERIOD,
        "multiplier": SUPERTREND_MULTIPLIER,
        "parameters_note": (
            "The source does not state Supertrend parameters; the conventional "
            "ATR(10) x 3.0 defaults are used."
        ),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 22: UFX + Vervoort Heiken Ashi
#
# LIMITATION (important): "UFX Pro" and "Vervoort Heiken Ashi - Long Term"
# are proprietary TradingView scripts whose formulas are NOT in this
# repository and NOT in the source article (which only quotes settings).
# This module therefore does NOT claim to reproduce them. It evaluates the
# article's RULE STRUCTURE using clearly labelled public-formula stand-ins:
#
#   UFX Pro stand-in      -> standard Supertrend-style ATR band with the
#                            article's ATR Period = 6 and Factor = 2. A
#                            direction flip on the signal candle plays the
#                            role of the "UFX BUY / SELL".
#   Vervoort HA stand-in  -> colour of the STANDARD Heikin-Ashi candle
#                            (HA close = (O+H+L+C)/4, HA open =
#                            (prev HA open + prev HA close)/2). The
#                            article's Timer Period = 11 and EMA Period = 44
#                            cannot be mapped to a documented formula and
#                            are therefore NOT used (listed in the response
#                            as `unmapped_source_settings`).
#
# BUY  setup: stand-in flips bullish on the bar, HA colour green, bullish
#             candle (close > open).
# SELL setup: stand-in flips bearish on the bar, HA colour red, bearish
#             candle (close < open).
# Risk model from the article: reward = 1.5 x risk. The article says only
# "stop-loss at the relevant zone"; the stand-in ATR band at the setup bar
# is used as that zone (assumption, flagged in the response).
# ---------------------------------------------------------------------------


def _heikin_ashi(frame: pd.DataFrame) -> pd.DataFrame:
    """Standard Heikin-Ashi candles. Missing `open` falls back to the
    close for the first HA open only (the recursion then self-corrects)."""
    ha_close = (frame["open"].fillna(frame["close"]) + frame["high"] + frame["low"] + frame["close"]) / 4
    ha_open = np.empty(len(frame))
    first_open = frame["open"].iloc[0]
    first_open = frame["close"].iloc[0] if pd.isna(first_open) else first_open
    ha_open[0] = (first_open + frame["close"].iloc[0]) / 2
    closes = ha_close.to_numpy()
    for i in range(1, len(frame)):
        ha_open[i] = (ha_open[i - 1] + closes[i - 1]) / 2
    return pd.DataFrame({"ha_open": ha_open, "ha_close": closes}, index=frame.index)


def compute_ufx_vervoort_heiken_ashi(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_UFX_VERVOORT:
        return None

    band = _supertrend(frame, UFX_ATR_PERIOD, UFX_FACTOR)
    ha = _heikin_ashi(frame)
    colour = _candle_colour(frame)

    direction = band["direction"]
    flip_up = (direction == 1) & (direction.shift(1) == -1)
    flip_down = (direction == -1) & (direction.shift(1) == 1)
    ha_green = ha["ha_close"] > ha["ha_open"]
    ha_red = ha["ha_close"] < ha["ha_open"]

    buy = flip_up & ha_green & (colour == "Green")
    sell = flip_down & ha_red & (colour == "Red")

    df = pd.DataFrame(
        {
            "close": frame["close"],
            "ufx_band": band["supertrend"],
            "direction": direction,
            "buy": buy,
            "sell": sell,
        }
    ).dropna(subset=["ufx_band", "direction"])
    if df.empty:
        return None
    df["setup_state"] = np.where(
        df["buy"], "Buy Setup", np.where(df["sell"], "Sell Setup", "No Setup")
    )

    last_idx = df.index[-1]
    setup_mask = df["buy"] | df["sell"]
    last_setup_type = last_setup_date = None
    entry = stop = target = None
    bars_since = None
    if setup_mask.any():
        setup_ts = df.index[setup_mask][-1]
        is_buy = bool(df.loc[setup_ts, "buy"])
        last_setup_type = "Buy" if is_buy else "Sell"
        last_setup_date = setup_ts.date().isoformat()
        bars_since = int(len(df) - 1 - df.index.get_loc(setup_ts))
        entry_v = float(df.loc[setup_ts, "close"])
        stop_v = float(df.loc[setup_ts, "ufx_band"])
        risk = entry_v - stop_v if is_buy else stop_v - entry_v
        if risk > 0:
            entry, stop = entry_v, stop_v
            target = entry_v + UFX_VERVOORT_RISK_REWARD * risk if is_buy else entry_v - UFX_VERVOORT_RISK_REWARD * risk

    last = df.iloc[-1]
    ha_last_green = bool(ha_green.loc[last_idx])
    ha_last_red = bool(ha_red.loc[last_idx])
    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(row["close"]),
            "ufx_stand_in_band": _round(row["ufx_band"]),
            "setup_state": row["setup_state"],
        }
        for ts, row in df.tail(CHART_LOOKBACK_DAYS).iterrows()
    ]

    return {
        "price": _round(float(last["close"])),
        "setup_state": str(last["setup_state"]),
        "buy_setup_active": bool(last["buy"]),
        "sell_setup_active": bool(last["sell"]),
        "ufx_stand_in_trend": "Bullish" if last["direction"] == 1 else "Bearish",
        "ufx_stand_in_flipped_today": bool(flip_up.loc[last_idx] or flip_down.loc[last_idx]),
        "ufx_stand_in_band": _round(float(last["ufx_band"])),
        "heiken_ashi_histogram_stand_in": (
            "Green" if ha_last_green else ("Red" if ha_last_red else "Neutral")
        ),
        "candle_confirmation": {"Green": "Bullish", "Red": "Bearish"}.get(
            str(colour.loc[last_idx]), "Neutral"
        ),
        "last_setup_type": last_setup_type,
        "last_setup_date": last_setup_date,
        "bars_since_last_setup": bars_since,
        "last_setup_entry": _round(entry),
        "last_setup_stop": _round(stop),
        "last_setup_target": _round(target),
        "risk_reward_ratio": UFX_VERVOORT_RISK_REWARD,
        "ufx_atr_period": UFX_ATR_PERIOD,
        "ufx_factor": UFX_FACTOR,
        "ufx_pro_exact": False,
        "vervoort_heiken_ashi_exact": False,
        "unmapped_source_settings": [
            f"Vervoort Timer Period = {VERVOORT_TIMER_PERIOD}",
            f"Vervoort EMA Period = {VERVOORT_EMA_PERIOD}",
        ],
        "implementation_note": (
            "UFX Pro and Vervoort Heiken Ashi (Long Term) are proprietary "
            "TradingView scripts and are NOT reproduced. A Supertrend-style "
            "ATR(6) x 2 band and the standard Heiken Ashi candle colour are "
            "used as labelled stand-ins; Vervoort's Timer/EMA settings are "
            "not applied. The stop level is the stand-in band at the setup "
            "candle (the source only says 'the relevant zone')."
        ),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 23: Trend Signals + Trend Tide Oscillator
#
# LIMITATION (important): the source describes the roles (Trend Signals =
# decision maker, Trend Tide = green/red momentum filter) and a 1 : 1.5
# TP/SL ratio, but gives NO formula, length or parameter for either
# indicator, and neither exists in this repository. Inventing them would be
# fabrication, so this strategy reports ONLY what is reliably derivable:
# the price-confirmation candle and the (static) risk/reward framework.
# `trend_signals_available` / `trend_tide_available` are False and the
# BUY / SELL setup flags can therefore not become True until the exact
# indicator definitions are supplied.
# ---------------------------------------------------------------------------


def compute_trend_signals_trend_tide(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_TREND_SIGNALS_TREND_TIDE:
        return None

    colour = _candle_colour(frame)
    last_colour = str(colour.iloc[-1])
    candle_confirmation = {"Green": "Bullish", "Red": "Bearish"}.get(last_colour, "Neutral")

    chart_data = [
        {
            "date": ts.date().isoformat(),
            "close": _round(frame["close"].loc[ts]),
            "candle_colour": colour.loc[ts],
        }
        for ts in frame.tail(CHART_LOOKBACK_DAYS).index
    ]

    return {
        "price": _round(float(frame["close"].iloc[-1])),
        "candle_confirmation": candle_confirmation,
        "trend_signals_available": False,
        "trend_tide_available": False,
        "trend_signals_state": "Unavailable",
        "trend_tide_state": "Unavailable",
        "buy_setup_active": False,
        "sell_setup_active": False,
        "setup_state": "Unavailable - proprietary components not reproducible",
        "risk_reward_ratio": TREND_SIGNALS_RISK_REWARD,
        "implementation_note": (
            "The source article gives no formula or parameters for Trend "
            "Signals with TP & SL or the Trend Tide Oscillator, and neither "
            "exists in this codebase. They are not approximated. Only the "
            "price-confirmation candle and the 1 : 1.5 risk/reward framework "
            "are reported; setup flags stay False until the exact indicator "
            "definitions are supplied."
        ),
        "chart_data": chart_data,
    }


# ---------------------------------------------------------------------------
# Strategy 24: Pivot Levels (classic floor-trader pivots)
#
# Levels are computed from the PREVIOUS completed daily candle (H, L, C):
#   PP = (H + L + C) / 3          R1 = 2PP - L        S1 = 2PP - H
#   R2 = PP + (H - L)             S2 = PP - (H - L)
#   R3 = H + 2(PP - L)            S3 = L - 2(H - PP)
# Zones follow the source's three-way reading: price above R1 = upper
# zone (profit-booking / short-bias context), between S1 and R1 = middle
# zone (range / chop), below S1 = lower zone (accumulation / long-bias
# context). These are descriptive context labels only; no buy/sell rule
# is derived. The source does not give an exact level formula, so the
# classic floor-trader set is used (documented assumption).
# ---------------------------------------------------------------------------


def _pivot_level_frame(frame: pd.DataFrame) -> pd.DataFrame:
    high, low, close = frame["high"], frame["low"], frame["close"]
    pp = (high + low + close) / 3
    rng = high - low
    return pd.DataFrame(
        {
            "pivot": pp,
            "r1": 2 * pp - low,
            "r2": pp + rng,
            "r3": high + 2 * (pp - low),
            "s1": 2 * pp - high,
            "s2": pp - rng,
            "s3": low - 2 * (high - pp),
        },
        index=frame.index,
    )


def compute_pivot_levels(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_PIVOT_LEVELS:
        return None

    # Levels for bar t come from bar t-1 (no look-ahead).
    levels = _pivot_level_frame(frame).shift(1)
    df = levels.join(frame["close"]).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    ordered = {k: float(last[k]) for k in ("s3", "s2", "s1", "pivot", "r1", "r2", "r3")}

    if price > ordered["r1"]:
        zone, context = "Upper Zone", "Profit-booking / short-bias area"
    elif price < ordered["s1"]:
        zone, context = "Lower Zone", "Accumulation / long-bias area"
    else:
        zone, context = "Middle Zone", "Range / chop area - avoid overtrading"

    above = [(k, v) for k, v in ordered.items() if v > price]
    below = [(k, v) for k, v in ordered.items() if v <= price]
    nearest_res = min(above, key=lambda kv: kv[1]) if above else None
    nearest_sup = max(below, key=lambda kv: kv[1]) if below else None

    def _dist(level: tuple[str, float] | None) -> float | None:
        return _round(abs(level[1] - price) / price * 100) if level and price else None

    return {
        "price": _round(price),
        "pivot": _round(ordered["pivot"]),
        "r1": _round(ordered["r1"]),
        "r2": _round(ordered["r2"]),
        "r3": _round(ordered["r3"]),
        "s1": _round(ordered["s1"]),
        "s2": _round(ordered["s2"]),
        "s3": _round(ordered["s3"]),
        "price_above_pivot": price > ordered["pivot"],
        "zone": zone,
        "zone_context": context,
        "nearest_resistance_level": nearest_res[0].upper() if nearest_res else None,
        "nearest_resistance": _round(nearest_res[1]) if nearest_res else None,
        "distance_to_resistance_percent": _dist(nearest_res),
        "nearest_support_level": nearest_sup[0].upper() if nearest_sup else None,
        "nearest_support": _round(nearest_sup[1]) if nearest_sup else None,
        "distance_to_support_percent": _dist(nearest_sup),
        "basis": "Previous daily candle (H, L, C) - classic floor-trader pivots",
        "chart_data": _chart_rows(df, ["close", "pivot", "r1", "r2", "r3", "s1", "s2", "s3"]),
    }


# ---------------------------------------------------------------------------
# Strategy 25: CPR - Central Pivot Range
#
#   Pivot = (H + L + C) / 3      BC = (H + L) / 2      TC = 2 x Pivot - BC
#   top_cpr = max(TC, BC)        bottom_cpr = min(TC, BC)
#   width % = (top_cpr - bottom_cpr) / Pivot x 100
# computed from the PREVIOUS daily candle and applied to the current one.
#
# Narrow / wide: the source gives no numeric threshold, so the width is
# ranked against the trailing CPR_WIDTH_LOOKBACK widths of the same symbol
# using a MID-RANK percentile (share of widths below + half the share equal
# to it): <= 25th percentile = Narrow, >= 75th = Wide, otherwise Normal.
# Mid-rank keeps a typical width in a uniform history "Normal" (rank 0.5)
# while still flagging a genuine outlier - plain quantile cut-offs collapse
# when many widths are identical. Documented ThrustWise convention.
# Bias: close above top_cpr = bullish day bias, below bottom_cpr = bearish,
# inside = neutral. CPR edges are exposed as support / resistance. No
# buy/sell rule is derived.
# ---------------------------------------------------------------------------


def compute_cpr(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_CPR:
        return None

    high, low, close = frame["high"], frame["low"], frame["close"]
    pivot = (high + low + close) / 3
    bc = (high + low) / 2
    tc = 2 * pivot - bc
    top = pd.concat([tc, bc], axis=1).max(axis=1)
    bottom = pd.concat([tc, bc], axis=1).min(axis=1)
    width = top - bottom
    width_pct = width / pivot.replace(0, np.nan) * 100

    cpr = pd.DataFrame(
        {
            "pivot": pivot,
            "top_cpr": top,
            "bottom_cpr": bottom,
            "cpr_width": width,
            "cpr_width_percent": width_pct,
        },
        index=frame.index,
    )
    applied = cpr.shift(1)  # previous candle's CPR applies to this candle
    df = applied.join(close).dropna()
    if len(df) < 2:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    top_v, bottom_v = float(last["top_cpr"]), float(last["bottom_cpr"])
    width_v = float(last["cpr_width_percent"])

    window = df["cpr_width_percent"].tail(CPR_WIDTH_LOOKBACK)
    below = float((window < width_v - CPR_WIDTH_EQUALITY_TOLERANCE).sum())
    equal = float(((window - width_v).abs() <= CPR_WIDTH_EQUALITY_TOLERANCE).sum())
    width_rank = (below + 0.5 * equal) / len(window)
    if width_rank <= CPR_NARROW_PERCENTILE:
        width_class, interpretation = "Narrow", "Potentially larger move"
    elif width_rank >= CPR_WIDE_PERCENTILE:
        width_class, interpretation = "Wide", "Potentially range-bound market"
    else:
        width_class, interpretation = "Normal", "Neutral CPR width"

    if price > top_v:
        position, bias = "Above CPR", "Bullish Day Bias"
        support, resistance = top_v, None
    elif price < bottom_v:
        position, bias = "Below CPR", "Bearish Day Bias"
        support, resistance = None, bottom_v
    else:
        position, bias = "Inside CPR", "Neutral"
        support, resistance = bottom_v, top_v

    return {
        "price": _round(price),
        "pivot": _round(float(last["pivot"])),
        "top_cpr": _round(top_v),
        "bottom_cpr": _round(bottom_v),
        "cpr_width": _round(float(last["cpr_width"]), 4),
        "cpr_width_percent": _round(width_v, 4),
        "cpr_width_percentile_rank": _round(width_rank * 100),
        "cpr_width_class": width_class,
        "cpr_width_interpretation": interpretation,
        "price_position": position,
        "market_bias": bias,
        "cpr_support": _round(support),
        "cpr_resistance": _round(resistance),
        "width_lookback_days": CPR_WIDTH_LOOKBACK,
        "basis": "Previous daily candle (H, L, C)",
        "chart_data": _chart_rows(df, ["close", "pivot", "top_cpr", "bottom_cpr"]),
    }


# ---------------------------------------------------------------------------
# Strategy 26: TA Candles / Trend Health
#
# LIMITATION (important): the source describes TA Candles / "MasterPiece"
# qualitatively (trend + volatility + momentum + strength consistency) but
# gives no formula, and it is a proprietary TradingView script. It is NOT
# reproduced and this output must never be read as that indicator
# (`is_exact_tradingview_indicator` is False).
#
# What IS provided is ThrustWise's own transparent trend-health read built
# from standard, documented measures covering the same four dimensions:
#   trend       : EMA21 vs EMA50 (min. 0.25% separation) and close vs EMA21
#   momentum    : 10-day rate of change
#   volatility  : ATR(14) as % of close vs its own 50-day median
#   consistency : share of green / red candles in the last 10 sessions
# All thresholds are constants at the top of this section so they are easy
# to audit or change.
# ---------------------------------------------------------------------------


def compute_ta_candles_trend_health(prices: list[DailyPrice]) -> dict | None:
    frame = _ohlc_clean(prices)
    if len(frame) < MIN_ROWS_TA_CANDLES:
        return None

    close = frame["close"]
    ema_fast = _ema(close, TA_EMA_FAST)
    ema_slow = _ema(close, TA_EMA_SLOW)
    roc = close.pct_change(TA_ROC_PERIOD) * 100
    atr_pct = _atr(frame, TA_ATR_PERIOD) / close.replace(0, np.nan) * 100
    atr_baseline = atr_pct.rolling(TA_VOLATILITY_BASELINE_WINDOW, min_periods=TA_ATR_PERIOD).median()
    colour = _candle_colour(frame)
    green_ratio = (colour == "Green").astype(float).rolling(TA_STRENGTH_WINDOW).mean()
    red_ratio = (colour == "Red").astype(float).rolling(TA_STRENGTH_WINDOW).mean()

    df = pd.DataFrame(
        {
            "close": close,
            "ema21": ema_fast,
            "ema50": ema_slow,
            "roc": roc,
            "atr_percent": atr_pct,
            "atr_baseline": atr_baseline,
            "green_ratio": green_ratio,
            "red_ratio": red_ratio,
        }
    ).dropna()
    if df.empty:
        return None

    last = df.iloc[-1]
    price = float(last["close"])
    e21, e50 = float(last["ema21"]), float(last["ema50"])
    roc_v = float(last["roc"])
    g_ratio, r_ratio = float(last["green_ratio"]), float(last["red_ratio"])

    # Require a minimum EMA separation: in a sideways market the sign of
    # EMA21 - EMA50 flips on noise and must not be read as a trend.
    separated = abs(e21 - e50) / e50 * 100 >= TA_TREND_MIN_SEPARATION_PERCENT if e50 else False
    if separated and e21 > e50 and price > e21:
        trend = "Bullish"
    elif separated and e21 < e50 and price < e21:
        trend = "Bearish"
    else:
        trend = "Mixed"

    momentum_state = "Positive" if roc_v > 0 else ("Negative" if roc_v < 0 else "Flat")

    base = float(last["atr_baseline"])
    atr_v = float(last["atr_percent"])
    if base > 0 and atr_v >= base * TA_VOLATILITY_EXPANDING_RATIO:
        volatility_state = "Expanding"
    elif base > 0 and atr_v <= base * TA_VOLATILITY_CONTRACTING_RATIO:
        volatility_state = "Contracting"
    else:
        volatility_state = "Normal"

    if trend == "Bullish" and momentum_state == "Positive" and g_ratio >= TA_ALIGNED_CANDLE_RATIO:
        alignment, health = "Aligned Bullish", "Healthy Trend (Bullish)"
    elif trend == "Bearish" and momentum_state == "Negative" and r_ratio >= TA_ALIGNED_CANDLE_RATIO:
        alignment, health = "Aligned Bearish", "Healthy Trend (Bearish)"
    elif trend == "Bullish" and (momentum_state != "Positive" or g_ratio < TA_WEAKENING_CANDLE_RATIO):
        alignment, health = "Mixed", "Weakening Bullish Trend"
    elif trend == "Bearish" and (momentum_state != "Negative" or r_ratio < TA_WEAKENING_CANDLE_RATIO):
        alignment, health = "Mixed", "Weakening Bearish Trend"
    else:
        alignment, health = "Mixed", "Choppy - Caution"

    streak_colour = str(colour.iloc[-1])
    streak = 0
    for c in reversed(colour.tolist()):
        if c == streak_colour and c != "Neutral":
            streak += 1
        else:
            break

    return {
        "price": _round(price),
        "ema21": _round(e21),
        "ema50": _round(e50),
        "trend": trend,
        "momentum_roc_percent": _round(roc_v),
        "momentum_state": momentum_state,
        "atr_percent": _round(atr_v),
        "volatility_state": volatility_state,
        "green_candle_ratio": _round(g_ratio, 2),
        "red_candle_ratio": _round(r_ratio, 2),
        "current_candle_streak": streak,
        "current_candle_streak_colour": streak_colour,
        "strength_alignment": alignment,
        "trend_health_state": health,
        "strength_window_candles": TA_STRENGTH_WINDOW,
        "is_exact_tradingview_indicator": False,
        "methodology_note": (
            "TA Candles / MasterPiece is a proprietary TradingView indicator "
            "whose formula is not given in the source and is NOT reproduced. "
            "This is ThrustWise's own trend-health read from standard "
            "measures (EMA21/EMA50 trend, 10-day ROC momentum, ATR(14)% "
            "volatility, 10-candle green/red consistency)."
        ),
        "chart_data": _chart_rows(df, ["close", "ema21", "ema50", "roc", "atr_percent", "green_ratio"]),
    }

# ---------------------------------------------------------------------------
# Strategies 27-28: fundamental-valuation QUANTITATIVE strategies
#
# 27. Benjamin Graham growth-valuation formula  (`graham_valuation`)
# 28. Joel Greenblatt's Magic Formula           (`greenblatt_magic_formula`)
#
# Unlike Strategies 1-26 these are NOT functions of an ETF's price history:
# they take already-extracted fundamental inputs (see
# app.scanners.data.extract_valuation_inputs and the
# `valuation_inputs_json` column of StockFundamentalsSnapshot) as plain
# arguments, so these functions stay pure (no DB/HTTP) and fully testable.
# Neither function ever fabricates a missing input: a missing / non-finite /
# invalid input yields `status == "unavailable"` plus explicit reasons.
# Descriptive / analytical only - not a buy/sell recommendation.
# ---------------------------------------------------------------------------

UNAVAILABLE = "unavailable"
AVAILABLE = "available"


def _is_num(value) -> bool:
    """True for a real, finite int/float (bools, NaN, +/-inf and None are not)."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# --- Strategy 27: Graham growth-valuation formula ---------------------------
#
#   Estimated Value = EPS x (8.5 + 2g) x (4.4 / Y)
#
#   EPS  earnings per share (selected basis/period is reported)
#   g    expected annual EPS growth, in PERCENT UNITS (8% -> 8, not 0.08)
#   Y    comparable AAA corporate bond yield, in PERCENT UNITS (5.1% -> 5.1)
#   8.5, 4.4  fixed constants of this (interest-rate-adjusted) formula.
#
# This is NOT the Graham Number (sqrt(22.5 x EPS x BVPS)), a different
# valuation. Upside/Downside % = (Estimated Value - Market Price) / Market
# Price x 100 (positive => estimate above market price).
GRAHAM_BASE_PE = 8.5
GRAHAM_GROWTH_MULTIPLIER = 2.0
GRAHAM_BASELINE_YIELD_PERCENT = 4.4
# Growth fed to the formula is capped (the formula is very sensitive to g and
# analyst long-term estimates for growth names can be extreme). The RAW growth
# is always reported alongside the applied value and a warning is added.
GRAHAM_MAX_GROWTH_PERCENT = 15.0
# Default margin of safety (percent below the estimate); configurable via
# Settings.graham_margin_of_safety_percent.
GRAHAM_DEFAULT_MARGIN_OF_SAFETY_PERCENT = 30.0
# Conservative growth scenarios: fraction of the applied growth. 0.0 is the
# "no growth" scenario. A scenario never RAISES growth above the applied
# value (for negative g it stays at g).
GRAHAM_SENSITIVITY_SCENARIOS = (
    ("No growth (g = 0)", 0.0),
    ("Half of applied growth", 0.5),
    ("75% of applied growth", 0.75),
)
GRAHAM_GROWTH_BASIS_FORWARD = "forward_analyst_consensus"
GRAHAM_GROWTH_BASIS_HISTORICAL = "historical_eps_cagr_proxy"
GRAHAM_DISCLAIMER = (
    "Approximate, assumption-driven estimate from a published formula. It is "
    "not a guaranteed intrinsic value, not a prediction of the stock's future "
    "price, and not investment advice. Results are highly sensitive to the "
    "growth and bond-yield inputs."
)


def _graham_value(eps: float, growth_percent: float, aaa_yield_percent: float) -> float | None:
    multiplier = GRAHAM_BASE_PE + GRAHAM_GROWTH_MULTIPLIER * growth_percent
    if multiplier <= 0 or aaa_yield_percent <= 0:
        return None
    return eps * multiplier * (GRAHAM_BASELINE_YIELD_PERCENT / aaa_yield_percent)


def _upside_percent(value: float | None, price: float | None) -> float | None:
    if value is None or not _is_num(price) or price <= 0:
        return None
    return (value - price) / price * 100.0


def compute_graham_valuation(
    *,
    eps,
    eps_period: str | None = None,
    eps_basis: str | None = None,
    growth_percent=None,
    growth_basis: str | None = None,
    growth_source: str | None = None,
    growth_horizon: str | None = None,
    aaa_yield_percent=None,
    aaa_yield_source: str | None = None,
    aaa_yield_as_of: str | None = None,
    price=None,
    price_as_of: str | None = None,
    margin_of_safety_percent=GRAHAM_DEFAULT_MARGIN_OF_SAFETY_PERCENT,
    max_growth_percent: float = GRAHAM_MAX_GROWTH_PERCENT,
    extra_warnings: list[str] | None = None,
) -> dict:
    """Graham growth-valuation estimate for ONE stock (always returns a dict).

    `growth_percent` and `aaa_yield_percent` are percent-unit numbers (8 =
    8%). Never raises on bad input: missing / non-finite / non-positive-EPS /
    non-positive-yield inputs produce `status == "unavailable"` with
    `unavailable_reasons` and no estimate. A missing price does NOT make the
    estimate unavailable - only the price-comparison fields are null.
    """
    warnings = list(extra_warnings or [])
    reasons: list[str] = []

    if not _is_num(eps):
        reasons.append("EPS is missing or not a finite number.")
    elif float(eps) <= 0:
        reasons.append(
            "EPS is zero or negative; the growth-valuation formula is not meaningful."
        )
    if not _is_num(growth_percent):
        reasons.append("Expected earnings growth is missing or not a finite number.")
    if not _is_num(aaa_yield_percent):
        reasons.append("AAA corporate bond yield is missing or not a finite number.")
    elif float(aaa_yield_percent) <= 0:
        reasons.append("AAA corporate bond yield must be greater than zero.")

    applied_growth: float | None = None
    if _is_num(growth_percent):
        raw = float(growth_percent)
        applied_growth = min(raw, float(max_growth_percent))
        if applied_growth < raw:
            warnings.append(
                f"Expected growth {raw:.2f}% exceeds the {max_growth_percent:g}% cap; "
                f"{applied_growth:g}% was used."
            )
        if growth_basis == GRAHAM_GROWTH_BASIS_HISTORICAL:
            warnings.append(
                "Growth is a HISTORICAL EPS CAGR proxy, not a forward analyst estimate."
            )
        if (
            not reasons
            and GRAHAM_BASE_PE + GRAHAM_GROWTH_MULTIPLIER * applied_growth <= 0
        ):
            reasons.append(
                "Growth is so negative that (8.5 + 2g) is not positive; no estimate."
            )

    mos_ok = _is_num(margin_of_safety_percent) and 0 <= float(margin_of_safety_percent) < 100
    if not mos_ok:
        warnings.append("Margin of safety percent is invalid; margin-of-safety fields omitted.")

    price_ok = _is_num(price) and float(price) > 0
    if not price_ok:
        warnings.append("Current market price is unavailable; no price comparison is shown.")

    out = {
        "status": UNAVAILABLE if reasons else AVAILABLE,
        "unavailable_reasons": reasons,
        "estimated_value": None,
        "eps": float(eps) if _is_num(eps) else None,
        "eps_period": eps_period,
        "eps_basis": eps_basis,
        "growth_percent_raw": float(growth_percent) if _is_num(growth_percent) else None,
        "growth_percent_applied": _round(applied_growth, 4) if applied_growth is not None else None,
        "growth_basis": growth_basis,
        "growth_source": growth_source,
        "growth_horizon": growth_horizon,
        "max_growth_percent": float(max_growth_percent),
        "aaa_yield_percent": float(aaa_yield_percent) if _is_num(aaa_yield_percent) else None,
        "aaa_yield_source": aaa_yield_source,
        "aaa_yield_as_of": aaa_yield_as_of,
        "baseline_yield_percent": GRAHAM_BASELINE_YIELD_PERCENT,
        "formula": "EPS x (8.5 + 2g) x (4.4 / Y)",
        "market_price": float(price) if price_ok else None,
        "market_price_as_of": price_as_of if price_ok else None,
        "difference_abs": None,
        "upside_downside_percent": None,
        "difference_definition": (
            "Upside/Downside % = (Estimated Value - Market Price) / Market Price x 100"
        ),
        "margin_of_safety_percent": float(margin_of_safety_percent) if mos_ok else None,
        "margin_of_safety_price": None,
        "price_within_margin_of_safety": None,
        "sensitivity": [],
        "data_quality_warnings": warnings,
        "disclaimer": GRAHAM_DISCLAIMER,
    }
    if reasons:
        return out

    value = _graham_value(float(eps), applied_growth, float(aaa_yield_percent))
    if value is None:  # defensive; guarded by the checks above
        out["status"] = UNAVAILABLE
        out["unavailable_reasons"] = ["Formula inputs produced no valid estimate."]
        return out

    out["estimated_value"] = _round(value)
    if price_ok:
        out["difference_abs"] = _round(value - float(price))
        out["upside_downside_percent"] = _round(_upside_percent(value, float(price)))
    if mos_ok:
        mos_price = value * (1 - float(margin_of_safety_percent) / 100.0)
        out["margin_of_safety_price"] = _round(mos_price)
        if price_ok:
            out["price_within_margin_of_safety"] = bool(float(price) <= mos_price)

    for label, factor in GRAHAM_SENSITIVITY_SCENARIOS:
        g = min(applied_growth, applied_growth * factor)
        scenario_value = _graham_value(float(eps), g, float(aaa_yield_percent))
        out["sensitivity"].append(
            {
                "scenario": label,
                "growth_percent": _round(g, 4),
                "estimated_value": _round(scenario_value),
                "upside_downside_percent": _round(
                    _upside_percent(scenario_value, float(price) if price_ok else None)
                ),
            }
        )
    return out


# --- Strategy 28: Greenblatt Magic Formula ----------------------------------
#
#   Earnings Yield    = EBIT / Enterprise Value
#   Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
#
# Definitions used (from yfinance statements, see app.scanners.data):
#   EBIT   latest annual "EBIT" (fallback "Operating Income", labelled).
#   EV     provider-reported enterpriseValue; else marketCap + totalDebt -
#          totalCash (never mixed silently - the source is reported per row).
#   NWC    (Current Assets - Cash & ST investments)
#          - (Current Liabilities - Current Debt)     [operating working capital]
#   NFA    Net PP&E (net fixed assets).
# Ranking: the two factors are ranked independently (highest first, standard
# competition ranking so equal values share a rank), the ranks are summed,
# and the list is ordered by combined rank ascending (lower is better), ties
# broken by earnings-yield rank, then return-on-capital rank, then symbol.
# Observations that are invalid (missing inputs, EV <= 0, invested capital
# <= 0) are NEVER ranked (no artificially favourable rank) and are listed in
# `excluded_by_reason`. Negative EBIT with a valid denominator is a valid
# (unfavourable) observation and is ranked normally.
MAGIC_FORMULA_EXCLUDED_SECTORS = frozenset({"financials", "financial services", "utilities"})
MAGIC_FORMULA_STALE_DAYS = 548  # ~18 months: statements older than this get a warning
MAGIC_FORMULA_RESPONSE_TOP_N = 25
MAGIC_FORMULA_DISCLAIMER = (
    "A relative ranking within the stated universe at one point in time. It is "
    "not an absolute measure of investment quality, not a guarantee of future "
    "returns, and not investment advice."
)
MAGIC_FORMULA_METHODOLOGY = [
    "Earnings Yield = EBIT / Enterprise Value.",
    "Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets).",
    (
        "Net Working Capital = (Current Assets - Cash & short-term investments) - "
        "(Current Liabilities - Current Debt); Net Fixed Assets = Net PP&E."
    ),
    (
        "Each factor is ranked highest-first (equal values share a rank); the two "
        "ranks are summed and results are sorted by combined rank ascending."
    ),
    (
        "Financials and utilities are excluded (Greenblatt's convention). Stocks "
        "with missing/invalid inputs are excluded from the ranking, never ranked."
    ),
    (
        "EBIT and balance-sheet figures are the latest ANNUAL statements (not "
        "trailing-twelve-month); data comes from Yahoo Finance via yfinance."
    ),
]


def _period_age_days(period: str | None, today: datetime.date) -> int | None:
    if not period:
        return None
    try:
        return (today - datetime.date.fromisoformat(str(period)[:10])).days
    except ValueError:
        return None


def compute_magic_formula_metrics(inp: dict, today: datetime.date | None = None) -> dict:
    """Earnings yield / return on capital for ONE stock from raw inputs.

    `inp` keys (all optional except symbol; missing => that factor is
    unavailable): symbol, company_name, sector, industry, ebit, ebit_period,
    ebit_label, reported_enterprise_value, market_cap, total_debt, total_cash,
    current_assets, cash, current_liabilities, current_debt, net_ppe,
    balance_sheet_period, financial_currency, price_currency.
    Returns a row with factor values (percent units), per-factor status,
    warnings, and null ranks (filled in by rank_magic_formula).
    """
    today = today or datetime.datetime.now(datetime.timezone.utc).date()
    warnings: list[str] = []
    g = inp.get

    ebit = float(g("ebit")) if _is_num(g("ebit")) else None
    if ebit is None:
        warnings.append("EBIT is missing.")

    # --- enterprise value (best available supported calculation) ---
    ev: float | None = None
    ev_source: str | None = None
    if _is_num(g("reported_enterprise_value")):
        ev = float(g("reported_enterprise_value"))
        ev_source = "Yahoo Finance enterpriseValue (provider-reported)"
    elif all(_is_num(g(k)) for k in ("market_cap", "total_debt", "total_cash")):
        ev = float(g("market_cap")) + float(g("total_debt")) - float(g("total_cash"))
        ev_source = "Computed: marketCap + totalDebt - totalCash (Yahoo Finance info)"
    else:
        warnings.append("Enterprise value is missing and cannot be derived from market cap, debt and cash.")

    earnings_yield: float | None = None
    ey_status, ey_reason = UNAVAILABLE, None
    if ebit is None:
        ey_reason = "EBIT missing"
    elif ev is None:
        ey_reason = "Enterprise value missing"
    elif ev <= 0:
        ey_reason = "Enterprise value is zero or negative"
        warnings.append("Enterprise value is zero or negative; earnings yield is not meaningful.")
    else:
        earnings_yield = ebit / ev * 100.0
        ey_status = AVAILABLE

    # --- invested capital ---
    nwc: float | None = None
    required = ("current_assets", "cash", "current_liabilities", "current_debt")
    missing_nwc = [k for k in required if not _is_num(g(k))]
    if missing_nwc:
        warnings.append("Net working capital unavailable; missing: " + ", ".join(missing_nwc) + ".")
    else:
        nwc = (float(g("current_assets")) - float(g("cash"))) - (
            float(g("current_liabilities")) - float(g("current_debt"))
        )
    nfa = float(g("net_ppe")) if _is_num(g("net_ppe")) else None
    if nfa is None:
        warnings.append("Net fixed assets (Net PP&E) are missing.")
    elif nfa < 0:
        warnings.append("Net PP&E is negative; treated as invalid.")
        nfa = None
    invested = (nwc + nfa) if (nwc is not None and nfa is not None) else None

    roc: float | None = None
    roc_status, roc_reason = UNAVAILABLE, None
    if ebit is None:
        roc_reason = "EBIT missing"
    elif invested is None:
        roc_reason = "Net working capital or net fixed assets missing"
    elif invested <= 0:
        roc_reason = "Invested capital (NWC + NFA) is zero or negative"
        warnings.append("Invested capital is zero or negative; return on capital is not meaningful.")
    else:
        roc = ebit / invested * 100.0
        roc_status = AVAILABLE

    fin_ccy, px_ccy = g("financial_currency"), g("price_currency")
    currency_mismatch = bool(fin_ccy and px_ccy and fin_ccy != px_ccy)
    if currency_mismatch:
        warnings.append(
            f"Financial statements are in {fin_ccy} but the share price is in {px_ccy}; "
            "ratios would mix currencies."
        )

    for label, period in (("EBIT", g("ebit_period")), ("Balance sheet", g("balance_sheet_period"))):
        age = _period_age_days(period, today)
        if age is not None and age > MAGIC_FORMULA_STALE_DAYS:
            warnings.append(f"{label} period {period} is {age} days old (stale).")

    return {
        "symbol": g("symbol"),
        "company_name": g("company_name"),
        "sector": g("sector"),
        "industry": g("industry"),
        "ebit": ebit,
        "ebit_period": g("ebit_period"),
        "ebit_basis": g("ebit_label"),
        "enterprise_value": ev,
        "enterprise_value_source": ev_source,
        "earnings_yield_percent": _round(earnings_yield, 6),
        "earnings_yield_status": ey_status,
        "earnings_yield_unavailable_reason": ey_reason,
        "net_working_capital": nwc,
        "net_fixed_assets": nfa,
        "invested_capital": invested,
        "balance_sheet_period": g("balance_sheet_period"),
        "return_on_capital_percent": _round(roc, 6),
        "return_on_capital_status": roc_status,
        "return_on_capital_unavailable_reason": roc_reason,
        "currency_mismatch": currency_mismatch,
        "earnings_yield_rank": None,
        "return_on_capital_rank": None,
        "combined_rank": None,
        "overall_position": None,
        "ranked": False,
        "exclusion_reason": None,
        "data_quality_warnings": warnings,
    }


def _competition_ranks(values: dict[str, float]) -> dict[str, int]:
    """Highest value => rank 1; equal values share the same (lowest) rank."""
    keyed = {s: round(v, 10) for s, v in values.items()}
    return {s: 1 + sum(1 for o in keyed.values() if o > v) for s, v in keyed.items()}


def rank_magic_formula(rows: list[dict]) -> list[dict]:
    """Fill ranks on `rows` (in place) for rows that are rankable
    (`exclusion_reason is None` and both factors available) and return the
    ranked rows in final order: combined rank ascending, then earnings-yield
    rank, return-on-capital rank, symbol (fully deterministic).
    """
    rankable = [
        r
        for r in rows
        if r["exclusion_reason"] is None
        and r["earnings_yield_status"] == AVAILABLE
        and r["return_on_capital_status"] == AVAILABLE
    ]
    ey = _competition_ranks({r["symbol"]: r["earnings_yield_percent"] for r in rankable})
    roc = _competition_ranks({r["symbol"]: r["return_on_capital_percent"] for r in rankable})
    for r in rankable:
        r["earnings_yield_rank"] = ey[r["symbol"]]
        r["return_on_capital_rank"] = roc[r["symbol"]]
        r["combined_rank"] = ey[r["symbol"]] + roc[r["symbol"]]
        r["ranked"] = True
    ordered = sorted(
        rankable,
        key=lambda r: (
            r["combined_rank"],
            r["earnings_yield_rank"],
            r["return_on_capital_rank"],
            r["symbol"],
        ),
    )
    for position, r in enumerate(ordered, start=1):
        r["overall_position"] = position
    return ordered


def screen_magic_formula(
    candidates: list[dict], universe: str, today: datetime.date | None = None
) -> dict:
    """Run the Magic Formula over `candidates` (one raw-input dict per
    universe member; a candidate with `"no_data": True` had no fundamentals
    row). One bad candidate never aborts the screen: it is recorded in
    `excluded_by_reason` and the rest are ranked.
    """
    rows: list[dict] = []
    for cand in candidates:
        try:
            if cand.get("no_data"):
                row = compute_magic_formula_metrics({"symbol": cand.get("symbol")}, today)
                row.update(company_name=cand.get("company_name"), sector=cand.get("sector"),
                           industry=cand.get("industry"))
                row["exclusion_reason"] = "no_fundamentals_data"
            else:
                row = compute_magic_formula_metrics(cand, today)
                sector = (cand.get("sector") or "").strip().lower()
                if sector in MAGIC_FORMULA_EXCLUDED_SECTORS:
                    row["exclusion_reason"] = "excluded_sector"
                elif row["currency_mismatch"]:
                    row["exclusion_reason"] = "currency_mismatch"
                elif (
                    row["earnings_yield_status"] != AVAILABLE
                    or row["return_on_capital_status"] != AVAILABLE
                ):
                    row["exclusion_reason"] = "insufficient_or_invalid_financial_data"
        except Exception as exc:  # noqa: BLE001 - one bad stock must not crash the ranking
            row = {
                "symbol": cand.get("symbol"), "company_name": cand.get("company_name"),
                "sector": cand.get("sector"), "industry": cand.get("industry"),
                "ranked": False, "exclusion_reason": "calculation_error",
                "earnings_yield_status": UNAVAILABLE, "return_on_capital_status": UNAVAILABLE,
                "data_quality_warnings": [f"Calculation failed: {exc}"],
            }
        rows.append(row)

    ranked = rank_magic_formula(rows)
    excluded_by_reason: dict[str, int] = {}
    for r in rows:
        if r["exclusion_reason"]:
            excluded_by_reason[r["exclusion_reason"]] = excluded_by_reason.get(r["exclusion_reason"], 0) + 1

    unranked = sorted(
        (r for r in rows if not r["ranked"]), key=lambda r: str(r.get("symbol"))
    )
    return {
        "status": AVAILABLE if ranked else UNAVAILABLE,
        "universe": universe,
        "universe_size": len(rows),
        "ranked_count": len(ranked),
        "excluded_count": len(unranked),
        "excluded_by_reason": dict(sorted(excluded_by_reason.items())),
        "ranked": ranked,
        "unranked": unranked,
    }


# ===========================================================================
# Strategies 29-32: article-based additions
#
#   29. Seasonal Investment Performance Heat Map   (calendar group)
#   30. Investment Diversification Heat Map        (portfolio group)
#   31. EMA + Bollinger Bands + ATR (backtest)     (technical group)
#   32. Technical Crash-Warning Risk Indicator     (quantitative group)
#
# Same conventions as everything above: pure functions over price rows that
# the router has already loaded (no DB / HTTP access here), pandas/numpy
# only, JSON-ready dicts out, `None` when history is insufficient, and no
# buy/sell/hold recommendation anywhere. #31 contains a historical
# backtest simulator; it is an analytics tool, NOT an execution system.
# ===========================================================================

ARTICLE_PRICE_BASES = ("adjusted", "close")
MONTH_LABELS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
QUARTER_LABELS = ("Q1", "Q2", "Q3", "Q4")
# A month/year whose last observation is within this many calendar days of the
# last weekday of the period is treated as complete (covers exchange holidays
# such as Good Friday / Thanksgiving closing the true last weekday).
PERIOD_END_TOLERANCE_DAYS = 4
# Day-over-day jump in (adjusted_close / close) treated as a probable split.
CORPORATE_ACTION_RATIO_JUMP = 0.20


def _last_weekday(year: int, month: int) -> datetime.date:
    last = (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()
    while last.weekday() >= 5:
        last -= datetime.timedelta(days=1)
    return last


def _month_end_date(year: int, month: int) -> datetime.date:
    return (pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)).date()


def _article_price_series(prices, basis: str) -> pd.Series:
    """Positive, finite prices of the chosen basis indexed by date (ascending,
    one row per date). Rows lacking a valid value are dropped, never zero-filled."""
    attr = "adjusted_close" if basis == "adjusted" else "close"
    rows = {}
    for p in prices or []:
        value = getattr(p, attr, None)
        if _is_num(value) and float(value) > 0 and p.date is not None:
            rows[pd.Timestamp(p.date)] = float(value)
    if not rows:
        return pd.Series(dtype=float)
    return pd.Series(rows, dtype=float).sort_index()


def _corporate_action_warnings(prices, basis: str) -> list[str]:
    """Data-quality notes about the price basis. EODHD `close` is NOT adjusted
    for splits; `adjusted_close` is adjusted for splits AND dividends."""
    warnings: list[str] = []
    both = {}
    for p in prices or []:
        c, a = getattr(p, "close", None), getattr(p, "adjusted_close", None)
        if _is_num(c) and _is_num(a) and float(c) > 0 and float(a) > 0:
            both[pd.Timestamp(p.date)] = float(a) / float(c)
    if len(both) < 3:
        return warnings
    ratio = pd.Series(both).sort_index()
    jumps = ratio.pct_change().abs()
    for ts, jump in jumps[jumps > CORPORATE_ACTION_RATIO_JUMP].items():
        if basis == "close":
            warnings.append(
                f"Probable stock split around {ts.date().isoformat()}: raw close is not split-adjusted, "
                "so returns spanning this date are distorted. Use price_basis='adjusted'."
            )
        else:
            warnings.append(
                f"Adjusted/raw close ratio changes by {jump * 100:.0f}% around {ts.date().isoformat()} "
                "(probable split or restated history); verify the stored adjusted_close history."
            )
    span_days = (ratio.index[-1] - ratio.index[0]).days
    if basis == "adjusted" and span_days > 730 and float(ratio.max() - ratio.min()) < 1e-9:
        warnings.append(
            "adjusted_close equals close for the entire history, so dividends are probably not "
            "reflected (stored history may pre-date dividend adjustment). Returns are then price-only; "
            "run POST /api/admin/backfill with restate_adjusted=true to restate adjusted_close."
        )
    return warnings


def _is_real_number(value) -> bool:
    """Strict numeric check for USER input: a real, finite int/float (not a bool,
    not a numeric string such as "60", not NaN/inf). `_is_num` above is lenient
    about numeric strings, which is wrong for validating request payloads."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _pct(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else _round(value * 100.0, digits)


def _sign(value: float | None) -> str | None:
    if value is None:
        return None
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _linked_return(returns: list[float]) -> float:
    """Geometric linking: prod(1 + r) - 1 (NEVER the sum of returns)."""
    total = 1.0
    for r in returns:
        total *= 1.0 + r
    return total - 1.0


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0, "average_return_percent": None, "median_return_percent": None,
            "positive_count": 0, "positive_rate_percent": None,
            "best_return_percent": None, "worst_return_percent": None,
        }
    arr = np.array(values, dtype=float)
    positive = int((arr > 0).sum())
    return {
        "count": len(values),
        "average_return_percent": _pct(float(arr.mean())),
        "median_return_percent": _pct(float(np.median(arr))),
        "positive_count": positive,
        "positive_rate_percent": _round(positive / len(values) * 100.0, 1),
        "best_return_percent": _pct(float(arr.max())),
        "worst_return_percent": _pct(float(arr.min())),
    }


# ---------------------------------------------------------------------------
# Strategy 29: Seasonal Investment Performance Heat Map
#
# Source: "How to Create Seasonal Investment Performance Heat Maps in Python"
# (C. L. Tanny). Calendar-year x month grid of returns, plus quarterly
# returns obtained by GEOMETRICALLY linking the three monthly returns:
#     quarter = prod(1 + monthly) - 1        (never a sum of percentages)
#
# Rules (all enforced in code and covered by tests):
#   * Monthly return = last observation of the month / last observation of
#     the IMMEDIATELY PRECEDING calendar month - 1. If the preceding month has
#     no observation the month is reported as missing/unavailable - it is never
#     treated as a 0% month, and a return is never stretched over a gap.
#   * The first month of the data has no base and is reported as `no_base`.
#   * A month still in progress (or whose data ends early) is `partial_month`
#     and is excluded from complete-period statistics.
#   * A quarter is `complete` only when all three monthly returns are valid.
#     Otherwise `returnPercent` is null; `partialReturnPercent` carries the
#     linked return of the leading run of valid months (quarter-to-date) and
#     `monthsIncluded` says how many months it covers.
#   * Adjustment policy: price_basis='adjusted' (default) = EODHD
#     adjusted_close (splits + dividends, i.e. total return); 'close' = raw
#     close (price only, NOT split-adjusted).
#   * The requested range has MONTH granularity (any month overlapping
#     [start, end] is shown) and is never hard-coded; history before `start`
#     is still used as the base for the first displayed month.
# ---------------------------------------------------------------------------

SEASONAL_DISCLAIMER = (
    "Historical, descriptive statistics only. Past seasonal patterns do not imply future returns, "
    "and a pattern across a small number of years is not statistically reliable."
)


def compute_seasonal_heat_map(
    prices,
    symbol: str | None = None,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    price_basis: str = "adjusted",
    as_of: datetime.date | None = None,
) -> dict | None:
    """Monthly + quarterly return heat-map dataset. None when fewer than two
    distinct months of data (no return can be formed) or nothing in range."""
    if price_basis not in ARTICLE_PRICE_BASES:
        raise ValueError(f"price_basis must be one of {ARTICLE_PRICE_BASES}")
    if start and end and start > end:
        raise ValueError("start must not be after end")

    series = _article_price_series(prices, price_basis)
    if series.empty:
        return None
    as_of = as_of or datetime.datetime.now(datetime.timezone.utc).date()

    periods = series.index.to_period("M")
    month_last_value = series.groupby(periods).last()
    month_last_date = series.index.to_series().groupby(periods).max()
    if len(month_last_value) < 2:
        return None

    first_period = month_last_value.index[0]
    last_period = month_last_value.index[-1]
    data_first = series.index[0].date()
    data_last = series.index[-1].date()

    # Final month: complete only if the calendar month is over AND the data reaches its end.
    final_complete = (
        as_of > _month_end_date(last_period.year, last_period.month)
        and data_last
        >= _last_weekday(last_period.year, last_period.month)
        - datetime.timedelta(days=PERIOD_END_TOLERANCE_DAYS)
    )

    range_first = pd.Period(start, "M") if start else first_period
    range_last = pd.Period(end, "M") if end else last_period
    shown_first = max(range_first, first_period)
    shown_last = min(range_last, last_period)
    if shown_first > shown_last:
        return None

    # Every month from the first displayed to the last displayed.
    cells: dict[tuple[int, int], dict] = {}
    for period in pd.period_range(shown_first, shown_last, freq="M"):
        key = (period.year, period.month)
        cell = {
            "year": period.year, "month": period.month, "label": MONTH_LABELS[period.month - 1],
            "return_percent": None, "sign": None, "status": "missing",
            "base_date": None, "end_date": None,
        }
        if period in month_last_value.index:
            cell["end_date"] = month_last_date[period].date().isoformat()
            prev = period - 1
            if period == first_period:
                cell["status"] = "no_base"
            elif prev not in month_last_value.index:
                cell["status"] = "prior_month_missing"
            else:
                r = float(month_last_value[period] / month_last_value[prev] - 1.0)
                cell["base_date"] = month_last_date[prev].date().isoformat()
                if period == last_period and not final_complete:
                    cell["status"] = "partial_month"
                    cell["partial_return_percent"] = _pct(r)
                else:
                    cell["status"] = "ok"
                    cell["return_percent"] = _pct(r)
                    cell["sign"] = _sign(r)
                    cell["_r"] = r
        cells[key] = cell

    def month_cell(year: int, month: int) -> dict:
        existing = cells.get((year, month))
        if existing is not None:
            return existing
        return {
            "year": year, "month": month, "label": MONTH_LABELS[month - 1],
            "return_percent": None, "sign": None, "status": "out_of_range",
            "base_date": None, "end_date": None,
        }

    rows = []
    quarter_values: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    month_values: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    all_abs: list[float] = []
    for year in range(shown_first.year, shown_last.year + 1):
        months = [month_cell(year, m) for m in range(1, 13)]
        quarters = []
        for q in range(4):
            trio = months[q * 3 : q * 3 + 3]
            ok_returns = [c["_r"] for c in trio if c["status"] == "ok"]
            leading = []
            for c in trio:
                if c["status"] != "ok":
                    break
                leading.append(c["_r"])
            q_cell = {
                "year": year, "quarter": q + 1, "label": QUARTER_LABELS[q],
                "return_percent": None, "partial_return_percent": None, "sign": None,
                "status": "missing", "months_included": len(leading),
            }
            if len(ok_returns) == 3:
                linked = _linked_return(ok_returns)
                q_cell.update(
                    return_percent=_pct(linked), sign=_sign(linked), status="complete", months_included=3
                )
                quarter_values[q + 1].append(linked)
            elif leading:
                q_cell.update(
                    partial_return_percent=_pct(_linked_return(leading)), status="partial"
                )
            elif ok_returns or any(c["status"] == "partial_month" for c in trio):
                q_cell["status"] = "partial"
            elif all(c["status"] == "out_of_range" for c in trio):
                q_cell["status"] = "out_of_range"
            quarters.append(q_cell)

        year_ok = [c["_r"] for c in months if c["status"] == "ok"]
        if len(year_ok) == 12:
            linked_year = _linked_return(year_ok)
            calendar_year = {"return_percent": _pct(linked_year), "sign": _sign(linked_year), "status": "complete"}
        else:
            calendar_year = {"return_percent": None, "sign": None, "status": "partial" if year_ok else "missing"}

        for c in months:
            if c["status"] == "ok":
                month_values[c["month"]].append(c["_r"])
                all_abs.append(abs(c["_r"]))
        rows.append({"year": year, "months": months, "quarters": quarters, "calendar_year": calendar_year})

    for row in rows:
        for c in row["months"]:
            c.pop("_r", None)

    warnings = _corporate_action_warnings(prices, price_basis)
    if start and start < data_first:
        warnings.append(
            f"Requested start {start.isoformat()} precedes the earliest stored price "
            f"({data_first.isoformat()}); earlier months are unavailable. Extend history with "
            "POST /api/admin/backfill."
        )
    if end and end > data_last:
        warnings.append(
            f"Requested end {end.isoformat()} is after the latest stored price ({data_last.isoformat()})."
        )
    if not final_complete and shown_last == last_period:
        warnings.append(
            f"{MONTH_LABELS[last_period.month - 1]} {last_period.year} is not a complete month; it is "
            "excluded from averages and from quarterly/annual linking."
        )
    if any(c["status"] in ("missing", "prior_month_missing") for r in rows for c in r["months"]):
        warnings.append(
            "Some months have no price data. They are shown as unavailable (never as 0% months), "
            "and any quarter or year containing them is not reported as complete."
        )

    return {
        "symbol": symbol,
        "price_basis": price_basis,
        "adjustment_policy": (
            "EODHD adjusted_close (adjusted for splits and dividends: total-return basis)"
            if price_basis == "adjusted"
            else "EODHD raw close (price only; NOT adjusted for splits or dividends)"
        ),
        "requested_start": start.isoformat() if start else None,
        "requested_end": end.isoformat() if end else None,
        "data_first_date": data_first.isoformat(),
        "data_last_date": data_last.isoformat(),
        "as_of": as_of.isoformat(),
        "month_labels": list(MONTH_LABELS),
        "quarter_labels": list(QUARTER_LABELS),
        "rows": rows,
        "monthly_summary": [
            {"month": m, "label": MONTH_LABELS[m - 1], **_summary_stats(month_values[m])} for m in range(1, 13)
        ],
        "quarterly_summary": [
            {"quarter": q, "label": QUARTER_LABELS[q - 1], **_summary_stats(quarter_values[q])} for q in range(1, 5)
        ],
        "color_scale": {
            "metric": "return_percent",
            "neutral": 0.0,
            "max_abs_return_percent": _pct(max(all_abs)) if all_abs else None,
            "note": "Diverging scale centred on 0%: positive = gain, negative = loss, null = no data (not 0%).",
        },
        "methodology": [
            "Monthly return = last price of the month / last price of the immediately preceding month - 1.",
            "Quarterly return = (1+m1)(1+m2)(1+m3) - 1 using the three monthly returns (geometric linking, not a sum).",
            "Months without a valid base are shown as unavailable, never as zero-return months.",
            "Incomplete months/quarters are labelled and excluded from complete-period statistics.",
            "The requested range has month granularity; earlier history is still used as the base for the first month.",
        ],
        "data_quality_warnings": warnings,
        "disclaimer": SEASONAL_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Strategy 30: Investment Diversification Heat Map
#
# Source: "How to Create the Investment Diversification Heat Map in Python"
# (C. L. Tanny). Calendar-year returns for several instruments, ranked within
# each year, with a user-weighted portfolio shown alongside them.
#
# Portfolio policy (explicit, because it changes the numbers):
#   * Long-only, fully invested: each weight >= 0 and the weights sum to 100%
#     (+/- 0.01). No cash, leverage or shorting. A ticker omitted from the
#     weights has a 0% weight (it is still shown/ranked as an instrument).
#   * ANNUAL REBALANCING: weights are reset to the targets at the start of each
#     calendar year and drift (buy-and-hold) inside the year, so
#         portfolio year return = sum_i w_i * r_i(year).
#     This is NOT equivalent to a buy-and-hold portfolio across years.
#   * A year is reported for the portfolio only if every positively-weighted
#     asset has a complete return for that year. Missing assets are never
#     dropped and weights are never silently renormalised.
#   * Annual return = last price of the year / last price of the immediately
#     preceding calendar year - 1 (adjusted_close by default). Listing years
#     (no base), in-progress years, and years following a data gap are NOT
#     complete and are excluded from ranking.
#
# A portfolio ranking between its constituents says nothing about its risk,
# and this chart alone does not establish diversification benefits.
# ---------------------------------------------------------------------------

DIVERSIFICATION_WEIGHT_TOLERANCE = 0.01  # percentage points
DIVERSIFICATION_DISCLAIMER = (
    "Descriptive annual returns only. A portfolio whose return ranks between its constituents is not "
    "thereby lower-risk, and this heat map alone does not establish diversification benefits (it ignores "
    "volatility, drawdowns and correlations). Past returns do not predict future returns."
)


def validate_portfolio_weights(symbols: list[str], weights_percent: dict) -> dict[str, float]:
    """Validate user weights (percent units). Raises ValueError with a precise
    message; returns {SYMBOL: weight_percent} (omitted symbols -> 0.0)."""
    if not isinstance(weights_percent, dict) or not weights_percent:
        raise ValueError("weights must be a non-empty mapping of ticker to percent weight.")
    known = {s.upper() for s in symbols}
    cleaned: dict[str, float] = {}
    for raw_symbol, value in weights_percent.items():
        symbol = str(raw_symbol).strip().upper()
        if symbol not in known:
            raise ValueError(f"Weight given for '{raw_symbol}', which is not in the requested tickers.")
        if symbol in cleaned:
            raise ValueError(f"Duplicate weight for '{symbol}'.")
        if not _is_real_number(value):
            raise ValueError(f"Weight for '{symbol}' must be a finite number (got {value!r}).")
        weight = float(value)
        if weight < 0:
            raise ValueError(f"Weight for '{symbol}' must not be negative (long-only portfolio).")
        if weight > 100:
            raise ValueError(f"Weight for '{symbol}' must not exceed 100%.")
        cleaned[symbol] = weight
    total = sum(cleaned.values())
    if abs(total - 100.0) > DIVERSIFICATION_WEIGHT_TOLERANCE:
        raise ValueError(
            f"Weights must sum to 100% (fully invested, no cash/leverage); they sum to {total:.4f}%."
        )
    return {s: cleaned.get(s.upper(), 0.0) for s in (x.upper() for x in symbols)}


def _annual_cells(series: pd.Series, years: list[int], as_of: datetime.date) -> dict[int, dict]:
    """Per-calendar-year return cells for one instrument (see policy above)."""
    out: dict[int, dict] = {}
    if series.empty:
        for y in years:
            out[y] = _empty_annual_cell(y, "missing")
        return out
    periods = series.index.to_period("Y")
    year_last_value = series.groupby(periods).last()
    year_last_date = series.index.to_series().groupby(periods).max()
    first_year = year_last_value.index[0].year
    first_obs_value, first_obs_date = float(series.iloc[0]), series.index[0].date()
    for y in years:
        cell = _empty_annual_cell(y, "not_listed" if y < first_year else "missing")
        period = pd.Period(year=y, freq="Y")
        if period in year_last_value.index:
            end_value = float(year_last_value[period])
            end_date = year_last_date[period].date()
            year_over = as_of > datetime.date(y, 12, 31)
            reaches_year_end = end_date >= _last_weekday(y, 12) - datetime.timedelta(days=PERIOD_END_TOLERANCE_DAYS)
            prev = period - 1
            if y == first_year:
                cell["status"] = "no_base"
                if end_date > first_obs_date:
                    partial = end_value / first_obs_value - 1.0
                    cell.update(
                        partial_return_percent=_pct(partial),
                        window_start=first_obs_date.isoformat(), window_end=end_date.isoformat(),
                    )
            elif prev not in year_last_value.index:
                cell["status"] = "prior_year_missing"
            else:
                base_value = float(year_last_value[prev])
                base_date = year_last_date[prev].date()
                r = end_value / base_value - 1.0
                cell.update(window_start=base_date.isoformat(), window_end=end_date.isoformat())
                if year_over and reaches_year_end:
                    cell.update(status="complete", return_percent=_pct(r), sign=_sign(r), _r=r)
                else:
                    cell.update(status="partial", partial_return_percent=_pct(r), _partial_r=r)
        out[y] = cell
    return out


def _empty_annual_cell(year: int, status: str) -> dict:
    return {
        "year": year, "return_percent": None, "partial_return_percent": None, "sign": None,
        "status": status, "rank": None, "window_start": None, "window_end": None,
    }


def _rank_desc(values: dict[str, float]) -> dict[str, int]:
    """Competition ranking, best (highest) = 1, ties share the lowest rank."""
    keyed = {k: round(v, 10) for k, v in values.items()}
    return {k: 1 + sum(1 for o in keyed.values() if o > v) for k, v in keyed.items()}


def compute_diversification_heat_map(
    prices_by_symbol: dict,
    weights_percent: dict | None = None,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    price_basis: str = "adjusted",
    as_of: datetime.date | None = None,
) -> dict | None:
    """Annual-return heat map + ranking (+ weighted portfolio). None when no
    instrument has at least two calendar years of data in range."""
    if price_basis not in ARTICLE_PRICE_BASES:
        raise ValueError(f"price_basis must be one of {ARTICLE_PRICE_BASES}")
    if start and end and start > end:
        raise ValueError("start must not be after end")
    symbols = [s.upper() for s in prices_by_symbol]
    weights = validate_portfolio_weights(symbols, weights_percent) if weights_percent else None
    as_of = as_of or datetime.datetime.now(datetime.timezone.utc).date()

    original_key = {k.upper(): k for k in prices_by_symbol}
    series = {s: _article_price_series(prices_by_symbol[original_key[s]], price_basis) for s in symbols}
    if all(s.empty for s in series.values()):
        return None
    data_years = [y for s in series.values() if not s.empty for y in (s.index[0].year, s.index[-1].year)]
    first_year = max(min(data_years), start.year) if start else min(data_years)
    last_year = min(max(data_years), end.year) if end else max(data_years)
    if first_year > last_year:
        return None
    years = list(range(first_year, last_year + 1))

    cells = {s: _annual_cells(series[s], years, as_of) for s in symbols}

    # Ranking within each year (complete asset returns only).
    year_rankings = []
    portfolio_cells: dict[int, dict] = {}
    warnings: list[str] = []
    for y in years:
        complete = {s: cells[s][y]["_r"] for s in symbols if cells[s][y]["status"] == "complete"}
        for s, rank in _rank_desc(complete).items():
            cells[s][y]["rank"] = rank

        p_cell = _empty_annual_cell(y, "no_weights")
        p_r = None
        if weights is not None:
            held = [s for s in symbols if weights[s] > 0]
            missing = [s for s in held if cells[s][y]["status"] != "complete"]
            if not missing:
                p_r = sum(weights[s] / 100.0 * cells[s][y]["_r"] for s in held)
                p_cell.update(status="complete", return_percent=_pct(p_r), sign=_sign(p_r), _r=p_r)
            elif all(cells[s][y].get("_partial_r") is not None for s in held):
                part = sum(weights[s] / 100.0 * cells[s][y]["_partial_r"] for s in held)
                p_cell.update(status="partial", partial_return_percent=_pct(part))
                ends = {cells[s][y]["window_end"] for s in held}
                if len(ends) > 1:
                    warnings.append(
                        f"{y}: weighted assets' latest observations differ ({', '.join(sorted(ends))}); "
                        "the partial-year portfolio figure mixes slightly different end dates."
                    )
            else:
                p_cell.update(
                    status="incomplete",
                    reason="Weighted asset(s) without a complete return for this year: " + ", ".join(missing),
                )
        portfolio_cells[y] = p_cell

        ranking_values = dict(complete)
        if p_r is not None:
            ranking_values["PORTFOLIO"] = p_r
        with_portfolio = _rank_desc(ranking_values)
        if p_r is not None:
            p_cell["rank"] = with_portfolio["PORTFOLIO"]
            p_cell["rank_among"] = len(ranking_values)
        year_rankings.append(
            {
                "year": y,
                "complete_asset_count": len(complete),
                "ranking": [
                    {
                        "rank": with_portfolio[k], "symbol": k, "is_portfolio": k == "PORTFOLIO",
                        "return_percent": _pct(v),
                    }
                    for k, v in sorted(ranking_values.items(), key=lambda kv: (-round(kv[1], 10), kv[0]))
                ],
            }
        )

    asset_rows = []
    for s in symbols:
        done = [cells[s][y]["_r"] for y in years if cells[s][y]["status"] == "complete"]
        best = max(((cells[s][y]["_r"], y) for y in years if cells[s][y]["status"] == "complete"), default=None)
        worst = min(((cells[s][y]["_r"], y) for y in years if cells[s][y]["status"] == "complete"), default=None)
        series_s = series[s]
        asset_rows.append(
            {
                "symbol": s,
                "weight_percent": _round(weights[s], 4) if weights else None,
                "first_date": series_s.index[0].date().isoformat() if not series_s.empty else None,
                "last_date": series_s.index[-1].date().isoformat() if not series_s.empty else None,
                "cells": [{k: v for k, v in cells[s][y].items() if not k.startswith("_")} for y in years],
                "complete_years": len(done),
                "average_annual_return_percent": _pct(float(np.mean(done))) if done else None,
                "best_year": {"year": best[1], "return_percent": _pct(best[0])} if best else None,
                "worst_year": {"year": worst[1], "return_percent": _pct(worst[0])} if worst else None,
            }
        )
        if series_s.empty:
            warnings.append(f"{s}: no usable price history; shown as unavailable in every year.")
        elif series_s.index[0].year > first_year:
            warnings.append(
                f"{s}: history starts {series_s.index[0].date().isoformat()}; earlier years are unavailable "
                "(not 0%), and its first calendar year has no base so it is not ranked."
            )

    all_complete = [
        cells[s][y]["_r"] for s in symbols for y in years if cells[s][y]["status"] == "complete"
    ]
    if not all_complete:
        return None
    if any(p["status"] == "partial" for c in cells.values() for p in c.values()):
        warnings.append("The in-progress year is shown as a partial figure and is not ranked.")
    for s in symbols:
        for w in _corporate_action_warnings(prices_by_symbol[original_key[s]], price_basis):
            warnings.append(f"{s}: {w}")

    portfolio = None
    if weights is not None:
        portfolio = {
            "weights_percent": {s: _round(weights[s], 4) for s in symbols},
            "rebalancing_policy": "annual_reset",
            "policy_description": (
                "Target weights are re-applied at the start of every calendar year and drift (buy-and-hold) "
                "within the year: portfolio return = sum(weight_i x return_i). This is not equivalent to a "
                "buy-and-hold portfolio across years."
            ),
            "cells": [{k: v for k, v in portfolio_cells[y].items() if not k.startswith("_")} for y in years],
        }

    return {
        "symbols": symbols,
        "price_basis": price_basis,
        "adjustment_policy": (
            "EODHD adjusted_close (splits + dividends: total-return basis)"
            if price_basis == "adjusted"
            else "EODHD raw close (price only; NOT adjusted for splits or dividends)"
        ),
        "requested_start": start.isoformat() if start else None,
        "requested_end": end.isoformat() if end else None,
        "years": years,
        "as_of": as_of.isoformat(),
        "assets": asset_rows,
        "portfolio": portfolio,
        "yearly_rankings": year_rankings,
        "color_scale": {
            "metric": "return_percent",
            "neutral": 0.0,
            "max_abs_return_percent": _pct(max(abs(v) for v in all_complete)),
            "note": "Diverging scale centred on 0%; rank 1 = best return in that calendar year.",
        },
        "methodology": [
            "Annual return = last price of the year / last price of the previous calendar year - 1.",
            "Only complete years are ranked; listing years, in-progress years and years after a data gap are not.",
            "Portfolio = fixed target weights reset every January (annual rebalancing): sum(weight x return).",
            "A year is reported for the portfolio only if every positively-weighted asset has a complete return.",
            "The requested range has calendar-year granularity.",
        ],
        "data_quality_warnings": warnings,
        "disclaimer": DIVERSIFICATION_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Strategy 31: EMA + Bollinger Bands + ATR (analytics + historical backtest)
#
# Source: "How I Coded a Trading Bot for a 60% YoY Return" (K. Borimechkov).
# THIS IS A STRATEGY-ANALYTICS / BACKTESTING FEATURE. It places no orders,
# talks to no broker and contains no execution subsystem. The article's
# reported return is NOT reproduced or claimed here.
#
# Reference rules (article defaults kept; every parameter is configurable):
#   EMA 30 | Bollinger 15 periods x 1.5 sd | RSI 10 | ATR 7 | trend look-back 7
#   Uptrend   : for each of the `trend_lookback` candles BEFORE the signal
#               candle, min(open, close) > EMA at that candle ("body above EMA").
#   Downtrend : for each of those candles, max(open, close) < EMA.
#   Long setup : uptrend and signal-candle close <= lower Bollinger Band.
#   Short setup: downtrend and signal-candle close >= upper Bollinger Band.
#   Stop distance = ATR(signal candle) x stop_loss_atr_mult
#   Take-profit distance = stop distance x take_profit_ratio
#   Extra exits: RSI >= 90 closes a long, RSI <= 10 closes a short.
#
# Documented interpretation choices (the article text leaves them open):
#   * The trend window is the `trend_lookback` candles PRECEDING the signal
#     candle (the signal candle itself must be the pull-back to the band).
#   * stop_loss_atr_mult (1.5) and take_profit_ratio (1.5) are configurable
#     implementation defaults, not values verified from the article.
#   * Bollinger sd uses the sample standard deviation (ddof=1), matching this
#     repository's existing Bollinger Bands strategy.
#
# Execution model (no look-ahead):
#   * Indicators at candle t use candles <= t only. A setup seen at the CLOSE
#     of candle t is filled at the OPEN of candle t+1 (never at the signal close).
#   * Stop/target levels are anchored to the actual entry fill. ATR is taken
#     from the signal candle, so it is known before entry.
#   * Stops/targets are checked against each candle's high/low; if one candle
#     touches both, the STOP is assumed first (conservative). A candle that
#     gaps through a level exits at the open, not at the level.
#   * RSI exits are seen at a candle's close and executed at the next open.
#   * Every fill pays `slippage_percent` against the trader, and each side pays
#     `commission_percent`. One position at a time, no pyramiding; the position
#     is `position_fraction` of equity (<= 1, no leverage). Short positions
#     assume borrow is available at no cost and ignore dividends.
#   * An open position is force-closed at the last close of each segment.
#
# Out-of-sample discipline: bars are split in time order into a TRAINING
# segment (first `train_fraction`) and an OUT-OF-SAMPLE segment (the rest).
# Each segment is simulated independently from a flat start. Optional
# parameter selection (`optimize=True`) looks at the training segment ONLY.
# ---------------------------------------------------------------------------

EBA_DEFAULTS: dict = {
    "ema_period": 30,
    "bb_period": 15,
    "bb_std": 1.5,
    "rsi_period": 10,
    "atr_period": 7,
    "trend_lookback": 7,
    "stop_loss_atr_mult": 1.5,
    "take_profit_ratio": 1.5,
    "rsi_long_exit": 90.0,
    "rsi_short_exit": 10.0,
    "commission_percent": 0.02,
    "slippage_percent": 0.05,
    "position_fraction": 1.0,
    "train_fraction": 0.6,
    "allow_short": True,
}
# name -> (kind, minimum, maximum)
EBA_BOUNDS: dict = {
    "ema_period": ("int", 2, 400),
    "bb_period": ("int", 2, 400),
    "bb_std": ("float", 0.25, 5.0),
    "rsi_period": ("int", 2, 100),
    "atr_period": ("int", 2, 100),
    "trend_lookback": ("int", 1, 50),
    "stop_loss_atr_mult": ("float", 0.1, 10.0),
    "take_profit_ratio": ("float", 0.1, 10.0),
    "rsi_long_exit": ("float", 50.0, 100.0),
    "rsi_short_exit": ("float", 0.0, 50.0),
    "commission_percent": ("float", 0.0, 2.0),
    "slippage_percent": ("float", 0.0, 2.0),
    "position_fraction": ("float", 0.01, 1.0),
    "train_fraction": ("float", 0.3, 0.9),
    "allow_short": ("bool", None, None),
}
EBA_TIMEFRAMES = ("1d", "1h")
EBA_MIN_SEGMENT_BARS = 30
EBA_MIN_TRAIN_TRADES = 5
EBA_OPT_STOP_MULTS = (1.0, 1.5, 2.0, 3.0)
EBA_OPT_TP_RATIOS = (1.0, 1.5, 2.0, 3.0)
EBA_MAX_TRADES_RETURNED = 200
EBA_EQUITY_POINTS = 250
EBA_DISCLAIMER = (
    "Historical simulation for analysis only: no orders are placed and nothing here is a recommendation. "
    "Results depend on the stated execution/cost assumptions, do not guarantee future performance, and the "
    "source article's reported return has not been reproduced or verified by this implementation."
)


@dataclasses.dataclass(frozen=True)
class PriceBar:
    """Minimal OHLCV bar (e.g. an EODHD hourly candle). Duck-types the DailyPrice
    attributes the OHLC helpers read; `date` may be a datetime (naive UTC)."""

    date: datetime.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def resolve_ema_bollinger_atr_params(overrides: dict | None = None) -> dict:
    """Defaults + validated overrides. Raises ValueError naming the bad parameter."""
    params = dict(EBA_DEFAULTS)
    for name, value in (overrides or {}).items():
        if value is None:
            continue
        if name not in EBA_BOUNDS:
            raise ValueError(f"Unknown parameter '{name}'. Valid: {', '.join(sorted(EBA_BOUNDS))}.")
        kind, lo, hi = EBA_BOUNDS[name]
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"'{name}' must be true or false.")
            params[name] = value
            continue
        if not _is_real_number(value):
            raise ValueError(f"'{name}' must be a finite number (got {value!r}).")
        number = float(value)
        if kind == "int":
            if number != int(number):
                raise ValueError(f"'{name}' must be a whole number.")
            number = int(number)
        if not lo <= number <= hi:
            raise ValueError(f"'{name}' must be between {lo} and {hi} (got {number}).")
        params[name] = number
    if params["rsi_short_exit"] >= params["rsi_long_exit"]:
        raise ValueError("rsi_short_exit must be lower than rsi_long_exit.")
    return params


def _ts_str(ts: pd.Timestamp, intraday: bool) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ") if intraday else ts.date().isoformat()


def _eba_clean_frame(prices) -> pd.DataFrame:
    frame = _ohlc_clean(prices)
    if frame.empty:
        return frame
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["open"])
    ok = (
        (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
        & (frame["high"] >= frame["low"])
        & np.isfinite(frame[["open", "high", "low", "close"]]).all(axis=1)
    )
    return frame[ok]


def _eba_indicators(frame: pd.DataFrame, p: dict) -> dict:
    """All indicator series, each value using only candles up to and including its own."""
    close = frame["close"]
    ema = _ema(close, p["ema_period"])
    mid = close.rolling(p["bb_period"]).mean()
    sd = close.rolling(p["bb_period"]).std()
    upper = mid + p["bb_std"] * sd
    lower = mid - p["bb_std"] * sd
    rsi = _rsi(close, p["rsi_period"])
    atr = _atr(frame, p["atr_period"])
    body_low = frame[["open", "close"]].min(axis=1)
    body_high = frame[["open", "close"]].max(axis=1)
    up_bar = (body_low > ema).astype(float)  # NaN EMA compares False
    down_bar = (body_high < ema).astype(float)
    lookback = p["trend_lookback"]
    # window = the `lookback` candles BEFORE the candle being evaluated
    uptrend = up_bar.rolling(lookback).min().shift(1).eq(1.0)
    downtrend = down_bar.rolling(lookback).min().shift(1).eq(1.0)
    valid = ema.notna() & upper.notna() & atr.notna()
    return {
        "ema": ema, "bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "rsi": rsi, "atr": atr,
        "uptrend": uptrend, "downtrend": downtrend,
        "long_setup": uptrend & (close <= lower) & valid,
        "short_setup": downtrend & (close >= upper) & valid,
    }


def _eba_simulate(frame: pd.DataFrame, ind: dict, p: dict, lo: int, hi: int, intraday: bool) -> dict:
    """Event-driven simulation over bars [lo, hi), starting flat. See block comment."""
    o = frame["open"].to_numpy(float)
    h = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    c = frame["close"].to_numpy(float)
    atr = ind["atr"].to_numpy(float)
    rsi = ind["rsi"].to_numpy(float)
    long_sig = ind["long_setup"].to_numpy(bool)
    short_sig = ind["short_setup"].to_numpy(bool) if p["allow_short"] else np.zeros(len(c), bool)
    index = frame.index
    slip = p["slippage_percent"] / 100.0
    comm = p["commission_percent"] / 100.0
    frac = p["position_fraction"]

    trades: list[dict] = []
    equity = np.ones(hi - lo)
    realized = 1.0
    pos: dict | None = None
    pending_entry: tuple | None = None
    pending_exit = False
    bars_in_market = 0
    ambiguous_bars = 0

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal pos, realized
        side = pos["side"]
        exit_fill = price * (1.0 - side * slip)
        price_return = side * (exit_fill / pos["entry"] - 1.0)
        net = frac * price_return - 2.0 * frac * comm
        realized = pos["eq0"] * (1.0 + net)
        trades.append(
            {
                "side": "long" if side > 0 else "short",
                "signal_date": _ts_str(index[pos["signal_idx"]], intraday),
                "entry_date": _ts_str(index[pos["entry_idx"]], intraday),
                "entry_price": _round(pos["entry"], 4),
                "exit_date": _ts_str(index[i], intraday),
                "exit_price": _round(exit_fill, 4),
                "exit_reason": reason,
                "stop_loss": _round(pos["sl"], 4),
                "take_profit": _round(pos["tp"], 4),
                "atr_at_signal": _round(pos["atr"], 4),
                "price_return_percent": _pct(price_return),
                "equity_return_percent": _pct(net),
                "bars_held": i - pos["entry_idx"],
                "_net": net,
            }
        )
        pos = None

    for i in range(lo, hi):
        if pos is not None and pending_exit:
            close_position(i, o[i], "rsi_exit")
            pending_exit = False

        if pending_entry is not None:
            side, atr_signal, signal_idx = pending_entry
            pending_entry = None
            entry = o[i] * (1.0 + side * slip)
            stop_distance = atr_signal * p["stop_loss_atr_mult"]
            sl = entry - side * stop_distance
            tp = entry + side * stop_distance * p["take_profit_ratio"]
            if sl > 0 and tp > 0:
                pos = {
                    "side": side, "entry": entry, "sl": sl, "tp": tp, "eq0": realized,
                    "entry_idx": i, "signal_idx": signal_idx, "atr": atr_signal,
                }

        if pos is not None:
            side = pos["side"]
            held_before = pos["entry_idx"] < i
            if side > 0:
                hit_sl, hit_tp = low[i] <= pos["sl"], h[i] >= pos["tp"]
                gap_sl, gap_tp = o[i] <= pos["sl"], o[i] >= pos["tp"]
            else:
                hit_sl, hit_tp = h[i] >= pos["sl"], low[i] <= pos["tp"]
                gap_sl, gap_tp = o[i] >= pos["sl"], o[i] <= pos["tp"]
            if held_before and gap_sl:
                close_position(i, o[i], "stop_loss_gap")
            elif held_before and gap_tp:
                close_position(i, o[i], "take_profit_gap")
            elif hit_sl:
                ambiguous_bars += int(hit_tp)
                close_position(i, pos["sl"], "stop_loss")
            elif hit_tp:
                close_position(i, pos["tp"], "take_profit")

        if pos is not None:
            bars_in_market += 1
            if i == hi - 1:
                close_position(i, c[i], "end_of_data")
            else:
                if (pos["side"] > 0 and rsi[i] >= p["rsi_long_exit"]) or (
                    pos["side"] < 0 and rsi[i] <= p["rsi_short_exit"]
                ):
                    pending_exit = True

        if pos is None and i + 1 < hi:
            if long_sig[i]:
                pending_entry = (1, float(atr[i]), i)
            elif short_sig[i]:
                pending_entry = (-1, float(atr[i]), i)

        if pos is None:
            equity[i - lo] = realized
        else:
            equity[i - lo] = pos["eq0"] * (
                1.0 + frac * pos["side"] * (c[i] / pos["entry"] - 1.0) - frac * comm
            )
    return {"trades": trades, "equity": equity, "bars_in_market": bars_in_market, "ambiguous_bars": ambiguous_bars}


def _eba_metrics(sim: dict, close: np.ndarray, index: pd.DatetimeIndex) -> dict:
    equity, trades = sim["equity"], sim["trades"]
    n = len(equity)
    net = [t["_net"] for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x < 0]
    peak = np.maximum.accumulate(equity)
    max_dd = float((equity / peak - 1.0).min())
    span_days = (index[-1] - index[0]).days
    total = float(equity[-1] - 1.0)
    cagr = float(equity[-1] ** (365.25 / span_days) - 1.0) if span_days >= 180 and equity[-1] > 0 else None
    benchmark = float(close[-1] / close[0] - 1.0)
    return {
        "bars": n,
        "start": index[0],
        "end": index[-1],
        "total_return_percent": _pct(total),
        "cagr_percent": _pct(cagr),
        "max_drawdown_percent": _pct(max_dd),
        "trade_count": len(trades),
        "long_trades": sum(1 for t in trades if t["side"] == "long"),
        "short_trades": sum(1 for t in trades if t["side"] == "short"),
        "win_rate_percent": _round(len(wins) / len(trades) * 100.0, 1) if trades else None,
        "profit_factor": _round(sum(wins) / -sum(losses), 2) if losses else None,
        "average_trade_percent": _pct(float(np.mean(net))) if net else None,
        "exposure_percent": _round(sim["bars_in_market"] / n * 100.0, 1),
        "ambiguous_bars": sim["ambiguous_bars"],
        "benchmark_buy_hold_return_percent": _pct(benchmark),
        "excess_return_vs_benchmark_percent": _pct(total - benchmark),
    }


def _eba_segment(sim: dict, frame: pd.DataFrame, lo: int, hi: int, name: str, intraday: bool) -> dict:
    metrics = _eba_metrics(sim, frame["close"].to_numpy(float)[lo:hi], frame.index[lo:hi])
    metrics["segment"] = name
    metrics["start"] = _ts_str(metrics["start"], intraday)
    metrics["end"] = _ts_str(metrics["end"], intraday)
    if metrics["profit_factor"] is None and metrics["trade_count"]:
        metrics["profit_factor_note"] = "No losing trades: profit factor is undefined."
    return metrics


def compute_ema_bollinger_atr(
    prices,
    timeframe: str = "1d",
    params: dict | None = None,
    optimize: bool = False,
    data_source: str | None = None,
) -> dict | None:
    """Indicator state + train / out-of-sample backtest. None when there is not
    enough history for the indicators and two evaluation segments."""
    if timeframe not in EBA_TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {EBA_TIMEFRAMES}")
    p = resolve_ema_bollinger_atr_params(params)
    intraday = timeframe == "1h"
    frame = _eba_clean_frame(prices)
    warmup = max(p["ema_period"], p["bb_period"], p["atr_period"] * 2, p["rsi_period"] * 2) + p["trend_lookback"]
    n = len(frame)
    if n < warmup + 2 * EBA_MIN_SEGMENT_BARS:
        return None
    split = int(n * p["train_fraction"])
    if split < EBA_MIN_SEGMENT_BARS or n - split < EBA_MIN_SEGMENT_BARS:
        return None

    ind = _eba_indicators(frame, p)
    warnings: list[str] = []

    selection = {
        "enabled": bool(optimize), "objective": "training-segment total return (min "
        f"{EBA_MIN_TRAIN_TRADES} trades)", "candidates": [], "selected": None,
        "note": "Parameters are chosen on the training segment only; the out-of-sample segment is never consulted.",
    }
    if optimize:
        for sl_mult in EBA_OPT_STOP_MULTS:
            for tp_ratio in EBA_OPT_TP_RATIOS:
                trial = {**p, "stop_loss_atr_mult": sl_mult, "take_profit_ratio": tp_ratio}
                m = _eba_segment(_eba_simulate(frame, ind, trial, 0, split, intraday), frame, 0, split, "train", intraday)
                selection["candidates"].append(
                    {
                        "stop_loss_atr_mult": sl_mult, "take_profit_ratio": tp_ratio,
                        "train_total_return_percent": m["total_return_percent"],
                        "train_trade_count": m["trade_count"],
                        "train_max_drawdown_percent": m["max_drawdown_percent"],
                    }
                )
        eligible = [x for x in selection["candidates"] if x["train_trade_count"] >= EBA_MIN_TRAIN_TRADES]
        if eligible:
            best = max(
                eligible,
                key=lambda x: (x["train_total_return_percent"], x["train_max_drawdown_percent"]),
            )
            p = {**p, "stop_loss_atr_mult": best["stop_loss_atr_mult"], "take_profit_ratio": best["take_profit_ratio"]}
            selection["selected"] = {
                "stop_loss_atr_mult": best["stop_loss_atr_mult"], "take_profit_ratio": best["take_profit_ratio"],
            }
        else:
            warnings.append(
                f"Parameter selection skipped: no combination produced >= {EBA_MIN_TRAIN_TRADES} training "
                "trades; defaults were kept."
            )

    train_sim = _eba_simulate(frame, ind, p, 0, split, intraday)
    test_sim = _eba_simulate(frame, ind, p, split, n, intraday)
    train = _eba_segment(train_sim, frame, 0, split, "train", intraday)
    test = _eba_segment(test_sim, frame, split, n, "out_of_sample", intraday)

    if test["trade_count"] < 30:
        warnings.append(
            f"Out-of-sample segment has only {test['trade_count']} trade(s); these statistics are "
            "statistically weak and should not be extrapolated."
        )
    if test["trade_count"] == 0:
        warnings.append("No setups occurred out-of-sample, so no out-of-sample performance can be assessed.")
    if intraday:
        warnings.append(
            "Hourly bars are raw (not adjusted for splits/dividends) with UTC timestamps; session coverage "
            "is as delivered by EODHD."
        )
    else:
        warnings.extend(_corporate_action_warnings(prices, "close"))
        warnings.append(
            "Daily backtest uses raw (unadjusted) OHLC like the other technical strategies; benchmark "
            "buy-and-hold is price-only (dividends excluded)."
        )

    trades_all = []
    for seg_name, sim, offset in (("train", train_sim, 0), ("out_of_sample", test_sim, split)):
        for t in sim["trades"]:
            trades_all.append({**{k: v for k, v in t.items() if k != "_net"}, "segment": seg_name})

    def curve(sim, lo, hi, name):
        closes = frame["close"].to_numpy(float)[lo:hi]
        out = []
        for k in range(hi - lo):
            out.append(
                {
                    "date": _ts_str(frame.index[lo + k], intraday), "segment": name,
                    "equity": _round(float(sim["equity"][k]), 4),
                    "benchmark": _round(float(closes[k] / closes[0]), 4),
                }
            )
        return out

    equity_curve = curve(train_sim, 0, split, "train") + curve(test_sim, split, n, "out_of_sample")
    step = max(1, math.ceil(len(equity_curve) / EBA_EQUITY_POINTS))
    equity_curve = equity_curve[::step]

    last = len(frame) - 1
    close_last = float(frame["close"].iloc[-1])
    atr_last = ind["atr"].iloc[-1]
    uptrend, downtrend = bool(ind["uptrend"].iloc[-1]), bool(ind["downtrend"].iloc[-1])
    latest = {
        "date": _ts_str(frame.index[last], intraday),
        "close": _round(close_last, 4),
        "ema": _round(ind["ema"].iloc[-1], 4),
        "bb_upper": _round(ind["bb_upper"].iloc[-1], 4),
        "bb_lower": _round(ind["bb_lower"].iloc[-1], 4),
        "rsi": _round(ind["rsi"].iloc[-1]),
        "atr": _round(atr_last, 4),
        "trend": "Uptrend" if uptrend else "Downtrend" if downtrend else "Neutral",
        "long_setup_active": bool(ind["long_setup"].iloc[-1]),
        "short_setup_active": bool(ind["short_setup"].iloc[-1]) and p["allow_short"],
        "reference_stop_distance": _round(float(atr_last) * p["stop_loss_atr_mult"], 4) if pd.notna(atr_last) else None,
        "reference_take_profit_distance": _round(
            float(atr_last) * p["stop_loss_atr_mult"] * p["take_profit_ratio"], 4
        ) if pd.notna(atr_last) else None,
        "note": "Reference distances are informational: ATR x stop multiplier, and x take-profit ratio.",
    }

    chart = frame.index[-CHART_LOOKBACK_DAYS:]
    chart_data = [
        {
            "date": _ts_str(ts, intraday),
            "close": _round(frame.at[ts, "close"], 4),
            "ema": _round(ind["ema"].at[ts], 4),
            "bb_upper": _round(ind["bb_upper"].at[ts], 4),
            "bb_lower": _round(ind["bb_lower"].at[ts], 4),
            "rsi": _round(ind["rsi"].at[ts]),
            "atr": _round(ind["atr"].at[ts], 4),
        }
        for ts in chart
    ]

    return {
        "timeframe": timeframe,
        "data_source": data_source
        or ("daily_prices (PostgreSQL, EODHD end-of-day, raw OHLC)" if not intraday else "EODHD intraday API (1h, raw)"),
        "bars_used": n,
        "first_bar": _ts_str(frame.index[0], intraday),
        "last_bar": _ts_str(frame.index[-1], intraday),
        "parameters": p,
        "latest": latest,
        "train_fraction": p["train_fraction"],
        "train": train,
        "out_of_sample": test,
        "parameter_selection": selection,
        "trade_count_total": len(trades_all),
        "trades": trades_all[-EBA_MAX_TRADES_RETURNED:],
        "equity_curve": equity_curve,
        "chart_data": chart_data,
        "assumptions": [
            "Signals use candles up to and including the signal candle's close; fills happen at the next candle's open.",
            "Trend window = the trend_lookback candles preceding the signal candle (body-above/below-EMA test).",
            "Stop/target anchored to the entry fill using the signal candle's ATR; stop assumed first if both touch in one candle.",
            f"Costs per side: commission {p['commission_percent']}% + slippage {p['slippage_percent']}% against the trade.",
            f"One position at a time, sized at {p['position_fraction'] * 100:g}% of equity; shorts assume free borrow.",
            "Training and out-of-sample segments are simulated independently, in time order.",
            "stop_loss_atr_mult and take_profit_ratio defaults are implementation choices, not values verified from the article.",
        ],
        "data_quality_warnings": warnings,
        "disclaimer": EBA_DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Strategy 32: Technical Crash-Warning Risk Indicator  (EXPERIMENTAL)
#
# Source: "This Financial API Warns You Before Stocks Crash!" (P. Saxena).
# The article describes moving averages, RSI, MACD and divergence; it does
# NOT provide a validated statistical crash model, and neither does this. The
# output is a transparent count of explicit technical conditions - there is no
# probability, no forecast and no promise that a crash will (or will not)
# follow. The article's 5-minute intraday variant is NOT implemented: this is
# the DAILY version only (raw EODHD daily OHLCV from daily_prices).
#
# Scored conditions (each true/false at every bar, using data up to that bar):
#   1 price_below_long_sma        close < SMA(200)
#   2 short_sma_below_long_sma    SMA(50) < SMA(200)  ("death-cross" state)
#   3 macd_bearish_cross_recent   MACD(12,26,9) line crossed below its signal
#                                 line within the last 10 bars and is still below
#   4 macd_histogram_weakening    MACD histogram lower for 3 consecutive bars
#   5 rsi_bearish_divergence      the last two confirmed swing highs (see below)
#                                 within 60 bars: price higher high, RSI(14) lower high
#   Swing high: a close that is the maximum of the surrounding +/-5 bars. It is
#   only CONFIRMED 5 bars later, so no future bar is used when evaluating bar t.
# Informational only (NOT scored): RSI oversold/overbought, MACD below zero,
# drawdown from the 252-bar high. An oversold RSI is ordinary market behaviour
# and is explicitly not treated as evidence of a crash.
#
# Overall: triggered_count of 5 -> Low (0-1) / Elevated (2-3) / High (4-5), and
# `warning_active` = triggered_count >= 3. These cut-offs are UNVALIDATED
# heuristics chosen for readability.
#
# Historical evaluation (time-ordered, out-of-sample): a "crash event" at bar t
# means the close falls >= 10% below close[t] at some point in the next 20
# bars. Labels use future bars ONLY to score past warnings, never to produce
# them. The first 60% of evaluable bars (minus a 20-bar purge) is the
# in-sample block, the remainder out-of-sample. Metrics: precision, recall,
# false positives, missed episodes, median lead (bars), compared with a simple
# baseline (close < SMA200) and with the base rate (always-warn precision).
# ---------------------------------------------------------------------------

CRASH_SMA_SHORT = 50
CRASH_SMA_LONG = 200
CRASH_MACD_RECENT_CROSS_BARS = 10
CRASH_HIST_DECLINE_BARS = 3
CRASH_SWING_K = 5
CRASH_DIVERGENCE_LOOKBACK = 60
CRASH_WARNING_MIN_TRIGGERED = 3
CRASH_RSI_OVERSOLD = 30.0
CRASH_RSI_OVERBOUGHT = 70.0
CRASH_DRAWDOWN_WINDOW = 252
CRASH_EVENT_THRESHOLD_PERCENT = -10.0
CRASH_EVENT_HORIZON_BARS = 20
CRASH_EVAL_TRAIN_FRACTION = 0.6
CRASH_MIN_EVAL_BARS = 100
CRASH_MIN_EPISODES_FOR_CONFIDENCE = 5
MIN_ROWS_CRASH_WARNING = CRASH_SMA_LONG + 60
CRASH_SCORED_CONDITIONS = (
    ("price_below_long_sma", "Price below 200-day SMA", f"close < SMA({CRASH_SMA_LONG})"),
    ("short_sma_below_long_sma", "50-day SMA below 200-day SMA", f"SMA({CRASH_SMA_SHORT}) < SMA({CRASH_SMA_LONG})"),
    (
        "macd_bearish_cross_recent", "MACD crossed below signal line recently",
        f"MACD line crossed below the signal line within the last {CRASH_MACD_RECENT_CROSS_BARS} bars and is still below it",
    ),
    (
        "macd_histogram_weakening", "MACD momentum weakening",
        f"MACD histogram lower than the previous bar for {CRASH_HIST_DECLINE_BARS} consecutive bars",
    ),
    (
        "rsi_bearish_divergence", "Bearish RSI divergence",
        (
            f"last two confirmed swing highs (+/-{CRASH_SWING_K} bars, confirmed {CRASH_SWING_K} bars later, "
            f"within {CRASH_DIVERGENCE_LOOKBACK} bars): price made a higher high while RSI made a lower high"
        ),
    ),
)
CRASH_DISCLAIMER = (
    "Experimental, descriptive technical-risk monitor. It counts explicit indicator conditions; it is not a "
    "crash prediction, produces no probability, and has not been shown to forecast market declines. "
    "Do not rely on it for investment decisions."
)


def _crash_frame(prices) -> pd.DataFrame:
    close = to_close_series(prices)
    close = close[close.notna() & (close > 0)]
    close = close[~close.index.duplicated(keep="last")]
    return pd.DataFrame({"close": close})


def _crash_swing_divergence(close: np.ndarray, rsi: np.ndarray) -> tuple[np.ndarray, list]:
    """Causal bearish RSI divergence flag for every bar + the (j1, j2) swing pair active at each bar."""
    n = len(close)
    k = CRASH_SWING_K
    swing = np.zeros(n, bool)
    for j in range(k, n - k):
        window = close[j - k : j + k + 1]
        swing[j] = close[j] == window.max() and np.isfinite(rsi[j])
    flag = np.zeros(n, bool)
    pair: list = [None] * n
    confirmed: list[int] = []
    for t in range(n):
        j = t - k
        if j >= k and swing[j]:
            confirmed.append(j)
        if len(confirmed) >= 2:
            j2 = confirmed[-1]
            if j2 > t - CRASH_DIVERGENCE_LOOKBACK:
                j1 = next(
                    (x for x in reversed(confirmed[:-1]) if j2 - x >= k and x > t - CRASH_DIVERGENCE_LOOKBACK),
                    None,
                )
                if j1 is not None and close[j2] > close[j1] and rsi[j2] < rsi[j1]:
                    flag[t] = True
                    pair[t] = (j1, j2)
    return flag, pair


def _crash_conditions(df: pd.DataFrame) -> dict:
    close = df["close"]
    sma_s = close.rolling(CRASH_SMA_SHORT, min_periods=CRASH_SMA_SHORT).mean()
    sma_l = close.rolling(CRASH_SMA_LONG, min_periods=CRASH_SMA_LONG).mean()
    rsi = _rsi(close, RSI_PERIOD)
    macd = _ema(close, MACD_FAST_PERIOD) - _ema(close, MACD_SLOW_PERIOD)
    signal = macd.ewm(span=MACD_SIGNAL_PERIOD, adjust=False, min_periods=MACD_SIGNAL_PERIOD).mean()
    hist = macd - signal
    valid = sma_l.notna() & signal.notna() & rsi.notna()

    below = macd < signal
    cross_down = (macd.shift(1) >= signal.shift(1)) & below
    recent_cross = cross_down.astype(float).rolling(CRASH_MACD_RECENT_CROSS_BARS).max().eq(1.0)
    weakening = pd.Series(True, index=close.index)
    for lag in range(CRASH_HIST_DECLINE_BARS):
        weakening &= hist.shift(lag) < hist.shift(lag + 1)
    div_flag, div_pair = _crash_swing_divergence(close.to_numpy(float), rsi.to_numpy(float))

    flags = {
        "price_below_long_sma": (close < sma_l) & valid,
        "short_sma_below_long_sma": (sma_s < sma_l) & valid,
        "macd_bearish_cross_recent": below & recent_cross & valid,
        "macd_histogram_weakening": weakening & valid,
        "rsi_bearish_divergence": pd.Series(div_flag, index=close.index) & valid,
    }
    count = sum(flags[cid].astype(int) for cid, _, _ in CRASH_SCORED_CONDITIONS)
    drawdown = close / close.rolling(CRASH_DRAWDOWN_WINDOW, min_periods=60).max() - 1.0
    return {
        "sma_short": sma_s, "sma_long": sma_l, "rsi": rsi, "macd": macd, "signal": signal, "hist": hist,
        "valid": valid, "flags": flags, "count": count, "divergence_pairs": div_pair, "drawdown": drawdown,
    }


def _crash_risk_level(count: int) -> str:
    return "Low" if count <= 1 else "Elevated" if count <= 3 else "High"


def _classification(label: np.ndarray, warn: np.ndarray, breach: np.ndarray, lo: int, hi: int) -> dict:
    y, w = label[lo:hi], warn[lo:hi]
    tp, fp = int((y & w).sum()), int((~y & w).sum())
    fn, tn = int((y & ~w).sum()), int((~y & ~w).sum())
    total = max(1, hi - lo)
    positives = tp + fn
    # Episodes = maximal runs of consecutive crash-labelled bars.
    episodes, detected, leads = 0, 0, []
    i = 0
    while i < len(y):
        if y[i]:
            j = i
            while j + 1 < len(y) and y[j + 1]:
                j += 1
            episodes += 1
            hits = np.where(w[i : j + 1])[0]
            if hits.size:
                detected += 1
                first = lo + i + int(hits[0])
                if np.isfinite(breach[first]):
                    leads.append(float(breach[first]))
            i = j + 1
        else:
            i += 1
    return {
        "bars": hi - lo,
        "crash_labelled_bars": positives,
        "base_rate_percent": _round(positives / total * 100.0, 1),
        "warning_bars": tp + fp,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn, "true_negatives": tn,
        "precision_percent": _round(tp / (tp + fp) * 100.0, 1) if tp + fp else None,
        "recall_percent": _round(tp / positives * 100.0, 1) if positives else None,
        "false_positive_rate_percent": _round(fp / (fp + tn) * 100.0, 1) if fp + tn else None,
        "episodes": episodes, "episodes_detected": detected, "episodes_missed": episodes - detected,
        "median_lead_bars": _round(float(np.median(leads)), 1) if leads else None,
    }


def _crash_validation(df: pd.DataFrame, cond: dict) -> dict:
    close = df["close"].to_numpy(float)
    n = len(close)
    horizon = CRASH_EVENT_HORIZON_BARS
    threshold = CRASH_EVENT_THRESHOLD_PERCENT / 100.0
    label = np.zeros(n, bool)
    breach = np.full(n, np.nan)
    for t in range(n - horizon):
        hit = np.where(close[t + 1 : t + horizon + 1] / close[t] - 1.0 <= threshold)[0]
        if hit.size:
            label[t] = True
            breach[t] = hit[0] + 1
    evaluable = np.where(cond["valid"].to_numpy(bool) & (np.arange(n) < n - horizon))[0]
    definition = {
        "crash_event": f"close falls >= {abs(CRASH_EVENT_THRESHOLD_PERCENT):g}% below the signal-bar close within "
        f"the next {horizon} bars",
        "prediction_horizon_bars": horizon,
        "warning_rule": f"triggered_count >= {CRASH_WARNING_MIN_TRIGGERED} of {len(CRASH_SCORED_CONDITIONS)}",
        "baseline_rule": f"close < SMA({CRASH_SMA_LONG}) (single condition)",
        "split": f"time-ordered: first {CRASH_EVAL_TRAIN_FRACTION:.0%} in-sample, remainder out-of-sample, "
        f"{horizon}-bar purge between them",
        "note": "Parameters are fixed in advance and were not tuned on this data; both blocks are reported "
        "for transparency.",
    }
    if len(evaluable) < CRASH_MIN_EVAL_BARS:
        return {
            "status": "unavailable",
            "reason": f"Needs at least {CRASH_MIN_EVAL_BARS} evaluable bars after indicator warm-up "
            f"(have {len(evaluable)}).",
            "definition": definition,
        }
    v0, v1 = int(evaluable[0]), int(evaluable[-1])
    split = v0 + int((v1 - v0 + 1) * CRASH_EVAL_TRAIN_FRACTION)
    warn = (cond["count"] >= CRASH_WARNING_MIN_TRIGGERED).to_numpy(bool)
    base = cond["flags"]["price_below_long_sma"].to_numpy(bool)
    blocks = {"in_sample": (v0, max(v0 + 1, split - horizon)), "out_of_sample": (split, v1 + 1)}
    out = {"status": "available", "definition": definition, "warnings": []}
    for name, (lo, hi) in blocks.items():
        strategy = _classification(label, warn, breach, lo, hi)
        baseline = _classification(label, base, breach, lo, hi)
        precision, base_rate = strategy["precision_percent"], strategy["base_rate_percent"]
        strategy["precision_lift_vs_base_rate"] = (
            _round(precision / base_rate, 2) if precision is not None and base_rate else None
        )
        out[name] = {"start": df.index[lo].date().isoformat(), "end": df.index[hi - 1].date().isoformat(),
                     "indicator": strategy, "baseline_sma200": baseline,
                     "always_warn_precision_percent": base_rate}
    oos_episodes = out["out_of_sample"]["indicator"]["episodes"]
    if oos_episodes < CRASH_MIN_EPISODES_FOR_CONFIDENCE:
        out["warnings"].append(
            f"Only {oos_episodes} independent crash episode(s) out-of-sample; precision/recall are statistically "
            "unreliable and must not be read as evidence of predictive skill."
        )
    out["warnings"].append(
        "Crash-labelled bars are heavily autocorrelated (one decline labels many consecutive bars); episode "
        "counts are the more honest sample size."
    )
    return out


def compute_crash_warning(prices) -> dict | None:
    """Daily technical-risk readout with component observations and a labelled,
    time-ordered historical evaluation. None when history is insufficient."""
    df = _crash_frame(prices)
    if len(df) < MIN_ROWS_CRASH_WARNING:
        return None
    cond = _crash_conditions(df)
    if not bool(cond["valid"].iloc[-1]):
        return None
    last = len(df) - 1
    last_ts = df.index[last]
    flags = cond["flags"]

    def run_start(cid: str) -> str | None:
        series = flags[cid].to_numpy(bool)
        if not series[last]:
            return None
        i = last
        while i - 1 >= 0 and series[i - 1]:
            i -= 1
        return df.index[i].date().isoformat()

    conditions = []
    for cid, label, rule in CRASH_SCORED_CONDITIONS:
        conditions.append(
            {
                "id": cid, "label": label, "rule": rule, "scored": True,
                "triggered": bool(flags[cid].iloc[last]),
                "triggered_since": run_start(cid),
                "observed_at": last_ts.date().isoformat(),
            }
        )
    rsi_last = float(cond["rsi"].iloc[last])
    informational = [
        {
            "id": "rsi_oversold", "label": "RSI oversold", "scored": False,
            "triggered": rsi_last < CRASH_RSI_OVERSOLD,
            "rule": f"RSI({RSI_PERIOD}) < {CRASH_RSI_OVERSOLD:g}",
            "note": "Informational only. Oversold readings are common in ordinary corrections and are not evidence of a crash.",
            "observed_at": last_ts.date().isoformat(),
        },
        {
            "id": "rsi_overbought", "label": "RSI overbought", "scored": False,
            "triggered": rsi_last > CRASH_RSI_OVERBOUGHT,
            "rule": f"RSI({RSI_PERIOD}) > {CRASH_RSI_OVERBOUGHT:g}", "note": "Informational only.",
            "observed_at": last_ts.date().isoformat(),
        },
        {
            "id": "macd_below_zero", "label": "MACD line below zero", "scored": False,
            "triggered": bool(cond["macd"].iloc[last] < 0), "rule": "MACD line < 0",
            "note": "Informational only.", "observed_at": last_ts.date().isoformat(),
        },
    ]

    divergence = None
    pair = cond["divergence_pairs"][last]
    if pair is not None:
        j1, j2 = pair
        divergence = {
            "first_swing_date": df.index[j1].date().isoformat(),
            "first_swing_close": _round(df["close"].iloc[j1]),
            "first_swing_rsi": _round(cond["rsi"].iloc[j1]),
            "second_swing_date": df.index[j2].date().isoformat(),
            "second_swing_close": _round(df["close"].iloc[j2]),
            "second_swing_rsi": _round(cond["rsi"].iloc[j2]),
            "confirmed_on": df.index[min(last, j2 + CRASH_SWING_K)].date().isoformat(),
        }

    count = int(cond["count"].iloc[last])
    tail = df.index[-CHART_LOOKBACK_DAYS:]
    chart = []
    for ts in tail:
        row = {
            "date": ts.date().isoformat(),
            "close": _round(df.at[ts, "close"]),
            "sma_short": _round(cond["sma_short"].at[ts]),
            "sma_long": _round(cond["sma_long"].at[ts]),
            "rsi": _round(cond["rsi"].at[ts]),
            "macd": _round(cond["macd"].at[ts], 3),
            "signal": _round(cond["signal"].at[ts], 3),
            "histogram": _round(cond["hist"].at[ts], 3),
            "triggered_count": int(cond["count"].at[ts]) if bool(cond["valid"].at[ts]) else None,
            "triggered": [cid for cid, _, _ in CRASH_SCORED_CONDITIONS if bool(flags[cid].at[ts])]
            if bool(cond["valid"].at[ts]) else [],
        }
        chart.append(row)

    warnings = _corporate_action_warnings(prices, "close")
    return {
        "timeframe": "1d",
        "data_basis": "daily_prices (PostgreSQL, EODHD end-of-day), raw close; daily version only - the article's "
        "5-minute intraday variant is not implemented",
        "as_of": last_ts.date().isoformat(),
        "bars_used": len(df),
        "indicators": {
            "close": _round(df["close"].iloc[last]),
            "sma_short": _round(cond["sma_short"].iloc[last]),
            "sma_long": _round(cond["sma_long"].iloc[last]),
            "rsi": _round(rsi_last),
            "macd": _round(cond["macd"].iloc[last], 3),
            "signal": _round(cond["signal"].iloc[last], 3),
            "histogram": _round(cond["hist"].iloc[last], 3),
            "drawdown_from_252d_high_percent": _pct(cond["drawdown"].iloc[last]),
        },
        "conditions": conditions,
        "informational": informational,
        "divergence": divergence,
        "triggered_count": count,
        "scored_condition_count": len(CRASH_SCORED_CONDITIONS),
        "risk_level": _crash_risk_level(count),
        "warning_active": count >= CRASH_WARNING_MIN_TRIGGERED,
        "assessment": (
            f"{count} of {len(CRASH_SCORED_CONDITIONS)} technical risk conditions are present as of "
            f"{last_ts.date().isoformat()} ({_crash_risk_level(count)} technical risk, heuristic cut-offs). "
            "This describes current indicator conditions only; it is not a forecast."
        ),
        "validation": _crash_validation(df, cond),
        "parameters": {
            "sma_short": CRASH_SMA_SHORT, "sma_long": CRASH_SMA_LONG, "rsi_period": RSI_PERIOD,
            "macd": [MACD_FAST_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD],
            "macd_recent_cross_bars": CRASH_MACD_RECENT_CROSS_BARS,
            "histogram_decline_bars": CRASH_HIST_DECLINE_BARS,
            "swing_half_window": CRASH_SWING_K, "divergence_lookback_bars": CRASH_DIVERGENCE_LOOKBACK,
            "warning_min_triggered": CRASH_WARNING_MIN_TRIGGERED,
        },
        "chart_data": chart,
        "methodology": [
            "Each condition is evaluated per bar using only data up to that bar; swing highs are confirmed 5 bars late.",
            "The overall level is a plain count of triggered conditions (unvalidated heuristic cut-offs), not a probability.",
            "Oversold RSI, overbought RSI, MACD<0 and drawdown are shown for context and are not scored.",
            "Historical evaluation labels future declines only to grade past warnings, in time order, out-of-sample.",
        ],
        "data_quality_warnings": warnings,
        "disclaimer": CRASH_DISCLAIMER,
    }
