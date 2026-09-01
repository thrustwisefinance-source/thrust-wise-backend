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

import datetime
import math

import numpy as np
import pandas as pd

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
    """High/Low/Close frame indexed by date, ascending.

    Used by strategies (e.g. Better Breakout) that need the daily range,
    not just the close series.
    """
    if not prices:
        return pd.DataFrame(columns=["high", "low", "close"])
    return pd.DataFrame(
        {
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