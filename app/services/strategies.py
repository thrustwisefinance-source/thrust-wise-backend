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

import math

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