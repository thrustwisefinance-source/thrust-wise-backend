"""
Unit tests for the Strategy Analytics engine (app.services.strategies).

Covers:
- compute_ema50_rsi_pullback: trend, distance, RSI-turning-up, pullback
- compute_ema8_21_pullback: trend, crossovers, distance, pullback
- Graceful handling of insufficient history (returns None, no crash)
"""

import datetime
from unittest.mock import MagicMock

import pytest

from app.services import strategies


def _make_price(symbol: str, date: datetime.date, close: float) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.open = close
    p.high = close
    p.low = close
    p.adjusted_close = close
    p.volume = 1_000_000
    return p


def _make_series(
    symbol: str, start: datetime.date, closes: list[float]
) -> list[MagicMock]:
    result = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price(symbol, current, close))
        current += datetime.timedelta(days=1)
    return result


class TestInsufficientHistory:
    def test_ema50_rsi_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_ema50_rsi_pullback(prices) is None

    def test_ema8_21_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 5)
        assert strategies.compute_ema8_21_pullback(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_ema50_rsi_pullback([]) is None
        assert strategies.compute_ema8_21_pullback([]) is None


class TestEma50RsiPullback:
    def _sufficient_uptrend(self) -> list[MagicMock]:
        # Steady uptrend, well beyond the 74-row minimum, so close ends up
        # above a rising EMA50 with RSI > 40 and no losses recently.
        closes = [100.0 + i * 0.5 for i in range(90)]
        return _make_series("VOO", datetime.date(2023, 1, 2), closes)

    def test_bullish_trend_and_shape(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_ema50_rsi_pullback(prices)

        assert result is not None
        assert result["trend"] == "Bullish"
        assert result["price_above_ema50"] is True
        assert result["price"] > result["ema50"]
        assert 0.0 <= result["rsi"] <= 100.0
        assert isinstance(result["pullback_active"], bool)
        assert isinstance(result["rsi_turning_up"], bool)
        assert "chart_data" in result
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS

        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "ema50", "rsi"}

    def test_distance_from_ema50_percent_formula(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_ema50_rsi_pullback(prices)
        expected = (result["price"] - result["ema50"]) / result["ema50"] * 100
        assert result["distance_from_ema50_percent"] == pytest.approx(
            expected, abs=0.01
        )

    def test_bearish_trend_when_price_below_ema50(self):
        # Sharp drop at the very end pulls the last close under a
        # slower-moving EMA50.
        closes = [100.0 + i * 0.5 for i in range(80)] + [70.0, 68.0, 65.0]
        prices = _make_series("VOO", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_ema50_rsi_pullback(prices)

        assert result is not None
        assert result["trend"] == "Bearish"
        assert result["price_above_ema50"] is False
        assert result["pullback_active"] is False  # pullback requires price > EMA50

    def test_chart_data_capped_at_180_rows(self):
        closes = [100.0 + (i % 10) * 0.1 for i in range(400)]
        prices = _make_series("VOO", datetime.date(2022, 1, 3), closes)
        result = strategies.compute_ema50_rsi_pullback(prices)
        assert len(result["chart_data"]) == strategies.CHART_LOOKBACK_DAYS


class TestEma8Ema21Pullback:
    def _sufficient_uptrend(self) -> list[MagicMock]:
        closes = [100.0 + i * 0.5 for i in range(40)]
        return _make_series("QQQ", datetime.date(2023, 1, 2), closes)

    def test_bullish_trend_and_shape(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_ema8_21_pullback(prices)

        assert result is not None
        assert result["trend"] == "Bullish"
        assert result["ema8"] > result["ema21"]
        assert isinstance(result["bullish_cross"], bool)
        assert isinstance(result["bearish_cross"], bool)
        assert isinstance(result["price_above_both_emas"], bool)

        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "ema8", "ema21"}

    def test_distance_to_ema8_percent_formula(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_ema8_21_pullback(prices)
        expected = (result["price"] - result["ema8"]) / result["ema8"] * 100
        assert result["distance_to_ema8_percent"] == pytest.approx(expected, abs=0.01)

    def test_bullish_cross_detected(self):
        # Downtrend long enough to warm up both EMAs and put EMA8 below
        # EMA21, then a sharp reversal that flips EMA8 above EMA21 on the
        # final bar.
        down = [130.0 - i * 0.8 for i in range(30)]
        up = [down[-1] + i * 3.0 for i in range(1, 6)]
        closes = down + up
        prices = _make_series("QQQ", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_ema8_21_pullback(prices)

        assert result is not None
        assert result["trend"] == "Bullish"
        assert result["bullish_cross"] is True
        assert result["bearish_cross"] is False

    def test_pullback_requires_price_above_both_emas_and_tight_distance(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_ema8_21_pullback(prices)
        if result["pullback_active"]:
            assert result["price_above_both_emas"] is True
            assert abs(result["distance_to_ema8_percent"]) <= (
                strategies.PULLBACK_THRESHOLD_EMA8_21 + 1e-6
            )


def _make_price_hlc(
    symbol: str, date: datetime.date, high: float, low: float, close: float
) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.open = close
    p.high = high
    p.low = low
    p.adjusted_close = close
    p.volume = 1_000_000
    return p


def _make_hlc_series(
    symbol: str, start: datetime.date, closes: list[float], daily_range: float = 1.0
) -> list[MagicMock]:
    """Series with a nonzero High/Low range around each close, so
    High/Low-dependent strategies (Better Breakout) get nonzero offsets."""
    result = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        high = close + daily_range
        low = close - daily_range
        result.append(_make_price_hlc(symbol, current, high, low, close))
        current += datetime.timedelta(days=1)
    return result


class TestInsufficientHistoryNewStrategies:
    def test_macd_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_macd(prices) is None

    def test_bollinger_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_bollinger_bands(prices) is None

    def test_better_breakout_returns_none_with_too_few_rows(self):
        prices = _make_hlc_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_better_breakout(prices) is None

    def test_sma_trend_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_sma_trend(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_macd([]) is None
        assert strategies.compute_bollinger_bands([]) is None
        assert strategies.compute_better_breakout([]) is None
        assert strategies.compute_sma_trend([]) is None


class TestMacd:
    def _sufficient_uptrend(self) -> list[MagicMock]:
        closes = [100.0 + i * 0.5 for i in range(60)]
        return _make_series("VOO", datetime.date(2023, 1, 2), closes)

    def test_bullish_trend_and_shape(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_macd(prices)

        assert result is not None
        assert result["trend"] == "Bullish"
        assert result["macd_line"] > result["signal_line"]
        assert isinstance(result["bullish_crossover"], bool)
        assert isinstance(result["bearish_crossover"], bool)
        assert "chart_data" in result
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS

        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "macd", "signal", "histogram"}

    def test_histogram_equals_macd_minus_signal(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_macd(prices)
        expected = result["macd_line"] - result["signal_line"]
        assert result["histogram"] == pytest.approx(expected, abs=0.01)

    def test_bullish_crossover_detected(self):
        down = [180.0 - i * 1.2 for i in range(60)]
        up = [down[-1] + i * 4.0 for i in range(1, 12)]
        closes = down + up
        prices = _make_series("VOO", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_macd(prices)

        assert result is not None
        assert result["trend"] == "Bullish"


class TestBollingerBands:
    def _sufficient_range(self) -> list[MagicMock]:
        closes = [100.0 + (i % 10) * 0.7 for i in range(40)]
        return _make_series("VOO", datetime.date(2023, 1, 2), closes)

    def test_shape_and_band_ordering(self):
        prices = self._sufficient_range()
        result = strategies.compute_bollinger_bands(prices)

        assert result is not None
        assert result["upper_band"] > result["sma20"] > result["lower_band"]
        assert result["breakout_status"] in (
            "Above Upper Band",
            "Below Lower Band",
            "Within Bands",
        )
        assert isinstance(result["price_near_upper_band"], bool)
        assert isinstance(result["price_near_lower_band"], bool)

        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "sma20", "upper_band", "lower_band"}

    def test_breakout_above_upper_band(self):
        closes = [100.0] * 30 + [100.0, 100.0, 100.0, 130.0]
        prices = _make_series("VOO", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_bollinger_bands(prices)

        assert result is not None
        assert result["price"] > result["upper_band"]
        assert result["breakout_status"] == "Above Upper Band"

    def test_band_width_percent_formula(self):
        prices = self._sufficient_range()
        result = strategies.compute_bollinger_bands(prices)
        expected = (
            (result["upper_band"] - result["lower_band"]) / result["sma20"] * 100
        )
        assert result["band_width_percent"] == pytest.approx(expected, abs=0.01)


class TestBetterBreakout:
    def _sufficient_history(self) -> list[MagicMock]:
        closes = [100.0 + i * 0.3 for i in range(60)]
        return _make_hlc_series(
            "VOO", datetime.date(2023, 1, 2), closes, daily_range=0.8
        )

    def test_shape_and_fields(self):
        prices = self._sufficient_history()
        result = strategies.compute_better_breakout(prices)

        assert result is not None
        assert result["max_offset"] >= result["high_offset"]
        assert result["max_offset"] >= result["low_offset"]
        assert isinstance(result["breakout_active"], bool)
        assert "sma40" in result
        assert "momentum" in result

        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "breakout_price", "sma40"}

    def test_momentum_equals_close_minus_sma40(self):
        prices = self._sufficient_history()
        result = strategies.compute_better_breakout(prices)
        expected = result["price"] - result["sma40"]
        assert result["momentum"] == pytest.approx(expected, abs=0.01)

    def test_breakout_active_when_price_exceeds_breakout_price(self):
        closes = [100.0] * 55 + [100.0, 100.0, 100.0, 100.0, 140.0]
        prices = _make_hlc_series(
            "VOO", datetime.date(2023, 1, 2), closes, daily_range=0.5
        )
        result = strategies.compute_better_breakout(prices)

        assert result is not None
        assert result["price"] > result["breakout_price"]
        assert result["breakout_active"] is True


class TestSmaTrend:
    def _sufficient_uptrend(self) -> list[MagicMock]:
        closes = [100.0 + i * 0.2 for i in range(220)]
        return _make_series("VOO", datetime.date(2022, 1, 3), closes)

    def test_bullish_trend_and_shape(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_sma_trend(prices)

        assert result is not None
        assert result["trend"] == "Bullish"
        assert result["price_above_sma200"] is True
        assert isinstance(result["golden_cross"], bool)
        assert isinstance(result["death_cross"], bool)

        row = result["chart_data"][-1]
        assert set(row.keys()) == {
            "date",
            "close",
            "sma20",
            "sma50",
            "sma100",
            "sma200",
        }

    def test_bearish_trend_when_price_below_all_smas(self):
        closes = [200.0 - i * 0.4 for i in range(220)]
        prices = _make_series("VOO", datetime.date(2022, 1, 3), closes)
        result = strategies.compute_sma_trend(prices)

        assert result is not None
        assert result["trend"] == "Bearish"
        assert result["price_above_sma200"] is False

    def test_chart_data_capped_at_180_rows(self):
        closes = [100.0 + (i % 10) * 0.1 for i in range(400)]
        prices = _make_series("VOO", datetime.date(2022, 1, 3), closes)
        result = strategies.compute_sma_trend(prices)
        assert len(result["chart_data"]) == strategies.CHART_LOOKBACK_DAYS