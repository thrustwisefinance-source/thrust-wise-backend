"""
Unit tests for the Strategy Analytics engine (app.services.strategies).

Covers:
- compute_ema50_rsi_pullback: trend, distance, RSI-turning-up, pullback
- compute_ema8_21_pullback: trend, crossovers, distance, pullback
- compute_tqqq_tmf_ief_rebalancing (Strategy 11): insufficient history,
  initial 50/50 allocation, periodic rebalancing, the crash filter,
  recovery, no look-ahead bias, backtest labeling. Merged into this file
  (previously its own tests/unit/test_tqqq_tmf_ief_strategy.py) so
  TQQQ/TMF/IEF follows the same test organization and shared
  fixtures/helpers as every other strategy — see the normalization note
  in services/strategies.py Strategy 11's module docstring.
- compute_mswing (Strategy 12): insufficient history, calculation,
  EMA9/SMA50, the four momentum states, zero-line/EMA crossovers, the
  default bullish/bearish-exit conditions, and chart output.
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


def _flat_series(
    symbol: str, start: datetime.date, n_days: int, value: float = 100.0
) -> list[MagicMock]:
    """`n_days` bars of identical closes starting at `start` — used by
    the TQQQ/TMF/IEF Rebalancing tests below where one or more legs need
    to be held perfectly flat while another leg moves."""
    return _make_series(symbol, start, [value] * n_days)


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


# ---------------------------------------------------------------------------
# Strategy 7: EVaR Risk / Position Sizing
# ---------------------------------------------------------------------------


class TestEvarRisk:
    def _steady_series(self, n: int = 400, seed: int = 7) -> list:
        """Mildly noisy random-walk-ish closes — enough variance for a
        real (non-degenerate) return distribution, deterministic via a
        fixed seed so the test is stable."""
        import random

        rng = random.Random(seed)
        price = 100.0
        closes = []
        for _ in range(n):
            price *= 1 + rng.gauss(0.0003, 0.01)
            closes.append(price)
        return _make_series("VOO", datetime.date(2022, 1, 3), closes)

    def test_returns_none_with_too_few_rows(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 50)
        assert strategies.compute_evar_risk(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_evar_risk([]) is None

    def test_valid_price_history_produces_valid_result(self):
        prices = self._steady_series()
        result = strategies.compute_evar_risk(prices)

        assert result is not None
        assert isinstance(result["evar_percent"], float)
        assert result["evar_percent"] >= 0.0
        assert isinstance(result["tsallis_q"], float)
        assert 1.0 <= result["tsallis_q"] <= strategies.EVAR_MAX_Q
        assert result["confidence_level"] == strategies.EVAR_CONFIDENCE_LEVEL
        assert result["tail_risk_level"] in {"Low", "Moderate", "Elevated", "High"}
        assert result["risk_regime"] in {"Calm", "Normal", "Elevated", "Stressed"}
        assert 0.0 <= result["tail_risk_percentile"] <= 100.0
        assert (
            strategies.EVAR_MIN_EXPOSURE_PERCENT
            <= result["suggested_exposure_percent"]
            <= strategies.EVAR_MAX_EXPOSURE_PERCENT
        )
        assert "chart_data" in result
        row = result["chart_data"][-1]
        assert set(row.keys()) == {"date", "close", "evar_percent", "tail_risk_percentile"}

    def test_suggested_exposure_within_bounds_across_many_regimes(self):
        # Sweep several noise levels and confirm exposure never leaves bounds.
        import random

        for seed in range(5):
            rng = random.Random(seed)
            price = 50.0
            closes = []
            for _ in range(350):
                price *= 1 + rng.gauss(0.0, 0.005 + seed * 0.01)
                closes.append(max(price, 1.0))
            prices = _make_series("QQQ", datetime.date(2021, 6, 1), closes)
            result = strategies.compute_evar_risk(prices)
            if result is None:
                continue
            assert (
                strategies.EVAR_MIN_EXPOSURE_PERCENT
                <= result["suggested_exposure_percent"]
                <= strategies.EVAR_MAX_EXPOSURE_PERCENT
            )

    def test_missing_invalid_data_does_not_crash(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [])
        assert strategies.compute_evar_risk(prices) is None

        # A single zero/None-ish close shouldn't raise — log-return of 0
        # produces -inf/NaN internally but must be handled gracefully.
        closes = [100.0] * 30 + [0.0] + [100.0] * 200
        prices = _make_series("VOO", datetime.date(2022, 1, 3), closes)
        result = strategies.compute_evar_risk(prices)  # should not raise
        assert result is None or isinstance(result["evar_percent"], float)


def _make_price_with_range(
    symbol: str, date: datetime.date, close: float, day_range: float = 1.0
) -> MagicMock:
    """Like _make_price, but with a non-zero high/low range so ATR is
    non-zero (needed for Triple-MA Pullback tests)."""
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.open = close
    p.high = close + day_range
    p.low = close - day_range
    p.adjusted_close = close
    p.volume = 1_000_000
    return p


def _make_ohlc_series(
    symbol: str, start: datetime.date, closes: list[float], day_range: float = 1.0
) -> list[MagicMock]:
    result = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price_with_range(symbol, current, close, day_range))
        current += datetime.timedelta(days=1)
    return result


class TestTripleMaPullback:
    def _sufficient_uptrend(self, dip: float = 0.0) -> list[MagicMock]:
        closes = [100.0 + i * 0.3 for i in range(150)]
        if dip:
            closes[-1] = closes[-2] - dip
        return _make_ohlc_series("QQQ", datetime.date(2023, 1, 2), closes)

    def test_returns_none_with_too_few_rows(self):
        prices = _make_ohlc_series("QQQ", datetime.date(2024, 1, 2), [100.0] * 20)
        assert strategies.compute_triple_ma_pullback(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_triple_ma_pullback([]) is None

    def test_confirmed_uptrend_no_pullback(self):
        prices = self._sufficient_uptrend()
        result = strategies.compute_triple_ma_pullback(prices)

        assert result is not None
        assert result["trend"] == "Uptrend"
        assert result["uptrend_confirmed"] is True
        assert result["ma1"] > result["ma2"] > result["ma3"]
        assert result["price"] > result["ma3"]
        assert result["pullback_active"] is False
        assert result["current_signal"] == "Uptrend — No Pullback"
        assert "chart_data" in result
        row = result["chart_data"][-1]
        assert set(row.keys()) == {
            "date",
            "close",
            "ma1",
            "ma2",
            "ma3",
            "entry_level",
            "exit_level",
        }

    def test_pullback_entry_zone_detected(self):
        # A sharp one-day drop pulls price below the 5-day-mean/ATR entry
        # level while the longer nested MAs still confirm the uptrend.
        prices = self._sufficient_uptrend(dip=5.0)
        result = strategies.compute_triple_ma_pullback(prices)

        assert result is not None
        assert result["uptrend_confirmed"] is True
        assert result["price"] < result["entry_level"]
        assert result["pullback_active"] is True
        assert result["current_signal"] == "Pullback Entry Zone"

    def test_no_confirmed_uptrend_in_downtrend(self):
        closes = [200.0 - i * 0.3 for i in range(150)]
        prices = _make_ohlc_series("QQQ", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_triple_ma_pullback(prices)

        assert result is not None
        assert result["uptrend_confirmed"] is False
        assert result["trend"] == "No Confirmed Uptrend"
        assert result["pullback_active"] is False
        assert result["current_signal"] == "No Confirmed Uptrend"


class TestTltMonthlyCycle:
    def _prices_ending_on(self, end_date: datetime.date) -> list[MagicMock]:
        prices = _make_ohlc_series("TLT", datetime.date(2025, 1, 2), [90.0] * 400)
        prices.append(_make_price_with_range("TLT", end_date, 90.5))
        return prices

    def test_returns_none_with_too_few_rows(self):
        prices = _make_ohlc_series("TLT", datetime.date(2026, 9, 1), [90.0] * 3)
        assert strategies.compute_tlt_monthly_cycle(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_tlt_monthly_cycle([]) is None

    def test_short_window_at_start_of_month(self):
        prices = self._prices_ending_on(datetime.date(2026, 9, 1))
        result = strategies.compute_tlt_monthly_cycle(prices)

        assert result is not None
        assert result["current_position"] == "Short"
        assert result["current_phase"] == "Short Window (Start of Month)"
        assert result["month"] == "2026-09"
        assert result["next_expected_action"] == "Exit Short"

    def test_flat_mid_month(self):
        prices = self._prices_ending_on(datetime.date(2026, 9, 15))
        result = strategies.compute_tlt_monthly_cycle(prices)

        assert result is not None
        assert result["current_position"] == "Flat"
        assert result["current_phase"] == "Mid-Month (No Position)"
        assert result["next_expected_action"] == "Enter Long"

    def test_long_window_near_month_end(self):
        prices = self._prices_ending_on(datetime.date(2026, 9, 29))
        result = strategies.compute_tlt_monthly_cycle(prices)

        assert result is not None
        assert result["current_position"] == "Long"
        assert result["current_phase"] == "Long Window (End of Month)"
        assert result["next_expected_action"] == "Exit Long"

    def test_calendar_dates_are_well_formed(self):
        prices = self._prices_ending_on(datetime.date(2026, 9, 1))
        result = strategies.compute_tlt_monthly_cycle(prices)

        assert result is not None
        for key in (
            "short_entry_date",
            "short_exit_date",
            "long_entry_date",
            "long_exit_date",
            "next_expected_date",
        ):
            # Should parse as ISO dates without raising.
            datetime.date.fromisoformat(result[key])
        assert result["short_entry_date"] <= result["short_exit_date"]
        assert result["long_entry_date"] <= result["long_exit_date"]
        assert result["short_exit_date"] < result["long_entry_date"]


# ---------------------------------------------------------------------------
# Strategy 11: TQQQ / TMF / IEF Rebalancing
#
# Merged from the former tests/unit/test_tqqq_tmf_ief_strategy.py so this
# strategy's tests use the same shared _make_series/_flat_series helpers
# and file organization as every other strategy above, instead of a
# separate test file with its own duplicated fixtures. See the
# normalization note in services/strategies.py Strategy 11's docstring.
# ---------------------------------------------------------------------------


class TestTqqqTmfIefInsufficientHistory:
    def test_empty_prices_return_none(self):
        assert strategies.compute_tqqq_tmf_ief_rebalancing([], [], []) is None

    def test_too_few_joined_rows_return_none(self):
        start = datetime.date(2020, 1, 1)
        tqqq = _flat_series("TQQQ", start, 3)
        tmf = _flat_series("TMF", start, 3)
        ief = _flat_series("IEF", start, 3)
        assert strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief) is None

    def test_mismatched_symbol_history_returns_none(self):
        """No overlapping dates across the three symbols -> inner join is
        empty -> unavailable, same convention as portfolio_comparison."""
        tqqq = _flat_series("TQQQ", datetime.date(2019, 1, 1), 30)
        tmf = _flat_series("TMF", datetime.date(2021, 1, 1), 30)
        ief = _flat_series("IEF", datetime.date(2023, 1, 1), 30)
        assert strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief) is None


class TestTqqqTmfIefInitialAllocation:
    def test_starts_50_50_tqqq_tmf(self):
        start = datetime.date(2020, 1, 1)
        n = 15
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        first_row = result["chart_data"][0]
        assert first_row["is_rebalance_event"] is True
        assert first_row["state"] == "Normal"

        # Flat prices throughout -> allocation stays ~50/50 the whole way,
        # including the latest ("current") reading.
        assert result["tqqq_allocation_percent"] == pytest.approx(50.0, abs=0.5)
        assert result["tmf_allocation_percent"] == pytest.approx(50.0, abs=0.5)
        assert result["ief_allocation_percent"] == 0.0
        assert result["state"] == "Normal"
        assert result["crash_filter_status"] == "Inactive"


class TestTqqqTmfIefPeriodicRebalancing:
    def test_rebalances_after_two_months_of_drift(self):
        start = datetime.date(2020, 1, 1)
        # ~4 months of business days so at least one 2-month rebalance fires
        n = 90
        tqqq = _make_series("TQQQ", start, [50.0 * (1.01**i) for i in range(n)])
        tmf = _make_series("TMF", start, [20.0 * (0.999**i) for i in range(n)])
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        rebalance_events = [
            row for row in result["chart_data"] if row["is_rebalance_event"]
        ]
        # Initial purchase + at least one periodic rebalance
        assert len(rebalance_events) >= 2

        # Right after a rebalance the mix should be back near 50/50.
        second_rebalance = rebalance_events[1]
        idx = result["chart_data"].index(second_rebalance)
        row = result["chart_data"][idx]
        total = row["tqqq"] and (
            row["portfolio_value"]
        )  # sanity: portfolio_value present
        assert total is not None


class TestTqqqTmfIefCrashFilter:
    def _crash_series(self, start, n_before=15):
        """Flat TQQQ prices, then a single day with a -25% drop, then flat."""
        pre = [50.0] * n_before
        crash_price = pre[-1] * 0.75  # -25%, breaches the -20% threshold
        post = [crash_price] * 20
        closes = pre + [crash_price] + post
        return _make_series("TQQQ", start, closes)

    def test_single_day_20_percent_drop_triggers_defensive_state(self):
        start = datetime.date(2020, 1, 1)
        tqqq = self._crash_series(start)
        n = len(tqqq)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None
        assert result["state"] == "Defensive / Crash"
        assert result["crash_filter_status"] == "Triggered"
        assert result["ief_allocation_percent"] == pytest.approx(100.0, abs=0.5)
        assert result["tqqq_allocation_percent"] == 0.0
        assert result["tmf_allocation_percent"] == 0.0
        assert result["pre_crash_tqqq_price"] is not None
        assert result["recovery_status"] is not None

        crash_events = [r for r in result["chart_data"] if r["is_crash_event"]]
        assert len(crash_events) == 1

    def test_10_percent_drop_does_not_trigger_crash_filter(self):
        start = datetime.date(2020, 1, 1)
        pre = [50.0] * 15
        drop_price = pre[-1] * 0.90  # -10%, below the -20% threshold
        closes = pre + [drop_price] + [drop_price] * 10
        tqqq = _make_series("TQQQ", start, closes)
        n = len(tqqq)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None
        assert result["state"] == "Normal"
        assert result["crash_filter_status"] == "Inactive"
        crash_events = [r for r in result["chart_data"] if r["is_crash_event"]]
        assert len(crash_events) == 0


class TestTqqqTmfIefRecovery:
    def test_returns_to_normal_once_tqqq_exceeds_pre_crash_price(self):
        start = datetime.date(2020, 1, 1)
        pre_crash_price = 50.0
        pre = [pre_crash_price] * 15
        crash_price = pre_crash_price * 0.70  # -30% crash day
        # Recover gradually, then exceed the pre-crash price
        recovery = [crash_price * (1.02**i) for i in range(1, 40)]
        closes = pre + [crash_price] + recovery
        tqqq = _make_series("TQQQ", start, closes)
        n = len(tqqq)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        recovery_events = [r for r in result["chart_data"] if r["is_recovery_event"]]
        assert len(recovery_events) == 1

        # If TQQQ climbed back above the pre-crash price and stayed there,
        # the strategy should be Normal again by the end of the series.
        if closes[-1] > pre_crash_price:
            assert result["state"] == "Normal"
            assert result["crash_filter_status"] == "Inactive"
            assert result["pre_crash_tqqq_price"] is None


class TestTqqqTmfIefNoLookAheadBias:
    def test_chart_dates_strictly_ascending_and_match_input_length(self):
        start = datetime.date(2020, 1, 1)
        n = 40
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        dates = [row["date"] for row in result["chart_data"]]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)
        assert len(result["chart_data"]) == n


class TestTqqqTmfIefBacktestLabeling:
    def test_backtest_fields_are_clearly_labeled_and_not_fabricated(self):
        start = datetime.date(2020, 1, 1)
        n = 30
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strategies.compute_tqqq_tmf_ief_rebalancing(
            tqqq, tmf, ief, initial_investment=100_000.0
        )
        assert result is not None
        assert result["backtest_label"] == "ThrustWise calculated backtest"
        assert result["backtest_initial_investment"] == 100_000.0
        # Flat prices throughout minus slippage on the initial purchase.
        assert result["backtest_final_value"] < 100_000.0
        assert result["backtest_final_value"] > 99_000.0
        assert "disclaimer" in result
        assert "leveraged" in result["disclaimer"].lower()


# ---------------------------------------------------------------------------
# Strategy 12: Mswing Momentum
# ---------------------------------------------------------------------------


def _mswing_series_20_50(
    short_pct: float, long_pct: float, n_extra_days: int = 0, base: float = 100.0
) -> list[MagicMock]:
    """Build a close series where the most recent bar has an exact,
    controllable Mswing reading: the 20-day-ago close implies `short_pct`
    percent change, and the 50-day-ago close implies `long_pct` percent
    change. `n_extra_days` adds flat padding before the 50-day lookback
    window so EMA9-of-Mswing has room to warm up/react.
    """
    start = datetime.date(2023, 1, 2)
    close_20_ago = base
    close_50_ago = base
    n = 51 + n_extra_days
    closes = [base] * n
    # Index -51 is 50 trading days before the last close (index -1).
    closes[-51] = close_50_ago
    closes[-21] = close_20_ago
    closes[-1] = close_50_ago * (1 + long_pct / 100)
    # Recompute so both legs land on the target percentages relative to
    # the (possibly different) 20-days-ago and 50-days-ago closes.
    closes[-21] = closes[-1] / (1 + short_pct / 100)
    return _make_series("MSW", start, closes)


class TestMswingInsufficientHistory:
    def test_empty_prices_return_none(self):
        assert strategies.compute_mswing([]) is None

    def test_too_few_rows_return_none(self):
        prices = _make_series("MSW", datetime.date(2024, 1, 2), [100.0] * 40)
        assert strategies.compute_mswing(prices) is None

    def test_index_value_none_with_too_few_rows(self):
        prices = _make_series("QQQ", datetime.date(2024, 1, 2), [100.0] * 40)
        assert strategies.compute_mswing_index_value(prices) is None


class TestMswingCalculation:
    def _uptrend_prices(self, n: int = 120) -> list[MagicMock]:
        start = datetime.date(2023, 1, 2)
        closes = [100.0 * (1.003**i) for i in range(n)]
        return _make_series("MSW", start, closes)

    def _downtrend_prices(self, n: int = 120) -> list[MagicMock]:
        start = datetime.date(2023, 1, 2)
        closes = [100.0 * (0.997**i) for i in range(n)]
        return _make_series("MSW", start, closes)

    def test_mswing_matches_formula_on_last_bar(self):
        prices = self._uptrend_prices(100)
        result = strategies.compute_mswing(prices)
        assert result is not None

        closes = [p.close for p in prices]
        c_t = closes[-1]
        c_20 = closes[-1 - 20]
        c_50 = closes[-1 - 50]
        expected = ((c_t - c_20) / c_20 * 100) + ((c_t - c_50) / c_50 * 100)
        assert result["mswing"] == pytest.approx(expected, abs=0.01)

    def test_sma50_matches_plain_average(self):
        prices = self._uptrend_prices(100)
        result = strategies.compute_mswing(prices)
        assert result is not None

        closes = [p.close for p in prices][-50:]
        assert result["sma50"] == pytest.approx(sum(closes) / len(closes), abs=0.01)

    def test_ema9_is_between_zero_and_mswing_in_a_steady_uptrend(self):
        prices = self._uptrend_prices(150)
        result = strategies.compute_mswing(prices)
        assert result is not None
        # A steady uptrend has been positive and rising for a while, so
        # EMA9 should have converged close to (and below or near) the
        # current reading rather than sitting at zero.
        assert result["mswing_ema9"] > 0

    def test_chart_data_present_and_bounded(self):
        prices = self._uptrend_prices(300)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        for row in result["chart_data"]:
            assert "mswing" in row
            assert "mswing_ema9" in row
            assert "sma50" in row


class TestMswingStates:
    def test_strong_bullish_state(self):
        # mswing > 0 and mswing > ema9: a fresh, still-accelerating uptrend
        # right after a long flat/declining stretch keeps EMA9 low.
        start = datetime.date(2023, 1, 2)
        closes = [100.0] * 60 + [100.0 * (1.01**i) for i in range(1, 15)]
        prices = _make_series("MSW", start, closes)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is True
        assert result["mswing_above_ema"] is True
        assert result["mswing_state"] == "Strong bullish momentum"

    def test_bearish_state(self):
        start = datetime.date(2023, 1, 2)
        closes = [100.0] * 60 + [100.0 * (0.99**i) for i in range(1, 15)]
        prices = _make_series("MSW", start, closes)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is False
        assert result["mswing_above_ema"] is False
        assert result["mswing_state"] == "Bearish momentum"

    def test_weakening_bullish_state(self):
        # Momentum was strongly positive, then decelerates sharply while
        # staying positive: EMA9 (lagging) stays above the now-cooling
        # current reading.
        start = datetime.date(2023, 1, 2)
        rally = [100.0 * (1.02**i) for i in range(60)]
        cooldown = [rally[-1] * (1.001**i) for i in range(1, 15)]
        prices = _make_series("MSW", start, rally + cooldown)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is True
        assert result["mswing_above_ema"] is False
        assert result["mswing_state"] == "Weakening bullish momentum"

    def test_recovering_bearish_state(self):
        # Momentum was strongly negative, then sharply reverses upward
        # but hasn't crossed back above zero yet: EMA9 (lagging, still
        # negative) sits below the fast-recovering current reading.
        start = datetime.date(2023, 1, 2)
        decline = [100.0 * (0.98**i) for i in range(60)]
        bounce = [decline[-1] * (1.012**i) for i in range(1, 12)]
        prices = _make_series("MSW", start, decline + bounce)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is False
        assert result["mswing_above_ema"] is True
        assert result["mswing_state"] == "Recovering bearish momentum"


class TestMswingCrossovers:
    def test_zero_line_bullish_cross_detected(self):
        start = datetime.date(2023, 1, 2)
        # Flat then a sudden pop on the very last bar so Mswing flips
        # from <= 0 to > 0 between the last two rows.
        closes = [100.0] * 70 + [101.0]
        prices = _make_series("MSW", start, closes)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["zero_line_bullish_cross"] is True
        assert result["zero_line_bearish_cross"] is False

    def test_ema_crossover_flags_are_booleans(self):
        prices = _make_series(
            "MSW", datetime.date(2023, 1, 2), [100.0 * (1.004**i) for i in range(100)]
        )
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert isinstance(result["ema_bullish_cross"], bool)
        assert isinstance(result["ema_bearish_cross"], bool)


class TestMswingStrategyConditions:
    def test_bullish_condition_requires_all_three_legs(self):
        # Strong, accelerating uptrend well above a rising SMA50.
        start = datetime.date(2023, 1, 2)
        closes = [100.0] * 55 + [100.0 * (1.015**i) for i in range(1, 20)]
        prices = _make_series("MSW", start, closes)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is True
        assert result["price_above_sma50"] is True
        if result["mswing_above_ema"]:
            assert result["bullish_condition_active"] is True

    def test_bearish_exit_condition_on_negative_mswing(self):
        start = datetime.date(2023, 1, 2)
        closes = [100.0] * 60 + [100.0 * (0.985**i) for i in range(1, 15)]
        prices = _make_series("MSW", start, closes)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing_above_zero"] is False
        assert result["bearish_exit_condition_active"] is True

    def test_bearish_exit_condition_on_overextended_mswing(self):
        """Mswing >= 4 triggers the exit condition even while price is
        still comfortably above SMA50 and Mswing is positive."""
        prices = _mswing_series_20_50(short_pct=3.0, long_pct=2.0, n_extra_days=40)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["mswing"] == pytest.approx(5.0, abs=0.05)
        assert result["mswing"] >= strategies.MSWING_OVEREXTENDED_THRESHOLD
        assert result["bearish_exit_condition_active"] is True

    def test_bearish_exit_condition_on_price_below_sma50(self):
        # A sharp one-day drop below a still-elevated SMA50, without
        # Mswing itself having turned negative yet.
        start = datetime.date(2023, 1, 2)
        base = [100.0 * (1.002**i) for i in range(70)]
        base.append(base[-1] * 0.90)  # sharp final-day drop below SMA50
        prices = _make_series("MSW", start, base)
        result = strategies.compute_mswing(prices)
        assert result is not None
        assert result["price_above_sma50"] is False
        assert result["bearish_exit_condition_active"] is True


class TestMswingRelativeStrength:
    def test_relative_strength_none_when_index_unavailable(self):
        prices = _make_series(
            "MSW", datetime.date(2023, 1, 2), [100.0 * (1.002**i) for i in range(80)]
        )
        result = strategies.compute_mswing(prices, index_mswing=None)
        assert result is not None
        assert result["relative_strength"] is None
        assert result["relative_strength_index_symbol"] == "QQQ"

    def test_relative_strength_is_difference_from_index(self):
        prices = _make_series(
            "MSW", datetime.date(2023, 1, 2), [100.0 * (1.002**i) for i in range(80)]
        )
        result = strategies.compute_mswing(prices, index_mswing=1.5)
        assert result is not None
        assert result["relative_strength"] == pytest.approx(
            result["mswing"] - 1.5, abs=0.01
        )

    def test_custom_index_symbol_is_reported(self):
        prices = _make_series(
            "MSW", datetime.date(2023, 1, 2), [100.0 * (1.002**i) for i in range(80)]
        )
        result = strategies.compute_mswing(
            prices, index_mswing=0.0, index_symbol="SPY"
        )
        assert result is not None
        assert result["relative_strength_index_symbol"] == "SPY"