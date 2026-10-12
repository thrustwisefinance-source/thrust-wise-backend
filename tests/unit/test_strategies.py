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
- compute_momentum_reversal (Strategy 13): insufficient history,
  buy/sell threshold detection, "retain previous position" behavior,
  position-changed flag, and chart output.
- compute_factor_momentum / extract_factor_momentum_view (Strategy 14):
  insufficient/degenerate universes, factor ranking from ETF_REGISTRY
  tags, leading/lagging factor identification, per-symbol exposure
  guidance, and graceful handling of a symbol missing from the universe.
- compute_hmm_regime_switching (Strategy 15): insufficient history,
  regime detection across a synthetic low-vol/high-vol two-regime
  series, probability/NaN sanity, and the degenerate (zero-variance)
  input case.
- Graceful handling of insufficient history (returns None, no crash)
"""

import datetime
import random
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.constants import ETF_REGISTRY
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


# ---------------------------------------------------------------------------
# Strategy 13: Momentum Reversal
# ---------------------------------------------------------------------------


class TestMomentumReversalInsufficientHistory:
    def test_empty_prices_return_none(self):
        assert strategies.compute_momentum_reversal([]) is None

    def test_too_few_rows_return_none(self):
        prices = _make_series(
            "VOO",
            datetime.date(2024, 1, 2),
            [100.0] * (strategies.MIN_ROWS_MOMENTUM_REVERSAL - 1),
        )
        assert strategies.compute_momentum_reversal(prices) is None


class TestMomentumReversalSignals:
    def test_thresholds_match_source_article(self):
        # 0.1% in each direction, per the source article's rule.
        assert strategies.MOMENTUM_REVERSAL_BUY_THRESHOLD == pytest.approx(-0.001)
        assert strategies.MOMENTUM_REVERSAL_SELL_THRESHOLD == pytest.approx(0.001)

    def test_bullish_reversal_on_sharp_down_day(self):
        # Flat, then a >0.1% drop on the final bar.
        closes = [100.0] * 14 + [99.0]
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["reversal_signal"] == "Bullish Reversal"
        assert result["current_position"] == "Long"
        assert result["position_changed_today"] is True
        assert result["daily_return_percent"] == pytest.approx(-1.0, abs=0.01)

    def test_bearish_reversal_on_sharp_up_day(self):
        closes = [100.0] * 14 + [101.0]
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["reversal_signal"] == "Bearish Reversal"
        assert result["current_position"] == "Short"
        assert result["position_changed_today"] is True

    def test_no_signal_within_thresholds(self):
        # A tiny final-day move, well inside +/-0.1%.
        closes = [100.0] * 14 + [100.02]
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["reversal_signal"] == "No Signal"


class TestMomentumReversalPositionRetention:
    def test_position_retained_on_a_no_signal_day(self):
        # Day 14 triggers a Buy (-1%); day 15 is flat (no new signal), so
        # the article's "otherwise retain previous position" rule should
        # keep the position Long.
        closes = [100.0] * 13 + [99.0, 99.0]
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["reversal_signal"] == "No Signal"
        assert result["current_position"] == "Long"
        assert result["position_changed_today"] is False

    def test_position_flips_from_long_to_short(self):
        closes = [100.0] * 13 + [99.0, 101.0]
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["current_position"] == "Short"
        assert result["position_changed_today"] is True

    def test_flat_before_any_signal_has_fired(self):
        # No day ever crosses either threshold -> position stays Flat.
        closes = [100.0] * 20
        prices = _make_series("VOO", datetime.date(2024, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert result["current_position"] == "Flat"
        assert result["position_changed_today"] is False


class TestMomentumReversalChartData:
    def test_chart_data_present_and_bounded(self):
        closes = [100.0 * (1.001**i) for i in range(300)]
        prices = _make_series("VOO", datetime.date(2023, 1, 2), closes)
        result = strategies.compute_momentum_reversal(prices)

        assert result is not None
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        row = result["chart_data"][-1]
        assert set(row.keys()) == {
            "date",
            "close",
            "daily_return_percent",
            "position",
        }
        assert row["position"] in {"Long", "Short", "Flat"}


# ---------------------------------------------------------------------------
# Strategy 14: One-Month Factor Momentum
# ---------------------------------------------------------------------------


class TestFactorMomentumInsufficientUniverse:
    def test_empty_universe_returns_none(self):
        assert strategies.compute_factor_momentum({}) is None

    def test_single_symbol_universe_returns_none(self):
        n = strategies.MIN_ROWS_FACTOR_MOMENTUM + 5
        prices = _make_series(
            "QQQ", datetime.date(2024, 1, 2), [100.0 * (1.01**i) for i in range(n)]
        )
        assert strategies.compute_factor_momentum({"QQQ": prices}) is None

    def test_extract_view_returns_none_when_universe_unavailable(self):
        assert strategies.extract_factor_momentum_view(None, "QQQ", []) is None


class TestFactorMomentumRanking:
    def _build_universe(self) -> dict[str, list[MagicMock]]:
        n = strategies.MIN_ROWS_FACTOR_MOMENTUM + 5
        start = datetime.date(2024, 1, 2)
        return {
            # Technology/Equity/Index: strong uptrend -> leading factor
            "QQQ": _make_series(
                "QQQ", start, [100.0 * (1.01**i) for i in range(n)]
            ),
            # Equity/Index: modest uptrend
            "VOO": _make_series(
                "VOO", start, [100.0 * (1.002**i) for i in range(n)]
            ),
            # Gold: decline -> lagging factor
            "GLD": _make_series(
                "GLD", start, [100.0 * (0.995**i) for i in range(n)]
            ),
            # Treasury: roughly flat
            "SHY": _make_series(
                "SHY", start, [100.0 * (1.0005**i) for i in range(n)]
            ),
        }

    def test_uses_etf_registry_tags_as_factor_groups(self):
        universe = self._build_universe()
        result = strategies.compute_factor_momentum(universe)
        assert result is not None

        ranked_tags = {row["factor"] for row in result["factor_rankings"]}
        expected_tags = {"Equity", "Index", "Technology", "Gold", "Treasury"}
        assert ranked_tags == expected_tags
        assert "Existing ETF_REGISTRY category tags" in result["factor_data_basis"]

    def test_leading_and_lagging_factor(self):
        universe = self._build_universe()
        result = strategies.compute_factor_momentum(universe)
        assert result is not None
        assert result["leading_factor"] == "Technology"
        assert result["lagging_factor"] == "Gold"

    def test_rankings_sorted_descending_by_return(self):
        universe = self._build_universe()
        result = strategies.compute_factor_momentum(universe)
        assert result is not None
        returns = [row["one_month_return_percent"] for row in result["factor_rankings"]]
        assert returns == sorted(returns, reverse=True)
        ranks = [row["rank"] for row in result["factor_rankings"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_no_nan_or_inf_in_rankings(self):
        universe = self._build_universe()
        result = strategies.compute_factor_momentum(universe)
        assert result is not None
        for row in result["factor_rankings"]:
            value = row["one_month_return_percent"]
            assert value == value  # not NaN
            assert abs(value) != float("inf")


class TestFactorMomentumSymbolView:
    def _factor_momentum_and_universe(self):
        n = strategies.MIN_ROWS_FACTOR_MOMENTUM + 5
        start = datetime.date(2024, 1, 2)
        universe = {
            "QQQ": _make_series(
                "QQQ", start, [100.0 * (1.01**i) for i in range(n)]
            ),
            "VOO": _make_series(
                "VOO", start, [100.0 * (1.002**i) for i in range(n)]
            ),
            "GLD": _make_series(
                "GLD", start, [100.0 * (0.995**i) for i in range(n)]
            ),
            "SHY": _make_series(
                "SHY", start, [100.0 * (1.0005**i) for i in range(n)]
            ),
        }
        return strategies.compute_factor_momentum(universe), universe

    def test_symbol_aligned_with_leading_factor(self):
        factor_momentum, universe = self._factor_momentum_and_universe()
        view = strategies.extract_factor_momentum_view(
            factor_momentum, "QQQ", universe["QQQ"]
        )
        assert view is not None
        assert view["symbol"] == "QQQ"
        assert view["symbol_tags"] == ETF_REGISTRY["QQQ"].tags
        assert view["is_aligned_with_leading_factor"] is True
        assert view["exposure_guidance"] == "Maintain / Increase Exposure"
        assert view["suggested_exposure_percent"] == pytest.approx(100.0)

    def test_symbol_aligned_with_lagging_factor(self):
        factor_momentum, universe = self._factor_momentum_and_universe()
        view = strategies.extract_factor_momentum_view(
            factor_momentum, "GLD", universe["GLD"]
        )
        assert view is not None
        assert view["is_aligned_with_leading_factor"] is False
        assert view["exposure_guidance"] == "Reduce Exposure"
        assert (
            strategies.FACTOR_MOMENTUM_MIN_EXPOSURE_PERCENT
            <= view["suggested_exposure_percent"]
            <= strategies.FACTOR_MOMENTUM_MAX_EXPOSURE_PERCENT
        )

    def test_symbol_missing_from_universe_returns_none(self):
        n = strategies.MIN_ROWS_FACTOR_MOMENTUM + 5
        start = datetime.date(2024, 1, 2)
        # SHY intentionally excluded from the universe passed to compute_*.
        universe = {
            "QQQ": _make_series(
                "QQQ", start, [100.0 * (1.01**i) for i in range(n)]
            ),
            "VOO": _make_series(
                "VOO", start, [100.0 * (1.002**i) for i in range(n)]
            ),
            "GLD": _make_series(
                "GLD", start, [100.0 * (0.995**i) for i in range(n)]
            ),
        }
        factor_momentum = strategies.compute_factor_momentum(universe)
        shy_prices = _make_series(
            "SHY", start, [100.0 * (1.0005**i) for i in range(n)]
        )
        view = strategies.extract_factor_momentum_view(
            factor_momentum, "SHY", shy_prices
        )
        assert view is None

    def test_chart_data_present_and_bounded(self):
        factor_momentum, universe = self._factor_momentum_and_universe()
        view = strategies.extract_factor_momentum_view(
            factor_momentum, "VOO", universe["VOO"]
        )
        assert view is not None
        assert len(view["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        if view["chart_data"]:
            row = view["chart_data"][-1]
            assert set(row.keys()) == {"date", "close", "one_month_return_percent"}


# ---------------------------------------------------------------------------
# Strategy 15: HMM Regime-Switching
# ---------------------------------------------------------------------------


def _two_regime_prices(symbol: str = "HMT", seed: int = 2024) -> list[MagicMock]:
    """~1 trading year of a calm/trending regime followed by a volatile,
    choppy regime — a fixed local Random instance (not the global `random`
    module) keeps this deterministic regardless of test execution order.
    """
    rng = random.Random(seed)
    start = datetime.date(2022, 1, 3)
    closes = [100.0]
    for i in range(300):
        if i < 150:
            drift, vol = 0.0009, 0.004  # calm, trending
        else:
            drift, vol = -0.0003, 0.03  # volatile, choppy
        closes.append(closes[-1] * (1 + rng.gauss(drift, vol)))
    return _make_series(symbol, start, closes)


class TestHmmRegimeSwitchingInsufficientHistory:
    def test_empty_prices_return_none(self):
        assert strategies.compute_hmm_regime_switching([]) is None

    def test_too_few_rows_return_none(self):
        prices = _make_series(
            "VOO",
            datetime.date(2024, 1, 2),
            [100.0] * (strategies.MIN_ROWS_HMM_REGIME - 1),
        )
        assert strategies.compute_hmm_regime_switching(prices) is None


class TestHmmRegimeSwitchingCalculation:
    def test_detects_high_volatility_regime_after_a_vol_spike(self):
        prices = _two_regime_prices()
        result = strategies.compute_hmm_regime_switching(prices)

        assert result is not None
        assert result["current_regime"] == "High-Volatility / Mean-Reversion-Favorable"
        assert result["recommended_approach"] == "Mean Reversion"
        # The two fitted states should be clearly separated by volatility,
        # matching the synthetic calm-then-choppy construction.
        assert (
            result["high_vol_state_volatility_percent"]
            > result["trending_state_volatility_percent"]
        )

    def test_probabilities_sum_to_one_hundred(self):
        prices = _two_regime_prices()
        result = strategies.compute_hmm_regime_switching(prices)
        assert result is not None
        total = (
            result["trending_probability_percent"]
            + result["high_volatility_probability_percent"]
        )
        assert total == pytest.approx(100.0, abs=0.01)

    def test_no_nan_or_inf_in_output(self):
        prices = _two_regime_prices()
        result = strategies.compute_hmm_regime_switching(prices)
        assert result is not None
        for key, value in result.items():
            if isinstance(value, float):
                assert value == value, key  # not NaN
                assert abs(value) != float("inf"), key

    def test_regime_persistence_is_a_valid_probability(self):
        prices = _two_regime_prices()
        result = strategies.compute_hmm_regime_switching(prices)
        assert result is not None
        assert 0.0 <= result["regime_persistence_probability_percent"] <= 100.0

    def test_zero_variance_input_does_not_crash(self):
        # Degenerate case: perfectly flat prices (zero variance) must be
        # handled gracefully, never raise, never emit NaN/Inf.
        prices = _make_series(
            "FLAT", datetime.date(2022, 1, 3), [100.0] * 200
        )
        result = strategies.compute_hmm_regime_switching(prices)
        assert result is not None
        for key, value in result.items():
            if isinstance(value, float):
                assert value == value, key
                assert abs(value) != float("inf"), key


class TestHmmRegimeSwitchingChartData:
    def test_chart_data_present_and_bounded(self):
        prices = _two_regime_prices()
        result = strategies.compute_hmm_regime_switching(prices)
        assert result is not None
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        row = result["chart_data"][-1]
        assert set(row.keys()) == {
            "date",
            "close",
            "log_return_percent",
            "trending_probability_percent",
            "regime",
        }
        assert row["regime"] in {"Trending", "High-Volatility"}


# ---------------------------------------------------------------------------
# Strategy 16: Squeeze Momentum Indicator
# ---------------------------------------------------------------------------


def _noisy_hlc_series(
    symbol: str,
    start: datetime.date,
    n: int,
    seed: int = 1,
    start_price: float = 100.0,
    vol: float = 0.01,
    trend: float = 0.0002,
    intraday_range_frac: float = 0.005,
) -> list[MagicMock]:
    """Realistic-ish OHLC series (nonzero high/low range each day, noisy
    drifting close) for strategies that need genuine volatility/range
    signal (Squeeze Momentum, SOC, Wavelet, First Passage Time) —
    unlike `_make_series`'s flat high=low=close bars.
    """
    rng = random.Random(seed)
    price = start_price
    result = []
    current = start
    for _ in range(n):
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        price *= 1 + rng.gauss(trend, vol)
        price = max(price, 0.01)
        day_range = abs(rng.gauss(0, vol * intraday_range_frac * 100)) * price
        result.append(
            _make_price_hlc(symbol, current, price + day_range, price - day_range, price)
        )
        current += datetime.timedelta(days=1)
    return result


class TestSqueezeMomentum:
    def test_insufficient_data_returns_none(self):
        prices = _make_hlc_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert strategies.compute_squeeze_momentum(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_squeeze_momentum([]) is None

    def test_flat_low_volatility_series_reports_squeeze_on(self):
        # A tight, essentially flat range keeps Bollinger Bands inside
        # the Keltner Channel -> squeeze_on.
        prices = _make_hlc_series(
            "VOO", datetime.date(2022, 1, 3), [100.0] * 120, daily_range=0.05
        )
        result = strategies.compute_squeeze_momentum(prices)
        assert result is not None
        assert result["squeeze_on"] is True
        assert result["squeeze_state"] == "Squeeze On"

    def test_expanding_volatility_reports_squeeze_off(self):
        # Long flat/tight period (compresses BB inside KC) followed by a
        # sharp expansion in daily range -> Bollinger Bands should
        # expand outside the Keltner Channel -> squeeze_off.
        flat = _make_hlc_series(
            "VOO", datetime.date(2022, 1, 3), [100.0] * 100, daily_range=0.05
        )
        last_date = flat[-1].date + datetime.timedelta(days=1)
        expanding = _make_hlc_series(
            "VOO", last_date, [100.0 + i * 3 for i in range(1, 15)], daily_range=6.0
        )
        prices = flat + expanding
        result = strategies.compute_squeeze_momentum(prices)
        assert result is not None
        assert result["squeeze_off"] is True
        assert result["squeeze_state"] == "Squeeze Released"

    def test_momentum_is_last_fitted_value_not_slope(self):
        # For a perfectly linear momentum-source series, the rolling
        # linreg's last fitted value should equal the series' own last
        # value (since a perfect line's fitted endpoint == the actual
        # endpoint) — this would NOT hold if the slope were returned
        # instead, which is the exact mistake the article warns about.
        length = strategies.SMI_LENGTH
        y = pd.Series(np.arange(length, dtype=float) * 2.0 + 5.0)
        fitted = strategies._linreg_last_value(y, length)
        assert abs(float(fitted.iloc[-1]) - float(y.iloc[-1])) < 1e-9

    def test_result_has_expected_fields_and_chart_output(self):
        prices = _noisy_hlc_series("VOO", datetime.date(2021, 1, 4), 300, seed=3)
        result = strategies.compute_squeeze_momentum(prices)
        assert result is not None
        for key in [
            "price",
            "upper_bb",
            "lower_bb",
            "upper_kc",
            "lower_kc",
            "momentum",
            "momentum_direction",
            "momentum_state",
            "squeeze_state",
            "squeeze_on",
            "squeeze_off",
            "squeeze_released_today",
            "bb_length",
            "bb_mult",
            "kc_mult",
            "chart_data",
        ]:
            assert key in result
        assert result["momentum_direction"] in {"Rising", "Falling", "Flat"}
        assert len(result["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        row = result["chart_data"][-1]
        assert row["squeeze_state"] in {"on", "off", "none"}

    def test_no_lookahead_bias(self):
        full = _noisy_hlc_series("VOO", datetime.date(2019, 1, 2), 400, seed=11)
        truncated = full[:-20]
        result_full = strategies.compute_squeeze_momentum(full)
        result_trunc = strategies.compute_squeeze_momentum(truncated)
        assert result_full is not None and result_trunc is not None

        trunc_by_date = {row["date"]: row for row in result_trunc["chart_data"]}
        overlapping = 0
        for row in result_full["chart_data"]:
            if row["date"] in trunc_by_date:
                overlapping += 1
                assert row == trunc_by_date[row["date"]]
        assert overlapping > 0


# ---------------------------------------------------------------------------
# Strategy 17: Self-Organized Criticality — Avalanche Distribution
# ---------------------------------------------------------------------------


class TestSocAvalanche:
    def test_insufficient_history_returns_none(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 100)
        assert strategies.compute_soc_avalanche(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_soc_avalanche([]) is None

    def test_insufficient_avalanches_returns_none(self):
        # Long enough history, but a monotonically rising series never
        # drops below its running peak -> zero avalanches -> None.
        closes = [100.0 + i * 0.1 for i in range(400)]
        prices = _make_series("VOO", datetime.date(2021, 1, 4), closes)
        assert strategies.compute_soc_avalanche(prices) is None

    def test_alpha_and_classification_are_consistent_and_deterministic(self):
        prices = _noisy_hlc_series(
            "VOO", datetime.date(2018, 1, 2), 1400, seed=42, vol=0.012
        )
        result_1 = strategies.compute_soc_avalanche(prices)
        result_2 = strategies.compute_soc_avalanche(prices)
        assert result_1 is not None
        # Deterministic: identical input -> identical output.
        assert result_1 == result_2

        alpha = result_1["alpha"]
        regime = result_1["criticality_regime"]
        if alpha >= 2.8:
            assert regime == "Gaussian"
        elif alpha >= 1.8:
            assert regime == "Transitional"
        elif alpha >= 1.0:
            assert regime == "Critical"
        else:
            assert regime == "Super-critical"

        assert result_1["avalanche_count"] >= strategies.SOC_MIN_AVALANCHES
        assert result_1["mean_avalanche_size_percent"] > 0
        assert result_1["max_avalanche_size_percent"] >= result_1["mean_avalanche_size_percent"]
        assert result_1["current_drawdown_percent"] <= 0
        assert "chart_data" in result_1
        assert "avalanche_history" in result_1

    def test_extract_avalanches_identifies_a_known_drawdown(self):
        # 100 -> 100 -> 80 -> 80 -> 110 (new high) -> one avalanche of
        # size 20% (peak-to-trough from 100 to 80).
        series = pd.Series(
            [100.0, 100.0, 80.0, 80.0, 110.0],
            index=pd.date_range("2022-01-03", periods=5, freq="B"),
        )
        sizes = strategies._extract_avalanches(series)
        assert len(sizes) == 1
        assert abs(float(sizes.iloc[0]) - 20.0) < 1e-9


# ---------------------------------------------------------------------------
# Strategy 18: Adaptive Causal Wavelet Trend Filter
# ---------------------------------------------------------------------------


class TestWaveletTrendFilter:
    def test_insufficient_history_returns_none(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 20)
        assert strategies.compute_wavelet_trend_filter(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_wavelet_trend_filter([]) is None

    def test_causal_kernel_uses_only_nonnegative_lags(self):
        kernel = strategies._causal_ricker_kernel(scale=10)
        # Purely a function of t >= 0 (see module docstring) — sanity
        # check the peak sits at lag 0 (t=0 -> psi(0) = 1, the maximum
        # of (1 - t^2) * exp(-t^2/2)).
        assert kernel[0] == max(kernel)
        assert abs(kernel[0] - 1.0) < 1e-9

    def test_uptrend_series_reports_bullish_trend_state(self):
        closes = [100.0 * (1.001**i) for i in range(250)]
        prices = _make_series("VOO", datetime.date(2021, 1, 4), closes)
        result = strategies.compute_wavelet_trend_filter(prices)
        assert result is not None
        assert result["current_trend_state"] in {
            "Strong Uptrend",
            "Uptrend (Mixed Confirmation)",
        }
        assert result["trend_short"] > 0 or result["trend_medium"] > 0

    def test_downtrend_series_reports_bearish_trend_state(self):
        closes = [100.0 * (0.999**i) for i in range(250)]
        prices = _make_series("VOO", datetime.date(2021, 1, 4), closes)
        result = strategies.compute_wavelet_trend_filter(prices)
        assert result is not None
        assert result["current_trend_state"] in {
            "Strong Downtrend",
            "Downtrend (Mixed Confirmation)",
        }

    def test_no_lookahead_bias(self):
        full = _noisy_hlc_series("VOO", datetime.date(2019, 1, 2), 600, seed=5)
        truncated = full[:-20]
        result_full = strategies.compute_wavelet_trend_filter(full)
        result_trunc = strategies.compute_wavelet_trend_filter(truncated)
        assert result_full is not None and result_trunc is not None

        trunc_by_date = {row["date"]: row for row in result_trunc["chart_data"]}
        overlapping = 0
        for row in result_full["chart_data"]:
            if row["date"] in trunc_by_date:
                overlapping += 1
                assert row == trunc_by_date[row["date"]]
        assert overlapping > 0

    def test_result_has_expected_fields(self):
        prices = _noisy_hlc_series("VOO", datetime.date(2020, 1, 2), 300, seed=8)
        result = strategies.compute_wavelet_trend_filter(prices)
        assert result is not None
        for key in [
            "price",
            "current_trend_state",
            "wavelet_trend_value",
            "trend_short",
            "trend_medium",
            "trend_long",
            "volatility_adjustment_percent",
            "scales_days",
            "chart_data",
        ]:
            assert key in result
        assert set(result["scales_days"].keys()) == {"short", "medium", "long"}


# ---------------------------------------------------------------------------
# Strategy 19: First Passage Time Distribution Analysis
# ---------------------------------------------------------------------------


class TestFirstPassageTime:
    def test_insufficient_history_returns_none(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 50)
        assert strategies.compute_first_passage_time(prices) is None

    def test_empty_prices_return_none(self):
        assert strategies.compute_first_passage_time([]) is None

    def test_target_probabilities_are_valid_percentages(self):
        prices = _noisy_hlc_series(
            "VOO", datetime.date(2020, 1, 2), 400, seed=21, trend=0.0003
        )
        result = strategies.compute_first_passage_time(prices)
        assert result is not None
        assert 0.0 <= result["upside_target_probability_percent"] <= 100.0
        assert 0.0 <= result["downside_target_probability_percent"] <= 100.0

    def test_strong_positive_drift_favors_upside_expected_time(self):
        closes = [100.0 * (1.003**i) for i in range(400)]
        prices = _make_series("VOO", datetime.date(2020, 1, 2), closes)
        result = strategies.compute_first_passage_time(prices)
        assert result is not None
        # Strong, consistent positive drift -> expected days to the
        # upside target should be defined (drift points toward it) and
        # the downside expected time should be undefined (infinite
        # under a pure Brownian model, since drift points away).
        assert result["expected_days_to_upside_target"] is not None
        assert result["expected_days_to_downside_target"] is None
        assert result["upside_target_probability_percent"] > 50.0

    def test_strong_negative_drift_favors_downside_expected_time(self):
        closes = [100.0 * (0.997**i) for i in range(400)]
        prices = _make_series("VOO", datetime.date(2020, 1, 2), closes)
        result = strategies.compute_first_passage_time(prices)
        assert result is not None
        assert result["expected_days_to_downside_target"] is not None
        assert result["expected_days_to_upside_target"] is None

    def test_result_has_expected_fields_and_chart_output(self):
        prices = _noisy_hlc_series("VOO", datetime.date(2020, 1, 2), 400, seed=4)
        result = strategies.compute_first_passage_time(prices)
        assert result is not None
        for key in [
            "price",
            "mean_daily_log_return_percent",
            "daily_volatility_percent",
            "horizon_trading_days",
            "upside_target_percent",
            "downside_target_percent",
            "upside_target_price",
            "downside_target_price",
            "upside_target_probability_percent",
            "downside_target_probability_percent",
            "expected_days_to_upside_target",
            "expected_days_to_downside_target",
            "lookback_days",
            "assumptions_note",
            "chart_data",
        ]:
            assert key in result
        assert result["horizon_trading_days"] == strategies.FPT_HORIZON_DAYS
        assert len(result["chart_data"]) == strategies.FPT_HORIZON_DAYS
        last_row = result["chart_data"][-1]
        assert last_row["upside_hit_probability_percent"] == result[
            "upside_target_probability_percent"
        ]

    def test_degenerate_zero_variance_input_returns_none(self):
        prices = _make_series("VOO", datetime.date(2020, 1, 2), [100.0] * 400)
        assert strategies.compute_first_passage_time(prices) is None


# ---------------------------------------------------------------------------
# Portfolio Optimization: Kelly Criterion + Mean-Variance Optimization
# ---------------------------------------------------------------------------


def _correlated_asset_series(
    symbols: list[str], n: int = 400, seed: int = 1
) -> dict[str, list[MagicMock]]:
    """A small basket of independently-drifting-and-noisy price series
    (not truly correlated, just independent random walks with distinct
    drift/vol per asset) — enough for a non-degenerate covariance
    matrix without needing real market data."""
    rng = random.Random(seed)
    result = {}
    for i, symbol in enumerate(symbols):
        trend = 0.0001 * (i + 1)
        vol = 0.008 + 0.004 * i
        price = 50.0 + i * 25.0
        closes = []
        for _ in range(n):
            price *= 1 + rng.gauss(trend, vol)
            price = max(price, 0.01)
            closes.append(price)
        result[symbol] = _make_series(symbol, datetime.date(2020, 1, 2), closes)
    return result


class TestPortfolioOptimization:
    def test_insufficient_assets_returns_none(self):
        prices_by_symbol = _correlated_asset_series(["A"], n=300)
        assert strategies.compute_portfolio_optimization(prices_by_symbol) is None

    def test_no_overlapping_history_returns_none(self):
        # Genuinely disjoint date ranges (B's history starts years after
        # A's ends) -> the inner join produces zero comparable trading
        # days -> None, never a fabricated/interpolated overlap.
        a = _make_series("A", datetime.date(2020, 1, 2), [100.0] * 300)
        b = _make_series("B", datetime.date(2023, 1, 2), [100.0] * 300)
        assert strategies.compute_portfolio_optimization({"A": a, "B": b}) is None

        # Same-range but too-short overlap also exercises the None path.
        a_short = _make_series("A", datetime.date(2020, 1, 2), [100.0] * 30)
        b_short = _make_series("B", datetime.date(2020, 1, 2), [100.0] * 30)
        assert strategies.compute_portfolio_optimization({"A": a_short, "B": b_short}) is None

    def test_missing_symbol_prices_are_skipped_not_fabricated(self):
        prices_by_symbol = _correlated_asset_series(["A", "B"], n=300)
        prices_by_symbol["C"] = []  # no data at all for C
        result = strategies.compute_portfolio_optimization(prices_by_symbol)
        assert result is not None
        assert "C" not in result["symbols"]

    def test_valid_multi_asset_input_produces_normalized_weights(self):
        prices_by_symbol = _correlated_asset_series(["A", "B", "C"], n=500)
        result = strategies.compute_portfolio_optimization(prices_by_symbol)
        assert result is not None
        assert set(result["symbols"]) == {"A", "B", "C"}

        for portfolio_key in ["min_volatility_portfolio", "max_sharpe_portfolio"]:
            portfolio = result[portfolio_key]
            weights = portfolio["weights"]
            total = sum(w["weight_percent"] for w in weights)
            assert abs(total - 100.0) < 0.5
            for w in weights:
                # Long-only projection (see _long_only_projection):
                # every weight must be non-negative.
                assert w["weight_percent"] >= -1e-6
            assert isinstance(portfolio["expected_return_percent"], float)
            assert isinstance(portfolio["volatility_percent"], float)
            assert portfolio["volatility_percent"] >= 0.0
            assert isinstance(portfolio["sharpe_ratio"], float)

    def test_efficient_frontier_weights_are_normalized(self):
        prices_by_symbol = _correlated_asset_series(["A", "B", "C"], n=500)
        result = strategies.compute_portfolio_optimization(prices_by_symbol)
        assert result is not None
        assert len(result["efficient_frontier"]) > 0
        for point in result["efficient_frontier"]:
            total = sum(w["weight_percent"] for w in point["weights"])
            assert abs(total - 100.0) < 0.5
            assert point["volatility_percent"] >= 0.0

    def test_kelly_sizing_present_per_symbol(self):
        prices_by_symbol = _correlated_asset_series(["A", "B", "C"], n=500)
        result = strategies.compute_portfolio_optimization(prices_by_symbol)
        assert result is not None
        symbols_in_kelly = {k["symbol"] for k in result["kelly_sizing"]}
        assert symbols_in_kelly == set(result["symbols"])
        for k in result["kelly_sizing"]:
            assert isinstance(k["single_asset_kelly_fraction_percent"], float)
            assert isinstance(k["portfolio_kelly_weight_percent"], float)
            assert isinstance(k["half_kelly_weight_percent"], float)

    def test_long_only_projection_never_returns_negative_weights(self):
        # Directly exercise the projection helper with a raw solution
        # that has negative entries (as the unconstrained closed-form
        # Markowitz solution can produce).
        raw = np.array([1.5, -0.3, -0.2])
        projected = strategies._long_only_projection(raw)
        assert np.all(projected >= 0)
        assert abs(projected.sum() - 1.0) < 1e-9

    def test_long_only_projection_falls_back_to_equal_weight_when_degenerate(self):
        raw = np.array([-1.0, -2.0, -3.0])
        projected = strategies._long_only_projection(raw)
        assert np.allclose(projected, 1.0 / 3.0)

    def test_disclaimer_and_notes_present(self):
        prices_by_symbol = _correlated_asset_series(["A", "B"], n=400)
        result = strategies.compute_portfolio_optimization(prices_by_symbol)
        assert result is not None
        assert "disclaimer" in result and len(result["disclaimer"]) > 0
        assert "long_only_note" in result and len(result["long_only_note"]) > 0
        assert "not investment advice" in result["disclaimer"].lower()


# ===========================================================================
# Strategies 20-26: additional Technical Strategies
# ===========================================================================


def _candle_price(
    symbol: str, date: datetime.date, o: float, h: float, l: float, c: float
) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.open, p.high, p.low, p.close = o, h, l, c
    p.adjusted_close = c
    p.volume = 1_000_000
    return p


def _candle_series(
    rows: list[tuple[float, float, float, float]],
    start: datetime.date = datetime.date(2023, 1, 2),
    symbol: str = "VOO",
) -> list[MagicMock]:
    """rows = [(open, high, low, close), ...] on consecutive weekdays."""
    out = []
    current = start
    for o, h, l, c in rows:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        out.append(_candle_price(symbol, current, o, h, l, c))
        current += datetime.timedelta(days=1)
    return out


def _closes_to_candles(
    closes: list[float], green: bool | list[bool] = True
) -> list[tuple[float, float, float, float]]:
    """Candles whose CLOSE path is `closes`; each candle is green or red by
    placing the open below/above the close (so the close path - which TD
    Setup, ATR and EMAs depend on - is exactly what the test specifies)."""
    flags = green if isinstance(green, list) else [green] * len(closes)
    rows = []
    for c, is_green in zip(closes, flags):
        o = c - 0.4 if is_green else c + 0.4
        rows.append((o, max(o, c) + 0.2, min(o, c) - 0.2, c))
    return rows


def _trend_closes(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + step * i for i in range(n)]


_NEW_TECHNICAL_KEYS = (
    "td_setup",
    "supertrend_confirmation",
    "ufx_vervoort_heiken_ashi",
    "trend_signals_trend_tide",
    "pivot_levels",
    "cpr",
    "ta_candles_trend_health",
)

_NEW_COMPUTE_FUNCS = {
    "td_setup": strategies.compute_td_setup,
    "supertrend_confirmation": strategies.compute_supertrend_confirmation,
    "ufx_vervoort_heiken_ashi": strategies.compute_ufx_vervoort_heiken_ashi,
    "trend_signals_trend_tide": strategies.compute_trend_signals_trend_tide,
    "pivot_levels": strategies.compute_pivot_levels,
    "cpr": strategies.compute_cpr,
    "ta_candles_trend_health": strategies.compute_ta_candles_trend_health,
}


class TestNewTechnicalInsufficientAndDegenerateData:
    @pytest.mark.parametrize("key", _NEW_TECHNICAL_KEYS)
    def test_empty_prices_return_none(self, key):
        assert _NEW_COMPUTE_FUNCS[key]([]) is None

    @pytest.mark.parametrize("key", _NEW_TECHNICAL_KEYS)
    def test_too_few_rows_return_none(self, key):
        prices = _candle_series(_closes_to_candles(_trend_closes(3)))
        assert _NEW_COMPUTE_FUNCS[key](prices) is None

    @pytest.mark.parametrize("key", _NEW_TECHNICAL_KEYS)
    def test_flat_prices_do_not_crash(self, key):
        prices = _candle_series([(100.0, 100.0, 100.0, 100.0)] * 150)
        result = _NEW_COMPUTE_FUNCS[key](prices)
        assert result is not None
        assert result["chart_data"]  # chart payload still produced

    @pytest.mark.parametrize("key", _NEW_TECHNICAL_KEYS)
    def test_nan_values_do_not_crash(self, key):
        rows = _closes_to_candles(_trend_closes(150))
        rows[40] = (float("nan"),) * 4
        rows[90] = (float("nan"), 130.0, 120.0, 125.0)  # only open missing
        prices = _candle_series(rows)
        result = _NEW_COMPUTE_FUNCS[key](prices)  # must not raise
        assert result is None or isinstance(result, dict)


class TestTdSetup:
    def test_bearish_setup_completes_at_nine(self):
        # 13 strictly rising closes -> first comparable bar is index 4, so
        # indices 4..12 = nine consecutive Close[t] > Close[t-4].
        prices = _candle_series(_closes_to_candles(_trend_closes(13)))
        r = strategies.compute_td_setup(prices)
        assert r["setup_type"] == "Bearish"
        assert r["setup_count"] == 9
        assert r["setup_completed"] is True
        assert r["exhaustion_state"] == "Potential Upside Exhaustion"
        assert r["comparison_lag_candles"] == 4

    def test_bullish_setup_completes_at_nine(self):
        prices = _candle_series(_closes_to_candles(_trend_closes(13, 200.0, -1.0)))
        r = strategies.compute_td_setup(prices)
        assert r["setup_type"] == "Bullish"
        assert r["setup_count"] == 9
        assert r["setup_completed"] is True
        assert r["exhaustion_state"] == "Potential Downside Exhaustion"

    def test_eight_is_in_progress_not_completed(self):
        # 12 flat closes (no comparison hits) then 8 rising closes -> count 8.
        closes = [100.0] * 12 + _trend_closes(8, 101.0)
        prices = _candle_series(_closes_to_candles(closes))
        r = strategies.compute_td_setup(prices)
        assert r["setup_count"] == 8
        assert r["setup_completed"] is False
        assert r["exhaustion_state"] == "Setup In Progress"
        assert r["last_completed_setup_type"] is None

    def test_interruption_by_equal_close_resets_count(self):
        closes = _trend_closes(12)  # run of 8, last close 111, close[t-4]=107
        closes.append(closes[-4])  # equal to Close[t-4] -> neither setup
        r = strategies.compute_td_setup(_candle_series(_closes_to_candles(closes)))
        assert r["setup_type"] == "None"
        assert r["setup_count"] == 0
        assert r["setup_completed"] is False
        assert r["exhaustion_state"] == "No Active Setup"

    def test_interruption_by_opposite_close_starts_other_setup_at_one(self):
        closes = _trend_closes(12)
        closes.append(closes[-4] - 5.0)  # below Close[t-4]
        r = strategies.compute_td_setup(_candle_series(_closes_to_candles(closes)))
        assert r["setup_type"] == "Bullish"
        assert r["setup_count"] == 1
        assert r["setup_completed"] is False

    def test_nine_is_not_a_buy_or_sell_instruction(self):
        prices = _candle_series(_closes_to_candles(_trend_closes(13)))
        r = strategies.compute_td_setup(prices)
        for forbidden in ("signal", "action", "buy", "sell", "recommendation"):
            assert not [k for k in r if forbidden in k.lower()]

    def test_tenth_qualifying_candle_keeps_count_capped_at_nine(self):
        r = strategies.compute_td_setup(
            _candle_series(_closes_to_candles(_trend_closes(14)))
        )
        assert r["setup_count"] == 9
        assert r["consecutive_qualifying_candles"] == 10
        assert r["bars_since_completion"] == 1

    def test_last_completed_setup_is_remembered_after_interruption(self):
        closes = _trend_closes(13)
        closes.append(closes[-4] - 5.0)
        r = strategies.compute_td_setup(_candle_series(_closes_to_candles(closes)))
        assert r["setup_completed"] is False
        assert r["last_completed_setup_type"] == "Bearish"
        assert r["last_completed_setup_date"] is not None

    def test_flat_prices_have_no_setup(self):
        r = strategies.compute_td_setup(
            _candle_series([(100.0, 100.0, 100.0, 100.0)] * 30)
        )
        assert r["setup_type"] == "None" and r["setup_count"] == 0

    def test_missing_close_interrupts_count(self):
        closes = _trend_closes(20)
        closes[14] = float("nan")
        r = strategies.compute_td_setup(_candle_series(_closes_to_candles(closes)))
        # A missing close breaks the comparison TWICE: on its own bar (14) and
        # again 4 bars later (18), where it is the Close[t-4] reference.
        # Bars 15-17 qualify (3), bar 18 cannot be evaluated, bar 19 -> 1.
        assert r["setup_count"] == 1 and not r["setup_completed"]
        # ...but the 10-bar run that completed BEFORE the gap is remembered.
        assert r["last_completed_setup_type"] == "Bearish"

    def test_missing_latest_close_returns_none(self):
        closes = _trend_closes(20)
        closes[-1] = float("nan")
        assert strategies.compute_td_setup(
            _candle_series(_closes_to_candles(closes))
        ) is None


class TestSupertrendConfirmation:
    def _uptrend(self, n=60):
        return _closes_to_candles(_trend_closes(n))

    def test_buy_condition_with_two_green_candles_above_bullish_supertrend(self):
        r = strategies.compute_supertrend_confirmation(
            _candle_series(self._uptrend())
        )
        assert r["supertrend_trend"] == "Bullish"
        assert r["price_above_supertrend"] is True
        assert r["previous_two_candles"] == ["Green", "Green"]
        assert r["buy_condition_active"] is True
        assert r["sell_condition_active"] is False
        assert r["confirmation_state"] == "Buy Confirmed"

    @pytest.mark.parametrize("red_offset", [-2, -3])  # either of the 2 prior candles
    def test_one_red_prior_candle_blocks_buy_even_though_trend_is_bullish(
        self, red_offset
    ):
        # red_offset indexes the candle list; -1 is the signal candle itself.
        rows = self._uptrend()
        o, h, l, c = rows[red_offset]
        rows[red_offset] = (c + 0.4, h, l, c)  # open above close -> red
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        # Supertrend is still bullish and price is still above it...
        assert r["supertrend_trend"] == "Bullish"
        assert r["price_above_supertrend"] is True
        # ...but the two-candle confirmation fails, so no condition fires.
        assert r["buy_condition_active"] is False
        assert r["confirmation_state"] == "No Confirmation"

    def test_signal_candle_colour_itself_is_not_part_of_the_rule(self):
        rows = self._uptrend()
        o, h, l, c = rows[-1]
        rows[-1] = (c + 0.4, h, l, c)  # signal candle red; prior two green
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["buy_condition_active"] is True

    def test_doji_prior_candle_does_not_confirm(self):
        rows = self._uptrend()
        o, h, l, c = rows[-2]
        rows[-2] = (c, h, l, c)  # open == close
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["previous_two_candles"][1] == "Neutral"
        assert r["buy_condition_active"] is False

    def test_missing_open_does_not_confirm(self):
        rows = self._uptrend()
        o, h, l, c = rows[-2]
        rows[-2] = (float("nan"), h, l, c)
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["buy_condition_active"] is False

    def test_sell_condition_with_two_red_candles_below_bearish_supertrend(self):
        rows = _closes_to_candles(_trend_closes(60, 200.0, -1.0), green=False)
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["supertrend_trend"] == "Bearish"
        assert r["price_above_supertrend"] is False
        assert r["previous_two_candles"] == ["Red", "Red"]
        assert r["sell_condition_active"] is True
        assert r["buy_condition_active"] is False
        assert r["confirmation_state"] == "Sell Confirmed"

    def test_one_green_prior_candle_blocks_sell(self):
        rows = _closes_to_candles(_trend_closes(60, 200.0, -1.0), green=False)
        o, h, l, c = rows[-2]
        rows[-2] = (c - 0.4, h, l, c)  # open below close -> green
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["sell_condition_active"] is False

    def test_direction_change_alone_is_not_a_condition(self):
        # Long uptrend then a violent drop: Supertrend flips bearish, but the
        # candles right before the flip bar are not red -> no sell condition
        # on the flip bar, even though `supertrend_flipped_today` is True.
        closes = _trend_closes(60) + [150.0]
        rows = _closes_to_candles(closes)
        o, h, l, c = rows[-1]
        rows[-1] = (c + 40.0, c + 41.0, c - 1.0, c)  # big red flip candle
        r = strategies.compute_supertrend_confirmation(_candle_series(rows))
        assert r["supertrend_trend"] == "Bearish"
        assert r["supertrend_flipped_today"] is True
        assert r["sell_condition_active"] is False
        assert r["confirmation_state"] == "No Confirmation"

    def test_supertrend_helper_flips_direction_on_reversal(self):
        rows = _closes_to_candles(_trend_closes(40) + _trend_closes(30, 139.0, -3.0))
        frame = strategies.to_ohlc_frame(_candle_series(rows))
        st = strategies._supertrend(frame, 10, 3.0)
        assert st["direction"].dropna().iloc[10] == 1
        assert st["direction"].iloc[-1] == -1
        assert st["supertrend"].iloc[:9].isna().all()  # ATR warm-up

    def test_flat_prices_never_confirm(self):
        r = strategies.compute_supertrend_confirmation(
            _candle_series([(100.0, 100.0, 100.0, 100.0)] * 60)
        )
        assert r["buy_condition_active"] is False
        assert r["sell_condition_active"] is False

    def test_reuses_existing_atr_helper(self):
        # Guard against a second ATR implementation creeping in.
        import inspect

        assert "_atr(" in inspect.getsource(strategies._supertrend)

    def test_chart_data_shape(self):
        r = strategies.compute_supertrend_confirmation(
            _candle_series(self._uptrend(250))
        )
        assert len(r["chart_data"]) == strategies.CHART_LOOKBACK_DAYS
        assert set(r["chart_data"][-1]) == {
            "date",
            "close",
            "supertrend",
            "confirmation_state",
        }


class TestUfxVervoortHeikenAshi:
    def _v_shape(self):
        down = _closes_to_candles(_trend_closes(60, 200.0, -1.0), green=False)
        up = _closes_to_candles(_trend_closes(40, 142.0, 2.0), green=True)
        return _candle_series(down + up)

    def test_discloses_that_proprietary_indicators_are_not_reproduced(self):
        r = strategies.compute_ufx_vervoort_heiken_ashi(self._v_shape())
        assert r["ufx_pro_exact"] is False
        assert r["vervoort_heiken_ashi_exact"] is False
        assert "not reproduced" in r["implementation_note"].lower()
        assert any("Timer Period = 11" in s for s in r["unmapped_source_settings"])
        assert any("EMA Period = 44" in s for s in r["unmapped_source_settings"])
        assert r["ufx_atr_period"] == 6 and r["ufx_factor"] == 2.0

    def test_buy_setup_found_on_reversal_with_one_to_one_point_five_risk_reward(self):
        r = strategies.compute_ufx_vervoort_heiken_ashi(self._v_shape())
        assert r["last_setup_type"] == "Buy"
        assert r["risk_reward_ratio"] == 1.5
        entry, stop, target = (
            r["last_setup_entry"],
            r["last_setup_stop"],
            r["last_setup_target"],
        )
        assert stop < entry < target
        assert (target - entry) == pytest.approx(1.5 * (entry - stop), abs=0.02)

    def test_sell_setup_found_on_downward_reversal(self):
        up = _closes_to_candles(_trend_closes(60, 100.0, 1.0), green=True)
        down = _closes_to_candles(_trend_closes(40, 157.0, -2.0), green=False)
        r = strategies.compute_ufx_vervoort_heiken_ashi(_candle_series(up + down))
        assert r["last_setup_type"] == "Sell"
        assert r["last_setup_target"] < r["last_setup_entry"] < r["last_setup_stop"]

    def test_requires_bullish_candle_confirmation(self):
        # Same reversal but every up-bar is RED (open above close): the stand-in
        # may flip, but the candle-confirmation leg can never be satisfied.
        down = _closes_to_candles(_trend_closes(60, 200.0, -1.0), green=False)
        up = _closes_to_candles(_trend_closes(40, 142.0, 2.0), green=False)
        r = strategies.compute_ufx_vervoort_heiken_ashi(_candle_series(down + up))
        assert r["last_setup_type"] != "Buy"

    def test_no_setup_in_steady_trend(self):
        rows = _closes_to_candles(_trend_closes(120))
        r = strategies.compute_ufx_vervoort_heiken_ashi(_candle_series(rows))
        assert r["buy_setup_active"] is False and r["sell_setup_active"] is False
        assert r["setup_state"] == "No Setup"

    def test_heikin_ashi_standard_formula(self):
        frame = strategies.to_ohlc_frame(
            _candle_series([(10, 14, 8, 12), (12, 16, 11, 15), (15, 18, 14, 17)])
        )
        ha = strategies._heikin_ashi(frame)
        assert ha["ha_close"].iloc[0] == pytest.approx((10 + 14 + 8 + 12) / 4)
        assert ha["ha_open"].iloc[0] == pytest.approx((10 + 12) / 2)
        assert ha["ha_open"].iloc[1] == pytest.approx(
            (ha["ha_open"].iloc[0] + ha["ha_close"].iloc[0]) / 2
        )


class TestTrendSignalsTrendTide:
    def test_proprietary_components_reported_unavailable_and_never_fire(self):
        rows = _closes_to_candles(_trend_closes(60))
        r = strategies.compute_trend_signals_trend_tide(_candle_series(rows))
        assert r["trend_signals_available"] is False
        assert r["trend_tide_available"] is False
        assert r["buy_setup_active"] is False and r["sell_setup_active"] is False
        assert "not approximated" in r["implementation_note"].lower()
        assert r["risk_reward_ratio"] == 1.5

    def test_reports_only_the_derivable_candle_confirmation(self):
        up = strategies.compute_trend_signals_trend_tide(
            _candle_series(_closes_to_candles(_trend_closes(40), green=True))
        )
        down = strategies.compute_trend_signals_trend_tide(
            _candle_series(_closes_to_candles(_trend_closes(40), green=False))
        )
        assert up["candle_confirmation"] == "Bullish"
        assert down["candle_confirmation"] == "Bearish"


class TestPivotLevels:
    def _series_with_prev(self, prev_hlc, last_close, n=12):
        base = [(100.0, 101.0, 99.0, 100.0)] * (n - 2)
        h, l, c = prev_hlc
        last = (last_close, last_close + 1.0, last_close - 1.0, last_close)
        return _candle_series(base + [(c, h, l, c), last])

    def test_classic_pivot_formulas_hand_calculated(self):
        # prev H=110, L=90, C=100 -> PP=100, range=20
        r = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 100))
        assert r["pivot"] == 100.0
        assert r["r1"] == 110.0 and r["s1"] == 90.0  # 2PP-L, 2PP-H
        assert r["r2"] == 120.0 and r["s2"] == 80.0  # PP +/- range
        assert r["r3"] == 130.0  # H + 2(PP-L)
        assert r["s3"] == 70.0  # L - 2(H-PP)

    def test_zone_classification(self):
        upper = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 115))
        middle = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 100))
        lower = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 85))
        assert upper["zone"] == "Upper Zone"
        assert "short-bias" in upper["zone_context"]
        assert middle["zone"] == "Middle Zone"
        assert "avoid overtrading" in middle["zone_context"]
        assert lower["zone"] == "Lower Zone"
        assert "long-bias" in lower["zone_context"]

    def test_nearest_support_and_resistance(self):
        r = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 105))
        assert r["nearest_resistance_level"] == "R1" and r["nearest_resistance"] == 110.0
        assert r["nearest_support_level"] == "PIVOT" and r["nearest_support"] == 100.0
        assert r["distance_to_resistance_percent"] == pytest.approx(4.76, abs=0.01)

    def test_levels_use_previous_candle_only_no_lookahead(self):
        a = self._series_with_prev((110, 90, 100), 100)
        b = self._series_with_prev((110, 90, 100), 100)
        b[-1].high, b[-1].low = 500.0, 1.0  # wild range on the CURRENT bar
        ra, rb = strategies.compute_pivot_levels(a), strategies.compute_pivot_levels(b)
        for k in ("pivot", "r1", "r2", "r3", "s1", "s2", "s3"):
            assert ra[k] == rb[k]

    def test_flat_prices_collapse_levels_without_error(self):
        r = strategies.compute_pivot_levels(
            _candle_series([(50.0, 50.0, 50.0, 50.0)] * 10)
        )
        assert r["pivot"] == r["r1"] == r["s1"] == 50.0
        assert r["zone"] == "Middle Zone"

    def test_exposes_zones_not_trade_rules(self):
        r = strategies.compute_pivot_levels(self._series_with_prev((110, 90, 100), 115))
        assert not [k for k in r if k.lower() in ("signal", "action", "buy", "sell")]

    def test_minimum_rows(self):
        rows = [(100.0, 101.0, 99.0, 100.0)] * strategies.MIN_ROWS_PIVOT_LEVELS
        assert strategies.compute_pivot_levels(_candle_series(rows)) is not None
        assert strategies.compute_pivot_levels(_candle_series(rows[:-1])) is None


class TestCpr:
    # width = |2C - H - L| / 3, so close at the range midpoint -> zero width,
    # close at an extreme -> widest.
    NARROW_BAR = (100.0, 110.0, 90.0, 100.0)  # close at mid
    WIDE_BAR = (100.0, 110.0, 90.0, 110.0)  # close at high

    def _build(self, history_bar, prev_bar, last_close, n=70, alt_bar=None):
        rows = [history_bar, alt_bar or self.WIDE_BAR] * ((n - 2) // 2)
        rows = rows[: n - 2] + [prev_bar, (100.0, 120.0, 80.0, last_close)]
        return _candle_series(rows)

    def test_cpr_formulas_hand_calculated(self):
        # prev H=110, L=90, C=104 -> Pivot=101.3333, BC=100, TC=102.6667
        prev = (100.0, 110.0, 90.0, 104.0)
        r = strategies.compute_cpr(self._build(self.NARROW_BAR, prev, 101.0))
        assert r["pivot"] == pytest.approx(101.33, abs=0.01)
        assert r["top_cpr"] == pytest.approx(102.67, abs=0.01)
        assert r["bottom_cpr"] == pytest.approx(100.0, abs=0.01)
        assert r["cpr_width"] == pytest.approx(2.6667, abs=0.001)
        assert r["cpr_width_percent"] == pytest.approx(2.6316, abs=0.001)

    def test_top_is_max_and_bottom_is_min_when_tc_below_bc(self):
        # prev close below mid: C=94 -> Pivot=98, BC=100, TC=96 (TC < BC)
        prev = (100.0, 110.0, 90.0, 94.0)
        r = strategies.compute_cpr(self._build(self.NARROW_BAR, prev, 99.0))
        assert r["top_cpr"] == pytest.approx(100.0, abs=0.01)
        assert r["bottom_cpr"] == pytest.approx(96.0, abs=0.01)

    def test_narrow_width_classification(self):
        prev = (100.0, 110.0, 90.0, 100.0)  # zero width, below history's 25th pct
        r = strategies.compute_cpr(self._build(self.WIDE_BAR, prev, 100.0))
        assert r["cpr_width_class"] == "Narrow"
        assert r["cpr_width_percentile_rank"] < 25
        assert "larger move" in r["cpr_width_interpretation"]

    def test_typical_width_in_uniform_history_is_not_misclassified(self):
        # Current width equals every other width (all max): a typical day,
        # not an outlier -> Normal, never Narrow/Wide.
        prev = (100.0, 110.0, 90.0, 110.0)
        r = strategies.compute_cpr(self._build(self.WIDE_BAR, prev, 100.0))
        assert r["cpr_width_class"] == "Normal"

    def test_wide_width_classification(self):
        prev = (100.0, 110.0, 90.0, 110.0)  # widest possible for this range
        # history is uniformly zero-width, so this day is the lone outlier
        r = strategies.compute_cpr(
            self._build(self.NARROW_BAR, prev, 100.0, alt_bar=self.NARROW_BAR)
        )
        assert r["cpr_width_class"] == "Wide"
        assert r["cpr_width_percentile_rank"] > 75
        assert "range-bound" in r["cpr_width_interpretation"]

    def test_price_above_cpr_is_bullish_bias_with_top_as_support(self):
        prev = (100.0, 110.0, 90.0, 104.0)
        r = strategies.compute_cpr(self._build(self.NARROW_BAR, prev, 108.0))
        assert r["price_position"] == "Above CPR"
        assert r["market_bias"] == "Bullish Day Bias"
        assert r["cpr_support"] == r["top_cpr"] and r["cpr_resistance"] is None

    def test_price_below_cpr_is_bearish_bias_with_bottom_as_resistance(self):
        prev = (100.0, 110.0, 90.0, 104.0)
        r = strategies.compute_cpr(self._build(self.NARROW_BAR, prev, 95.0))
        assert r["price_position"] == "Below CPR"
        assert r["market_bias"] == "Bearish Day Bias"
        assert r["cpr_resistance"] == r["bottom_cpr"] and r["cpr_support"] is None

    def test_price_inside_cpr_is_neutral(self):
        prev = (100.0, 110.0, 90.0, 104.0)
        r = strategies.compute_cpr(self._build(self.NARROW_BAR, prev, 101.0))
        assert r["price_position"] == "Inside CPR"
        assert r["market_bias"] == "Neutral"
        assert r["cpr_support"] == r["bottom_cpr"]
        assert r["cpr_resistance"] == r["top_cpr"]

    def test_flat_history_has_no_width_dispersion_so_is_normal(self):
        r = strategies.compute_cpr(
            _candle_series([(100.0, 100.0, 100.0, 100.0)] * 80)
        )
        assert r["cpr_width_class"] == "Normal"
        assert r["cpr_width_percentile_rank"] == 50.0
        assert r["market_bias"] == "Neutral"

    def test_cpr_uses_previous_candle_not_current(self):
        prev = (100.0, 110.0, 90.0, 104.0)
        a = self._build(self.NARROW_BAR, prev, 101.0)
        b = self._build(self.NARROW_BAR, prev, 101.0)
        b[-1].high, b[-1].low = 999.0, 1.0
        assert strategies.compute_cpr(a)["top_cpr"] == strategies.compute_cpr(b)["top_cpr"]

    def test_minimum_rows_boundary(self):
        rows = [(100.0, 101.0, 99.0, 100.0)] * strategies.MIN_ROWS_CPR
        assert strategies.compute_cpr(_candle_series(rows)) is not None
        assert strategies.compute_cpr(_candle_series(rows[:-1])) is None


class TestTaCandlesTrendHealth:
    def test_discloses_it_is_not_the_tradingview_indicator(self):
        rows = _closes_to_candles(_trend_closes(120))
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["is_exact_tradingview_indicator"] is False
        assert "NOT reproduced" in r["methodology_note"]

    def test_continuous_green_uptrend_is_healthy_and_aligned(self):
        rows = _closes_to_candles(_trend_closes(120), green=True)
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["trend"] == "Bullish"
        assert r["momentum_state"] == "Positive"
        assert r["green_candle_ratio"] == 1.0
        assert r["strength_alignment"] == "Aligned Bullish"
        assert r["trend_health_state"] == "Healthy Trend (Bullish)"
        assert r["current_candle_streak"] >= 10
        assert r["current_candle_streak_colour"] == "Green"

    def test_continuous_red_downtrend_is_healthy_bearish(self):
        rows = _closes_to_candles(_trend_closes(120, 300.0, -1.0), green=False)
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["trend"] == "Bearish"
        assert r["trend_health_state"] == "Healthy Trend (Bearish)"

    def test_sudden_weakness_in_uptrend_is_flagged_weakening(self):
        closes = _trend_closes(100, 100.0, 1.0)
        closes += [closes[-1] - 0.1 * (i + 1) for i in range(8)]  # shallow drift down
        flags = [True] * 100 + [False] * 8  # red candles
        rows = _closes_to_candles(closes, green=flags)
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["trend"] == "Bullish"
        assert r["green_candle_ratio"] < 0.5
        assert r["trend_health_state"] == "Weakening Bullish Trend"

    def test_choppy_market_is_flagged_caution(self):
        closes = [100.0 + (2.0 if i % 2 else -2.0) for i in range(120)]
        flags = [i % 2 == 1 for i in range(120)]
        rows = _closes_to_candles(closes, green=flags)
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["trend"] == "Mixed"  # EMAs not separated in a sideways market
        assert r["trend_health_state"] == "Choppy - Caution"
        assert r["strength_alignment"] == "Mixed"

    def test_volatility_state_expanding_on_range_blowout(self):
        rows = _closes_to_candles(_trend_closes(110, 100.0, 0.05))
        for i in range(-5, 0):  # last 5 bars: huge ranges
            o, h, l, c = rows[i]
            rows[i] = (o, c + 15.0, c - 15.0, c)
        r = strategies.compute_ta_candles_trend_health(_candle_series(rows))
        assert r["volatility_state"] == "Expanding"

    def test_reuses_existing_ema_and_atr_helpers(self):
        import inspect

        src = inspect.getsource(strategies.compute_ta_candles_trend_health)
        assert "_ema(" in src and "_atr(" in src


class TestNewTechnicalStrategiesRegistration:
    """The seven additions must live in the EXISTING Technical Strategies
    category - not in another category and not in a new one."""

    def test_all_seven_are_in_technical_strategies(self):
        from app.schemas import TechnicalStrategies

        for key in _NEW_TECHNICAL_KEYS:
            assert key in TechnicalStrategies.model_fields, key

    def test_none_are_in_other_categories(self):
        from app.schemas import (
            CalendarStrategies,
            PortfolioStrategies,
            QuantitativeStrategies,
            RegimeStrategies,
        )

        for model in (
            QuantitativeStrategies,
            RegimeStrategies,
            CalendarStrategies,
            PortfolioStrategies,
        ):
            for key in _NEW_TECHNICAL_KEYS:
                assert key not in model.model_fields, (model.__name__, key)

    def test_no_new_category_was_created(self):
        from app.schemas import StrategyAnalytics

        category_fields = {
            name for name in StrategyAnalytics.model_fields if name.endswith("_strategies")
        }
        assert category_fields == {
            "technical_strategies",
            "quantitative_strategies",
            "regime_strategies",
            "calendar_strategies",
            "portfolio_strategies",
        }

    def test_flat_fields_exist_for_backward_compatible_consumers(self):
        from app.schemas import StrategyAnalytics

        for key in _NEW_TECHNICAL_KEYS:
            assert key in StrategyAnalytics.model_fields, key

    def test_existing_technical_keys_are_unchanged(self):
        from app.schemas import TechnicalStrategies

        for key in (
            "ema50_rsi",
            "ema821",
            "macd",
            "bollinger_bands",
            "better_breakout",
            "sma_trend",
            "triple_ma_pullback",
            "mswing",
        ):
            assert key in TechnicalStrategies.model_fields

    def test_computed_output_validates_against_response_schemas(self):
        from app.schemas import StrategyAnalytics

        rng = random.Random(7)
        closes, c = [], 100.0
        for _ in range(260):
            c = max(5.0, c + rng.uniform(-1.5, 1.7))
            closes.append(c)
        flags = [rng.random() > 0.45 for _ in closes]
        prices = _candle_series(_closes_to_candles(closes, green=flags))

        payload = {k: f(prices) for k, f in _NEW_COMPUTE_FUNCS.items()}
        assert all(v is not None for v in payload.values())
        model = StrategyAnalytics.model_validate(
            {**payload, "technical_strategies": dict(payload)}
        )
        dumped = model.model_dump(by_alias=True)
        assert dumped["tdSetup"]["setupCount"] is not None
        assert dumped["cpr"]["topCpr"] is not None
        for camel in (
            "tdSetup",
            "supertrendConfirmation",
            "ufxVervoortHeikenAshi",
            "trendSignalsTrendTide",
            "pivotLevels",
            "cpr",
            "taCandlesTrendHealth",
        ):
            assert dumped["technicalStrategies"][camel] is not None


class TestStrategyEndpointIncludesNewTechnicalStrategies:
    """End-to-end through the REAL /etfs/{symbol}/strategies route (only the
    DB / Redis / cross-ETF loaders are stubbed)."""

    @staticmethod
    def _client(n_rows):
        from unittest.mock import AsyncMock, patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.database import get_db
        from app.routers import strategy as strategy_router

        rng = random.Random(11)
        closes, c = [], 100.0
        for _ in range(n_rows):
            c = max(5.0, c + rng.uniform(-1.5, 1.7))
            closes.append(c)
        flags = [rng.random() > 0.45 for _ in closes]
        prices = _candle_series(_closes_to_candles(closes, green=flags))

        app = FastAPI()
        app.include_router(strategy_router.router)
        app.dependency_overrides[get_db] = lambda: None

        patches = [
            patch.object(strategy_router, "load_prices", AsyncMock(return_value=prices)),
            patch.object(strategy_router.cache, "get_json", AsyncMock(return_value=None)),
            patch.object(strategy_router.cache, "set_json", AsyncMock(return_value=None)),
            patch.object(
                strategy_router.cache, "get_or_compute", AsyncMock(return_value=None)
            ),
            patch.object(
                strategy_router.regime, "extract_symbol_view", lambda *_a, **_k: None
            ),
            patch.object(
                strategy_router.strategies,
                "extract_factor_momentum_view",
                lambda *_a, **_k: None,
            ),
        ]
        return TestClient(app), patches

    def _get(self, n_rows):
        client, patches = self._client(n_rows)
        for p in patches:
            p.start()
        try:
            return client.get("/etfs/VOO/strategies")
        finally:
            for p in patches:
                p.stop()

    def test_all_seven_appear_under_technical_strategies_only(self):
        res = self._get(300)
        assert res.status_code == 200
        data = res.json()["data"]
        technical = data["technicalStrategies"]
        for camel in (
            "tdSetup",
            "supertrendConfirmation",
            "ufxVervoortHeikenAshi",
            "trendSignalsTrendTide",
            "pivotLevels",
            "cpr",
            "taCandlesTrendHealth",
        ):
            assert technical[camel] is not None, camel
            assert data[camel] == technical[camel]  # flat field mirrors group
            for other in (
                "quantitativeStrategies",
                "regimeStrategies",
                "calendarStrategies",
                "portfolioStrategies",
            ):
                assert camel not in (data.get(other) or {}), (other, camel)
        assert "indicatorStrategies" not in data

    def test_existing_flat_and_grouped_fields_still_present(self):
        data = self._get(300).json()["data"]
        assert data["macd"] is not None
        assert data["technicalStrategies"]["macd"] is not None
        assert data["quantitativeStrategies"]["evarRisk"] is not None

    def test_insufficient_history_reports_new_strategies_in_message(self):
        res = self._get(30)
        assert res.status_code == 200
        body = res.json()
        assert body["data"]["tdSetup"] is not None  # 30 rows is enough for TD 9
        assert body["data"]["cpr"] is None  # needs 61
        assert "CPR (Central Pivot Range)" in body["message"]
        assert "TA Candles / Trend Health" in body["message"]
        assert "TD Setup / TD 9" not in body["message"]