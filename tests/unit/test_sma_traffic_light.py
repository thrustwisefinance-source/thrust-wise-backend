"""
Tests for the Moving Average Traffic Light enhancement of SMA Trend
(app.services.sma_traffic_light). Deterministic synthetic prices, no network,
no database.
"""

import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.schemas import SmaTrendStrategy, SmaTrafficLight
from app.services import strategies as stl
from app.services import strategies

D0 = datetime.date(2024, 1, 1)


def bars(closes, start=D0):
    out, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        out.append(
            SimpleNamespace(symbol="T", date=d, open=c, high=c, low=c, close=c, adjusted_close=c, volume=1000)
        )
        d += datetime.timedelta(days=1)
    return out


class TestClassification:
    def test_bullish_ordering(self):
        tl = stl.compute_sma_traffic_light(bars(np.linspace(100, 200, 80)))
        assert tl["state"] == "Bullish" and tl["colour"] == "Green"
        assert tl["bullish_condition_met"] and not tl["bearish_condition_met"]
        assert tl["price"] > tl["sma10"] > tl["sma25"] > tl["sma50"]
        assert tl["ordering_descending"] == ["Close", "SMA10", "SMA25", "SMA50"]
        assert tl["ordering_text"] == "Close > SMA10 > SMA25 > SMA50"

    def test_bearish_ordering(self):
        tl = stl.compute_sma_traffic_light(bars(np.linspace(200, 100, 80)))
        assert tl["state"] == "Bearish" and tl["colour"] == "Red"
        assert tl["bearish_condition_met"] and not tl["bullish_condition_met"]
        assert tl["price"] < tl["sma10"] < tl["sma25"] < tl["sma50"]
        assert tl["ordering_descending"] == ["SMA50", "SMA25", "SMA10", "Close"]

    def test_mixed_when_price_dips_below_short_average_in_uptrend(self):
        closes = list(np.linspace(100, 200, 79)) + [150.0]
        tl = stl.compute_sma_traffic_light(bars(closes))
        assert tl["state"] == "Mixed" and tl["colour"] == "Amber"
        assert not tl["bullish_condition_met"] and not tl["bearish_condition_met"]
        assert tl["price"] < tl["sma10"]  # not automatically bullish or bearish

    def test_flat_series_ties_are_mixed_not_bullish_or_bearish(self):
        tl = stl.compute_sma_traffic_light(bars([100.0] * 70))
        assert tl["state"] == "Mixed" and tl["has_ties"] is True
        assert "=" in tl["ordering_text"]

    def test_sma_values_match_manual_means(self):
        closes = list(np.random.default_rng(1).uniform(90, 110, 90))
        tl = stl.compute_sma_traffic_light(bars(closes))
        assert tl["sma10"] == pytest.approx(np.mean(closes[-10:]), abs=0.01)
        assert tl["sma25"] == pytest.approx(np.mean(closes[-25:]), abs=0.01)
        assert tl["sma50"] == pytest.approx(np.mean(closes[-50:]), abs=0.01)
        assert tl["price"] == pytest.approx(closes[-1], abs=0.01)


class TestHistoricalComparisonAndSlope:
    def test_lookback_indexing_is_close_n_bars_earlier_not_the_sma(self):
        closes = list(range(1, 61))  # last close 60
        tl = stl.compute_sma_traffic_light(bars(closes))
        comp = {c["period"]: c for c in tl["historical_price_comparison"]["comparisons"]}
        assert comp[10]["reference_close"] == 50.0  # closes[-1-10]
        assert comp[25]["reference_close"] == 35.0
        assert comp[50]["reference_close"] == 10.0
        assert comp[10]["position"] == "Above"
        assert comp[10]["change_percent"] == pytest.approx((60 / 50 - 1) * 100, abs=0.01)
        # differs from the SMA10 comparison (SMA10 = 55.5)
        pos = {p["period"]: p for p in tl["price_positioning"]}
        assert pos[10]["sma"] == 55.5 and pos[10]["sma"] != comp[10]["reference_close"]

    def test_exactly_50_closes_marks_50_bar_comparison_unavailable_and_no_slope(self):
        tl = stl.compute_sma_traffic_light(bars(list(range(1, 51))))
        comp = {c["period"]: c for c in tl["historical_price_comparison"]["comparisons"]}
        assert comp[25]["available"] is True and comp[50]["available"] is False
        assert comp[50]["reference_close"] is None
        assert tl["slopes"]["sma10"]["slope_percent"] is None

    def test_slope_directions(self):
        up = stl.compute_sma_traffic_light(bars(np.linspace(100, 200, 80)))
        down = stl.compute_sma_traffic_light(bars(np.linspace(200, 100, 80)))
        flat = stl.compute_sma_traffic_light(bars([100.0] * 80))
        assert up["slopes"]["sma10"]["direction"] == "Rising"
        assert down["slopes"]["sma10"]["direction"] == "Falling"
        assert flat["slopes"]["sma10"]["direction"] == "Flat"
        assert flat["slopes"]["sma10"]["slope_percent"] == 0.0


class TestCrossovers:
    def test_bullish_crossovers_detected_with_bars_ago(self):
        closes = [100.0] * 52 + [102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0]
        tl = stl.compute_sma_traffic_light(bars(closes))
        events = tl["crossovers"]["events"]
        assert {e["pair"] for e in events} == {"SMA10/SMA25", "SMA25/SMA50", "SMA10/SMA50"}
        assert all(e["type"] == "Bullish" and e["bars_ago"] == 7 for e in events)

    def test_bearish_crossover(self):
        closes = [100.0] * 52 + [98.0, 96.0, 94.0, 92.0, 90.0, 88.0, 86.0, 84.0]
        events = stl.compute_sma_traffic_light(bars(closes))["crossovers"]["events"]
        assert events and all(e["type"] == "Bearish" for e in events)

    def test_old_crossover_outside_window_is_not_reported(self):
        closes = [100.0] * 52 + list(np.linspace(102, 140, 8)) + [140.0] * 12
        tl = stl.compute_sma_traffic_light(bars(closes))
        assert tl["crossovers"]["events"] == [] and tl["crossovers"]["any_recent_crossover"] is False


class TestEdgeCasesAndCompatibility:
    def test_insufficient_history_returns_none(self):
        assert stl.compute_sma_traffic_light(bars(list(range(1, 50)))) is None
        assert stl.compute_sma_traffic_light([]) is None

    def test_invalid_closes_are_ignored(self):
        closes = [float("nan")] * 5 + list(np.linspace(100, 150, 60))
        assert stl.compute_sma_traffic_light(bars(closes)) is not None

    def test_attach_preserves_every_existing_sma_trend_value(self):
        prices = bars(np.linspace(100, 300, 260))
        original = strategies.compute_sma_trend(prices)
        enhanced = stl.attach_traffic_light(original, prices)
        assert {k: v for k, v in enhanced.items() if k != "traffic_light"} == original
        assert enhanced is not original and "traffic_light" not in original  # original not mutated
        assert enhanced["traffic_light"]["state"] == "Bullish"

    def test_attach_none_stays_none(self):
        assert stl.attach_traffic_light(None, bars([1.0] * 5)) is None

    def test_failure_never_breaks_sma_trend(self, monkeypatch):
        prices = bars(np.linspace(100, 300, 260))
        monkeypatch.setattr(stl, "compute_sma_traffic_light", lambda _p: (_ for _ in ()).throw(RuntimeError("x")))
        out = stl.attach_traffic_light(strategies.compute_sma_trend(prices), prices)
        assert out["traffic_light"] is None and out["sma50"] is not None

    def test_chart_rows_and_schema_serialisation(self):
        prices = bars(np.linspace(100, 300, 260))
        out = stl.attach_traffic_light(strategies.compute_sma_trend(prices), prices)
        model = SmaTrendStrategy.model_validate(out)
        dumped = model.model_dump(by_alias=True)
        assert dumped["trafficLight"]["state"] == "Bullish"
        assert len(dumped["trafficLight"]["chartData"]) == strategies.CHART_LOOKBACK_DAYS
        assert set(dumped["trafficLight"]["chartData"][0]) == {"date", "close", "sma10", "sma25", "sma50", "state"}
        assert isinstance(model.traffic_light, SmaTrafficLight)

    def test_old_cached_payload_without_traffic_light_still_validates(self):
        prices = bars(np.linspace(100, 300, 260))
        legacy = strategies.compute_sma_trend(prices)
        assert SmaTrendStrategy.model_validate(legacy).traffic_light is None