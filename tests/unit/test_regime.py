"""
Unit tests for the Risk-On/Risk-Off Market Regime engine
(app.services.regime).

Covers:
- Multiple ETFs processed correctly end-to-end.
- Weekly technical features are generated (non-crashing, sane ranges).
- Standardization produces roughly zero-mean columns.
- DTW distance is symmetric, zero for an identical series, and finite.
- Clustering splits N>2 symbols into exactly two groups.
- Risk-On/Risk-Off classification is one of the two valid labels.
- Insufficient / missing ETF history is handled safely (no crash, None
  or partial result rather than an exception).
- compute_market_regime returns valid chart_data.
"""

import datetime
import random
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services import regime


def _make_price(symbol: str, date: datetime.date, close: float, volume: int = 1_000_000):
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.high = close * 1.01
    p.low = close * 0.99
    p.adjusted_close = close
    p.volume = volume
    return p


def _make_series(symbol: str, start: datetime.date, closes: list[float]) -> list:
    result = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price(symbol, current, close))
        current += datetime.timedelta(days=1)
    return result


def _random_walk(n: int, seed: int, drift: float = 0.0003, vol: float = 0.01) -> list[float]:
    rng = random.Random(seed)
    price = 100.0
    closes = []
    for _ in range(n):
        price *= 1 + rng.gauss(drift, vol)
        closes.append(max(price, 1.0))
    return closes


def _full_universe(n_days: int = 700) -> dict[str, list]:
    """Enough daily history (~700 trading days ≈ 2.7yrs ≈ 140 weeks) for
    every symbol in the test universe to clear MIN_WEEKS_REQUIRED."""
    start = datetime.date(2020, 1, 2)
    return {
        "SPY": _make_series("SPY", start, _random_walk(n_days, seed=1, vol=0.01)),
        "VOO": _make_series("VOO", start, _random_walk(n_days, seed=2, vol=0.01)),
        "VTI": _make_series("VTI", start, _random_walk(n_days, seed=3, vol=0.011)),
        "QQQ": _make_series("QQQ", start, _random_walk(n_days, seed=4, vol=0.015)),
        "GLD": _make_series("GLD", start, _random_walk(n_days, seed=5, drift=0.0001, vol=0.008)),
        "SHY": _make_series("SHY", start, _random_walk(n_days, seed=6, drift=0.00005, vol=0.001)),
    }


class TestWeeklyFeatures:
    def test_weekly_resample_shape(self):
        prices = _make_series("SPY", datetime.date(2021, 1, 4), _random_walk(400, seed=1))
        weekly = regime._to_weekly_frame(prices)
        assert not weekly.empty
        assert set(["high", "low", "close", "volume"]).issubset(weekly.columns)
        # Weekly bars should be far fewer than daily rows.
        assert len(weekly) < 400

    def test_features_generated_without_crashing(self):
        prices = _make_series("QQQ", datetime.date(2021, 1, 4), _random_walk(700, seed=4))
        weekly = regime._to_weekly_frame(prices)
        features, returns, scaled = regime._build_features(weekly, spy_scaled_return=None)
        assert list(features.columns) == regime.FEATURE_COLUMNS
        assert len(features) == len(weekly)
        # Should have at least some non-NaN values once warmed up.
        assert features.dropna().shape[0] > 0

    def test_empty_prices_do_not_crash(self):
        weekly = regime._to_weekly_frame([])
        assert weekly.empty
        features, returns, scaled = regime._build_features(weekly, spy_scaled_return=None)
        assert features.empty


class TestStandardization:
    def test_standardized_columns_are_roughly_zero_mean(self):
        prices = _make_series("SPY", datetime.date(2020, 1, 2), _random_walk(700, seed=1))
        weekly = regime._to_weekly_frame(prices)
        features, _, _ = regime._build_features(weekly, spy_scaled_return=None)
        standardized = regime._standardize(features)
        means = standardized.mean(skipna=True)
        for col in regime.FEATURE_COLUMNS:
            if not np.isnan(means[col]):
                assert abs(means[col]) < 1e-6


class TestDtw:
    def test_dtw_distance_zero_for_identical_series(self):
        a = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])
        assert regime._dtw_distance(a, a) == pytest.approx(0.0, abs=1e-9)

    def test_dtw_distance_symmetric(self):
        a = np.random.default_rng(1).normal(size=(10, 3))
        b = np.random.default_rng(2).normal(size=(10, 3))
        assert regime._dtw_distance(a, b) == pytest.approx(regime._dtw_distance(b, a))

    def test_dtw_distance_finite_and_nonnegative(self):
        a = np.random.default_rng(3).normal(size=(20, 5))
        b = np.random.default_rng(4).normal(size=(15, 5))
        dist = regime._dtw_distance(a, b)
        assert dist >= 0.0
        assert np.isfinite(dist)

    def test_dtw_distance_empty_sequence_is_infinite(self):
        a = np.zeros((0, 3))
        b = np.random.default_rng(5).normal(size=(5, 3))
        assert regime._dtw_distance(a, b) == float("inf")


class TestClustering:
    def test_splits_into_exactly_two_clusters(self):
        symbols = ["A", "B", "C", "D", "E", "F"]
        distance = {}
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                # A/B/C close together; D/E/F close together; groups far apart.
                group_a = {"A", "B", "C"}
                same_group = (symbols[i] in group_a) == (symbols[j] in group_a)
                distance[(symbols[i], symbols[j])] = 1.0 if same_group else 50.0

        assignment = regime._cluster_into_two(symbols, distance)
        assert set(assignment.keys()) == set(symbols)
        assert len(set(assignment.values())) == 2
        assert assignment["A"] == assignment["B"] == assignment["C"]
        assert assignment["D"] == assignment["E"] == assignment["F"]
        assert assignment["A"] != assignment["D"]


class TestComputeMarketRegime:
    def test_multiple_etfs_processed_correctly(self):
        result = regime.compute_market_regime(_full_universe())
        assert result is not None
        assert result["market_regime"] in {"Risk-On", "Risk-Off"}
        assert 0.0 <= result["regime_score"] <= 100.0
        assert set(result["etf_universe"]).issubset(
            {"SPY", "VOO", "VTI", "QQQ", "GLD", "SHY"}
        )
        assert len(result["etf_universe"]) >= 2
        for symbol, entry in result["per_symbol"].items():
            assert entry["etf_regime"] in {"Risk-On", "Risk-Off"}
            assert entry["cluster"] in {0, 1}

    def test_valid_chart_data(self):
        result = regime.compute_market_regime(_full_universe())
        assert result is not None
        assert len(result["chart_data"]) > 0
        row = result["chart_data"][-1]
        assert set(row.keys()) == {
            "date",
            "regime_score",
            "regime_state",
            "avg_rolling_volatility",
        }
        assert row["regime_state"] in {"Risk-On", "Risk-Off"}

    def test_insufficient_history_returns_none(self):
        universe = _full_universe(n_days=30)  # far below MIN_WEEKS_REQUIRED
        assert regime.compute_market_regime(universe) is None

    def test_single_etf_returns_none(self):
        universe = _full_universe()
        only_spy = {"SPY": universe["SPY"]}
        assert regime.compute_market_regime(only_spy) is None

    def test_missing_etf_data_does_not_crash_whole_strategy(self):
        universe = _full_universe()
        universe["GLD"] = []  # missing entirely
        result = regime.compute_market_regime(universe)
        assert result is not None
        assert "GLD" not in result["etf_universe"]
        assert "GLD" in result["skipped_symbols"]

    def test_empty_universe_returns_none(self):
        assert regime.compute_market_regime({}) is None


class TestExtractSymbolView:
    def test_extract_for_valid_symbol(self):
        market = regime.compute_market_regime(_full_universe())
        view = regime.extract_symbol_view(market, "SPY")
        assert view is not None
        assert view["symbol"] == "SPY"
        assert view["market_regime"] in {"Risk-On", "Risk-Off"}
        assert "chart_data" in view

    def test_extract_when_market_regime_is_none(self):
        assert regime.extract_symbol_view(None, "SPY") is None

    def test_extract_for_symbol_not_in_universe(self):
        market = regime.compute_market_regime(_full_universe())
        assert regime.extract_symbol_view(market, "NOT_A_REAL_SYMBOL") is None