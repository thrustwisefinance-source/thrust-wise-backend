"""
Unit tests for allocation-mix blending and index snapshots.
"""

import datetime
from unittest.mock import MagicMock

from app.constants import MarketIndexMeta
from app.services import analytics


def _make_price(symbol: str, date: datetime.date, close: float) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.adjusted_close = close
    p.open = close
    p.high = close
    p.low = close
    p.volume = 1000000
    return p


def _make_series(
    symbol: str, start: datetime.date, prices: list[float]
) -> list[MagicMock]:
    result = []
    current = start
    for price in prices:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price(symbol, current, price))
        current += datetime.timedelta(days=1)
    return result


START = datetime.date(2024, 1, 1)


def _growing_series(
    symbol: str, n: int = 60, daily_growth: float = 1.001
) -> list[MagicMock]:
    prices = [100.0 * (daily_growth**i) for i in range(n)]
    return _make_series(symbol, START, prices)


class TestComputeAllocation:
    def test_weights_normalize_to_100(self):
        a = _growing_series("AAA")
        b = _growing_series("BBB")
        result = analytics.compute_allocation(["AAA", "BBB"], [2.0, 2.0], [a, b])
        assert [item["weight"] for item in result["allocations"]] == [50.0, 50.0]

    def test_identical_series_blend_equals_constituent(self):
        a = _growing_series("AAA")
        b = _growing_series("BBB")
        blend = analytics.compute_allocation(["AAA", "BBB"], [50, 50], [a, b])
        # Blending two identical assets must not change risk/return
        single = analytics.compute_allocation(["AAA", "BBB"], [100, 0.0001], [a, b])
        assert blend["metrics"]["volatility"] == single["metrics"]["volatility"]
        assert (
            blend["metrics"]["annualized_return"]
            == single["metrics"]["annualized_return"]
        )

    def test_flat_series_has_zero_risk(self):
        a = _make_series("AAA", START, [100.0] * 60)
        b = _make_series("BBB", START, [50.0] * 60)
        result = analytics.compute_allocation(["AAA", "BBB"], [60, 40], [a, b])
        assert result["metrics"]["volatility"] == 0.0
        assert result["metrics"]["max_drawdown"] == 0.0
        assert result["metrics"]["risk_score"] == 1.0

    def test_monotonic_growth_has_no_drawdown(self):
        a = _growing_series("AAA")
        b = _growing_series("BBB", daily_growth=1.002)
        result = analytics.compute_allocation(["AAA", "BBB"], [50, 50], [a, b])
        assert result["metrics"]["max_drawdown"] == 0.0
        assert result["metrics"]["annualized_return"] > 0

    def test_short_history_returns_zeroed_metrics(self):
        a = _make_series("AAA", START, [100.0, 101.0])
        b = _make_series("BBB", START, [50.0, 51.0])
        result = analytics.compute_allocation(["AAA", "BBB"], [50, 50], [a, b])
        assert result["metrics"]["annualized_return"] == 0.0
        assert result["metrics"]["risk_score"] == 1.0

    def test_correlation_matrix_shape_matches_inputs(self):
        a = _growing_series("AAA")
        b = _growing_series("BBB", daily_growth=1.0005)
        result = analytics.compute_allocation(["AAA", "BBB"], [70, 30], [a, b])
        matrix = result["correlation_matrix"]
        assert len(matrix) == 2
        assert all(len(row) == 2 for row in matrix)
        assert matrix[0][0] == 1.0


class TestComputeIndexSnapshot:
    META = MarketIndexMeta(symbol="GSPC", name="S&P 500")

    def test_basic_snapshot(self):
        prices = _make_series("GSPC", START, [7400.0, 7450.0, 7500.0])
        snap = analytics.compute_index_snapshot(self.META, prices)
        assert snap["symbol"] == "GSPC"
        assert snap["name"] == "S&P 500"
        assert snap["level"] == 7500.0
        assert snap["change_amount"] == 50.0
        assert round(snap["change_percent"], 2) == 0.67
        assert len(snap["sparkline"]) == 3

    def test_empty_prices(self):
        snap = analytics.compute_index_snapshot(self.META, [])
        assert snap["level"] == 0.0
        assert snap["change_percent"] == 0.0
