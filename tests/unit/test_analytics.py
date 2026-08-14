"""
Unit tests for the analytics engine using synthetic price fixtures.

These tests use hand-computed expected values to verify:
- compute_snapshot
- compute_quote
- compute_performance_stats
- compute_risk_metrics (volatility, beta, sharpe, risk_score, max_drawdown)
- compute_performance_series
- compute_max_drawdown
- compute_correlation_matrix
"""

import datetime
from unittest.mock import MagicMock

import pytest

from app.services import analytics


def _make_price(
    symbol: str,
    date: datetime.date,
    close: float,
    adjusted_close: float | None = None,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: int = 1000000,
) -> MagicMock:
    """Create a mock DailyPrice row."""
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.adjusted_close = adjusted_close if adjusted_close is not None else close
    p.open = open_ if open_ is not None else close
    p.high = high if high is not None else close
    p.low = low if low is not None else close
    p.volume = volume
    return p


def _make_series(
    symbol: str,
    start: datetime.date,
    prices: list[float],
) -> list[MagicMock]:
    """Create a series of mock DailyPrice rows with consecutive business days."""
    result = []
    current = start
    for price in prices:
        # Skip weekends
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price(symbol, current, price))
        current += datetime.timedelta(days=1)
    return result


class TestComputeSnapshot:
    def test_basic_snapshot(self):
        prices = _make_series(
            "VOO", datetime.date(2024, 1, 2), [100.0, 102.0, 101.0, 103.0]
        )
        meta = MagicMock()
        meta.symbol = "VOO"
        meta.name = "Vanguard S&P 500"
        meta.category = "US Equity"
        meta.issuer = "Vanguard"
        meta.expense_ratio = 0.03
        meta.risk_level = "Moderate"

        snap = analytics.compute_snapshot(meta, prices)

        assert snap["symbol"] == "VOO"
        assert snap["price"] == 103.0
        # change from 101.0 → 103.0
        assert snap["change_amount"] == 2.0
        assert snap["change_percent"] == pytest.approx(1.98, abs=0.01)
        assert isinstance(snap["sparkline"], list)
        assert len(snap["sparkline"]) == 4

    def test_single_price(self):
        prices = _make_series("SPY", datetime.date(2024, 1, 2), [500.0])
        meta = MagicMock()
        meta.symbol = "SPY"
        meta.name = "SPDR S&P 500"
        meta.category = "US Equity"
        meta.issuer = "State Street"
        meta.expense_ratio = 0.09
        meta.risk_level = "Moderate"

        snap = analytics.compute_snapshot(meta, prices)
        assert snap["price"] == 500.0
        assert snap["change_amount"] == 0.0
        assert snap["change_percent"] == 0.0


class TestComputeQuote:
    def test_basic_quote(self):
        p1 = _make_price(
            "VOO",
            datetime.date(2024, 7, 1),
            520.0,
            open_=518.0,
            high=522.0,
            low=517.0,
            volume=3500000,
        )
        p2 = _make_price(
            "VOO",
            datetime.date(2024, 7, 2),
            525.0,
            open_=521.0,
            high=526.0,
            low=520.0,
            volume=4000000,
        )
        meta = MagicMock()
        meta.symbol = "VOO"

        quote = analytics.compute_quote(meta, [p1, p2])

        assert quote["symbol"] == "VOO"
        assert quote["price"] == 525.0
        assert quote["change_amount"] == 5.0
        assert quote["change_percent"] == pytest.approx(0.96, abs=0.01)
        assert quote["open"] == 521.0
        assert quote["high"] == 526.0
        assert quote["low"] == 520.0
        assert quote["previous_close"] == 520.0
        assert quote["volume"] == 4000000
        assert quote["as_of"] == "2024-07-02"


class TestComputePerformanceStats:
    def test_flat_prices(self):
        """Flat prices should yield ~0% returns everywhere."""
        prices = _make_series("VOO", datetime.date(2019, 1, 2), [100.0] * 1300)
        stats = analytics.compute_performance_stats(prices)

        assert stats["ytd"] == pytest.approx(0.0, abs=0.01)
        assert stats["one_year"] == pytest.approx(0.0, abs=0.01)
        assert stats["annualized"] == pytest.approx(0.0, abs=0.01)

    def test_known_return(self):
        """100 → 110 over a full year is 10% return."""
        start = datetime.date(2023, 1, 2)
        # Create ~252 prices linearly from 100 to 110
        prices = _make_series(
            "VOO",
            start,
            [100 + 10 * i / 251 for i in range(252)],
        )
        stats = analytics.compute_performance_stats(prices)
        assert stats["one_year"] == pytest.approx(10.0, abs=1.0)


class TestComputeRiskMetrics:
    def test_zero_volatility(self):
        """Flat prices → volatility 0, risk_score 1."""
        prices = _make_series("VOO", datetime.date(2023, 1, 2), [100.0] * 260)
        bench = prices  # self-benchmark

        metrics = analytics.compute_risk_metrics(prices, bench, is_benchmark=True)
        assert metrics["volatility"] == 0.0
        assert metrics["risk_score"] == 1.0
        assert metrics["beta"] == 1.0
        assert metrics["max_drawdown"] == 0.0

    def test_known_volatility(self):
        """Alternating +1%/-1% daily returns should give a known volatility."""
        base = 100.0
        prices_list = [base]
        for i in range(259):
            if i % 2 == 0:
                prices_list.append(prices_list[-1] * 1.01)
            else:
                prices_list.append(prices_list[-1] * 0.99)

        prices = _make_series("QQQ", datetime.date(2023, 1, 2), prices_list)
        bench = prices

        metrics = analytics.compute_risk_metrics(prices, bench, is_benchmark=True)
        # ~1% daily vol → ~15.9% annualized
        assert 14.0 <= metrics["volatility"] <= 17.0
        assert metrics["risk_score"] >= 4.0

    def test_max_drawdown_present(self):
        """Prices that drop should have a negative max_drawdown."""
        prices = _make_series("GLD", datetime.date(2023, 1, 2), [100, 110, 90, 95, 100])
        bench = prices

        metrics = analytics.compute_risk_metrics(prices, bench, is_benchmark=True)
        # Peak was 110, trough was 90 → drawdown ≈ -18.18%
        assert metrics["max_drawdown"] < -15.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing → 0% drawdown."""
        prices = _make_series(
            "VOO", datetime.date(2024, 1, 2), [100, 101, 102, 103, 104]
        )
        dd = analytics.compute_max_drawdown(prices)
        assert dd == 0.0

    def test_known_drawdown(self):
        """100 → 120 → 90 → 100: peak=120, trough=90, drawdown = -25%."""
        prices = _make_series("SPY", datetime.date(2024, 1, 2), [100, 120, 90, 100])
        dd = analytics.compute_max_drawdown(prices)
        assert dd == pytest.approx(-25.0, abs=0.1)

    def test_full_recovery(self):
        """Drawdown measures peak-to-trough, not final value."""
        prices = _make_series("QQQ", datetime.date(2024, 1, 2), [100, 200, 150, 250])
        dd = analytics.compute_max_drawdown(prices)
        # Peak 200, trough 150 → -25%
        assert dd == pytest.approx(-25.0, abs=0.1)


class TestCorrelationMatrix:
    def test_identical_series(self):
        """Two identical series should have correlation 1.0."""
        prices = _make_series(
            "VOO",
            datetime.date(2024, 1, 2),
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        )
        corr = analytics.compute_correlation_matrix([prices, prices])
        assert corr[0][0] == 1.0
        assert corr[0][1] == 1.0
        assert corr[1][0] == 1.0

    def test_single_series(self):
        """Single series → 1x1 matrix."""
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100, 101, 102])
        corr = analytics.compute_correlation_matrix([prices])
        assert corr == [[1.0]]


class TestPerformanceSeries:
    def test_produces_all_ranges(self):
        """Should produce entries for all 5 range keys."""
        prices = _make_series(
            "VOO", datetime.date(2019, 1, 2), [100 + i * 0.1 for i in range(1300)]
        )
        series = analytics.compute_performance_series(prices)
        assert set(series.keys()) == {"1M", "3M", "6M", "1Y", "5Y"}
        for key in series:
            assert len(series[key]) > 0
            # Each point should have label and value
            assert "label" in series[key][0]
            assert "value" in series[key][0]

    def test_empty_prices(self):
        series = analytics.compute_performance_series([])
        assert set(series.keys()) == {"1M", "3M", "6M", "1Y", "5Y"}
        for key in series:
            assert series[key] == []

    def test_downsamples_to_max_points(self):
        """Long series should be downsampled to ≤ MAX_CHART_POINTS."""
        prices = _make_series(
            "VOO", datetime.date(2019, 1, 2), [100 + i * 0.01 for i in range(1300)]
        )
        series = analytics.compute_performance_series(prices)
        for key in series:
            assert (
                len(series[key]) <= analytics.MAX_CHART_POINTS + 1
            )  # +1 for final point


class TestRiskScore:
    def test_low_volatility(self):
        assert analytics._risk_score(2.0) == 1.0

    def test_moderate_volatility(self):
        assert analytics._risk_score(14.0) == 5.0

    def test_extreme_volatility(self):
        assert analytics._risk_score(50.0) == 10.0
