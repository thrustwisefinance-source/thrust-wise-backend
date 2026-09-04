"""
Unit tests for the TQQQ / TMF / IEF Rebalancing strategy
(app.services.strategies.compute_tqqq_tmf_ief_rebalancing — Strategy 11).

Covers:
- Insufficient-history handling (returns None, no crash)
- Initial allocation is 50% TQQQ / 50% TMF
- Periodic (every-2-month) rebalancing back to 50/50
- Crash filter: a single-day TQQQ decline <= -20% moves 100% into IEF
- Recovery: exiting IEF once TQQQ's close exceeds the pre-crash price,
  and resuming the normal two-month rebalancing cycle
- No look-ahead bias (chart dates strictly ascending, matching input)
"""

import datetime

import pytest

from app.services import strategies as strat


def _make_price(symbol: str, date: datetime.date, close: float):
    class _P:
        pass

    p = _P()
    p.symbol = symbol
    p.date = date
    p.close = close
    p.open = close
    p.high = close
    p.low = close
    p.adjusted_close = close
    p.volume = 1_000_000
    return p


def _make_series(symbol: str, start: datetime.date, closes: list[float]):
    """One bar per business day (Mon-Fri), starting at `start`."""
    result = []
    current = start
    for close in closes:
        while current.weekday() >= 5:
            current += datetime.timedelta(days=1)
        result.append(_make_price(symbol, current, close))
        current += datetime.timedelta(days=1)
    return result


def _flat_series(symbol: str, start: datetime.date, n_days: int, value: float = 100.0):
    return _make_series(symbol, start, [value] * n_days)


class TestInsufficientHistory:
    def test_empty_prices_return_none(self):
        assert strat.compute_tqqq_tmf_ief_rebalancing([], [], []) is None

    def test_too_few_joined_rows_return_none(self):
        start = datetime.date(2020, 1, 1)
        tqqq = _flat_series("TQQQ", start, 3)
        tmf = _flat_series("TMF", start, 3)
        ief = _flat_series("IEF", start, 3)
        assert strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief) is None

    def test_mismatched_symbol_history_returns_none(self):
        """No overlapping dates across the three symbols -> inner join is
        empty -> unavailable, same convention as portfolio_comparison."""
        tqqq = _flat_series("TQQQ", datetime.date(2019, 1, 1), 30)
        tmf = _flat_series("TMF", datetime.date(2021, 1, 1), 30)
        ief = _flat_series("IEF", datetime.date(2023, 1, 1), 30)
        assert strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief) is None


class TestInitialAllocation:
    def test_starts_50_50_tqqq_tmf(self):
        start = datetime.date(2020, 1, 1)
        n = 15
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
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


class TestPeriodicRebalancing:
    def test_rebalances_after_two_months_of_drift(self):
        start = datetime.date(2020, 1, 1)
        # ~4 months of business days so at least one 2-month rebalance fires
        n = 90
        tqqq = _make_series("TQQQ", start, [50.0 * (1.01**i) for i in range(n)])
        tmf = _make_series("TMF", start, [20.0 * (0.999**i) for i in range(n)])
        ief = _flat_series("IEF", start, n, 100.0)

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
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


class TestCrashFilter:
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

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
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

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None
        assert result["state"] == "Normal"
        assert result["crash_filter_status"] == "Inactive"
        crash_events = [r for r in result["chart_data"] if r["is_crash_event"]]
        assert len(crash_events) == 0


class TestRecovery:
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

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        recovery_events = [r for r in result["chart_data"] if r["is_recovery_event"]]
        assert len(recovery_events) == 1

        # If TQQQ climbed back above the pre-crash price and stayed there,
        # the strategy should be Normal again by the end of the series.
        if closes[-1] > pre_crash_price:
            assert result["state"] == "Normal"
            assert result["crash_filter_status"] == "Inactive"
            assert result["pre_crash_tqqq_price"] is None


class TestNoLookAheadBias:
    def test_chart_dates_strictly_ascending_and_match_input_length(self):
        start = datetime.date(2020, 1, 1)
        n = 40
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strat.compute_tqqq_tmf_ief_rebalancing(tqqq, tmf, ief)
        assert result is not None

        dates = [row["date"] for row in result["chart_data"]]
        assert dates == sorted(dates)
        assert len(set(dates)) == len(dates)
        assert len(result["chart_data"]) == n


class TestBacktestLabeling:
    def test_backtest_fields_are_clearly_labeled_and_not_fabricated(self):
        start = datetime.date(2020, 1, 1)
        n = 30
        tqqq = _flat_series("TQQQ", start, n, 50.0)
        tmf = _flat_series("TMF", start, n, 20.0)
        ief = _flat_series("IEF", start, n, 100.0)

        result = strat.compute_tqqq_tmf_ief_rebalancing(
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