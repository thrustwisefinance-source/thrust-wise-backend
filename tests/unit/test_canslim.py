"""
Unit tests for the CANSLIM screener (app.services.canslim).

Covers:
- compute_canslim_score: pure scoring logic with deterministic inputs
  (all seven pass, exactly five pass, four pass, every criterion's
  boundary, missing data, score aggregation, minimum-score filtering)
- compute_price_derived_inputs: insufficient history, 52-week high/low,
  SMA200 extraction from list[DailyPrice]
- screen_universe: ranking/ordering and insufficient-history handling

No live external API calls anywhere in this file — every input is a
deterministic keyword argument or a synthetic in-memory price series,
same convention as tests/unit/test_strategies.py.
"""

import datetime
from unittest.mock import MagicMock

import pytest

from app.services import canslim


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


# A baseline set of inputs where every one of the seven criteria passes,
# used as a starting point that individual tests tweak one field at a
# time to probe each boundary in isolation.
_ALL_PASS_KWARGS = {
    "earnings_quarterly_growth": 0.30,  # >= 25%
    "revenue_growth": 0.25,  # >= 20%
    "return_on_equity": 0.20,  # >= 17%
    "current_price": 95.0,
    "week_52_high": 100.0,  # 95/100 = 0.95 >= 0.85
    "week_52_low": 50.0,  # (95-50)/50*100 = 90% >= 30%
    "float_shares": 100_000_000,  # < 500,000,000
    "institutional_ownership": 0.50,  # between 0.30 and 0.85
    "sma200": 90.0,  # 95 > 90
}


class TestCanslimAllSevenPass:
    def test_all_seven_pass_scores_seven(self):
        result = canslim.compute_canslim_score(**_ALL_PASS_KWARGS)
        assert result["score"] == 7
        assert result["max_score"] == 7
        assert result["c_pass"] is True
        assert result["a_pass"] is True
        assert result["n_pass"] is True
        assert result["s_pass"] is True
        assert result["l_pass"] is True
        assert result["i_pass"] is True
        assert result["m_pass"] is True
        assert result["data_complete"] is True
        assert result["criteria_evaluated"] == 7
        assert result["unavailable_criteria"] == []
        assert result["meets_minimum_score"] is True


class TestCanslimPartialScores:
    def test_exactly_five_pass_scores_five(self):
        kwargs = dict(_ALL_PASS_KWARGS)
        # Fail S (float too large) and I (ownership too low) -> 5 remain.
        kwargs["float_shares"] = 600_000_000
        kwargs["institutional_ownership"] = 0.10
        result = canslim.compute_canslim_score(**kwargs)
        assert result["score"] == 5
        assert result["s_pass"] is False
        assert result["i_pass"] is False
        assert result["meets_minimum_score"] is True  # 5 >= default minimum (5)

    def test_four_pass_scores_four(self):
        kwargs = dict(_ALL_PASS_KWARGS)
        kwargs["float_shares"] = 600_000_000
        kwargs["institutional_ownership"] = 0.10
        kwargs["earnings_quarterly_growth"] = 0.10  # fails C
        result = canslim.compute_canslim_score(**kwargs)
        assert result["score"] == 4
        assert result["meets_minimum_score"] is False  # 4 < default minimum (5)


class TestCanslimCBoundary:
    def test_exactly_25_percent_passes(self):
        result = canslim.compute_canslim_score(earnings_quarterly_growth=0.25)
        assert result["c_pass"] is True

    def test_just_under_25_percent_fails(self):
        result = canslim.compute_canslim_score(earnings_quarterly_growth=0.2499)
        assert result["c_pass"] is False


class TestCanslimABoundary:
    def test_revenue_boundary_exactly_20_percent_with_sufficient_roe_passes(self):
        result = canslim.compute_canslim_score(
            revenue_growth=0.20, return_on_equity=0.17
        )
        assert result["a_pass"] is True

    def test_revenue_just_under_20_percent_fails_even_with_high_roe(self):
        result = canslim.compute_canslim_score(
            revenue_growth=0.1999, return_on_equity=0.30
        )
        assert result["a_pass"] is False

    def test_roe_boundary_exactly_17_percent_with_sufficient_revenue_passes(self):
        result = canslim.compute_canslim_score(
            revenue_growth=0.30, return_on_equity=0.17
        )
        assert result["a_pass"] is True

    def test_roe_just_under_17_percent_fails_even_with_high_revenue(self):
        result = canslim.compute_canslim_score(
            revenue_growth=0.30, return_on_equity=0.1699
        )
        assert result["a_pass"] is False


class TestCanslimNBoundary:
    def test_exactly_85_percent_of_52w_high_passes(self):
        result = canslim.compute_canslim_score(current_price=85.0, week_52_high=100.0)
        assert result["n_pass"] is True

    def test_just_under_85_percent_fails(self):
        result = canslim.compute_canslim_score(current_price=84.9, week_52_high=100.0)
        assert result["n_pass"] is False


class TestCanslimSBoundary:
    def test_just_under_500_million_float_passes(self):
        result = canslim.compute_canslim_score(float_shares=499_999_999)
        assert result["s_pass"] is True

    def test_exactly_500_million_float_fails(self):
        result = canslim.compute_canslim_score(float_shares=500_000_000)
        assert result["s_pass"] is False


class TestCanslimLBoundary:
    def test_exactly_30_percent_above_52w_low_passes(self):
        result = canslim.compute_canslim_score(current_price=65.0, week_52_low=50.0)
        assert result["l_pass"] is True

    def test_just_under_30_percent_above_52w_low_fails(self):
        result = canslim.compute_canslim_score(current_price=64.9, week_52_low=50.0)
        assert result["l_pass"] is False


class TestCanslimIBoundary:
    def test_lower_boundary_exactly_30_percent_passes(self):
        result = canslim.compute_canslim_score(institutional_ownership=0.30)
        assert result["i_pass"] is True

    def test_just_under_lower_boundary_fails(self):
        result = canslim.compute_canslim_score(institutional_ownership=0.2999)
        assert result["i_pass"] is False

    def test_upper_boundary_exactly_85_percent_passes(self):
        result = canslim.compute_canslim_score(institutional_ownership=0.85)
        assert result["i_pass"] is True

    def test_just_over_upper_boundary_fails(self):
        result = canslim.compute_canslim_score(institutional_ownership=0.8501)
        assert result["i_pass"] is False


class TestCanslimMBoundary:
    def test_price_above_sma200_passes(self):
        result = canslim.compute_canslim_score(current_price=100.01, sma200=100.0)
        assert result["m_pass"] is True

    def test_price_at_or_below_sma200_fails(self):
        result = canslim.compute_canslim_score(current_price=100.0, sma200=100.0)
        assert result["m_pass"] is False


class TestCanslimMissingData:
    def test_no_kwargs_all_unavailable(self):
        result = canslim.compute_canslim_score()
        assert result["score"] == 0
        assert result["criteria_evaluated"] == 0
        assert result["data_complete"] is False
        for key in ("c_pass", "a_pass", "n_pass", "s_pass", "l_pass", "i_pass", "m_pass"):
            assert result[key] is None
        assert len(result["unavailable_criteria"]) == 7

    def test_partial_data_only_counts_evaluated_criteria(self):
        result = canslim.compute_canslim_score(
            earnings_quarterly_growth=0.30,  # C passes
            current_price=95.0,
            sma200=90.0,  # M passes
        )
        assert result["score"] == 2
        assert result["criteria_evaluated"] == 2
        assert result["data_complete"] is False
        assert result["a_pass"] is None
        assert result["n_pass"] is None  # week_52_high missing
        assert result["s_pass"] is None
        assert result["l_pass"] is None
        assert result["i_pass"] is None
        assert len(result["unavailable_criteria"]) == 5

    def test_a_requires_both_revenue_and_roe_not_just_one(self):
        assert canslim.compute_canslim_score(revenue_growth=0.30)["a_pass"] is None
        assert (
            canslim.compute_canslim_score(return_on_equity=0.30)["a_pass"] is None
        )

    def test_zero_week_52_high_does_not_raise_and_is_unavailable(self):
        result = canslim.compute_canslim_score(current_price=10.0, week_52_high=0.0)
        assert result["n_pass"] is None

    def test_zero_week_52_low_does_not_raise_and_is_unavailable(self):
        result = canslim.compute_canslim_score(current_price=10.0, week_52_low=0.0)
        assert result["l_pass"] is None


class TestCanslimScoreAggregation:
    def test_score_never_counts_none_as_a_pass(self):
        kwargs = dict(_ALL_PASS_KWARGS)
        del kwargs["float_shares"]  # S becomes unavailable, not a fail
        result = canslim.compute_canslim_score(**kwargs)
        assert result["s_pass"] is None
        assert result["score"] == 6  # the other six still pass
        assert result["criteria_evaluated"] == 6


class TestCanslimMinimumScoreFiltering:
    def test_default_minimum_score_is_five(self):
        assert canslim.CANSLIM_DEFAULT_MINIMUM_SCORE == 5

    def test_meets_minimum_score_flag_matches_default_threshold(self):
        kwargs = dict(_ALL_PASS_KWARGS)
        kwargs["float_shares"] = 600_000_000  # fails S
        kwargs["institutional_ownership"] = 0.10  # fails I
        # Dropping current_price below week_52_high/low/sma200 also fails
        # N, L, and M (all three depend on current_price) -> only C and A
        # (which don't depend on current_price) remain passing: score 2.
        kwargs["current_price"] = 40.0
        result = canslim.compute_canslim_score(**kwargs)
        assert result["score"] == 2
        assert result["meets_minimum_score"] is False


class TestCanslimPriceDerivedInputs:
    def test_insufficient_history_returns_none(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 50)
        assert canslim.compute_price_derived_inputs(prices) is None

    def test_empty_prices_returns_none(self):
        assert canslim.compute_price_derived_inputs([]) is None

    def test_52_week_high_low_and_sma200_are_derived_correctly(self):
        n = 260
        start = datetime.date(2023, 1, 2)
        # Rising sawtooth so max/min over the trailing 252 bars are
        # unambiguous and distinct from the plain 200-bar SMA.
        closes = [100.0 + (i % 20) + i * 0.05 for i in range(n)]
        prices = _make_series("VOO", start, closes)

        result = canslim.compute_price_derived_inputs(prices)
        assert result is not None
        assert result["current_price"] == pytest.approx(closes[-1])

        tail_252 = closes[-252:]
        assert result["week_52_high"] == pytest.approx(max(tail_252))
        assert result["week_52_low"] == pytest.approx(min(tail_252))

        tail_200 = closes[-200:]
        assert result["sma200"] == pytest.approx(sum(tail_200) / len(tail_200))


class TestCanslimScoreSymbolFromPrices:
    def test_returns_none_with_insufficient_history(self):
        prices = _make_series("VOO", datetime.date(2024, 1, 2), [100.0] * 10)
        assert canslim.score_symbol_from_prices("VOO", prices) is None

    def test_scores_only_price_derived_criteria(self):
        n = 260
        start = datetime.date(2023, 1, 2)
        closes = [100.0 + i * 0.05 for i in range(n)]  # steady uptrend
        prices = _make_series("VOO", start, closes)

        result = canslim.score_symbol_from_prices("VOO", prices)
        assert result is not None
        assert result["symbol"] == "VOO"
        # Only N, L, M are ever price-derivable; C, A, S, I stay
        # unavailable since ThrustWise has no fundamentals data source.
        assert result["c_pass"] is None
        assert result["a_pass"] is None
        assert result["s_pass"] is None
        assert result["i_pass"] is None
        assert result["criteria_evaluated"] == 3
        assert result["data_complete"] is False
        # A steady uptrend ending at the series high, well above SMA200.
        assert result["n_pass"] is True
        assert result["m_pass"] is True


class TestCanslimScreenUniverse:
    def _steady_uptrend(self, symbol: str, n: int = 260) -> list[MagicMock]:
        start = datetime.date(2023, 1, 2)
        closes = [100.0 + i * 0.05 for i in range(n)]
        return _make_series(symbol, start, closes)

    def _flat_then_crash(self, symbol: str, n: int = 260) -> list[MagicMock]:
        start = datetime.date(2023, 1, 2)
        closes = [100.0] * (n - 5) + [70.0] * 5  # well below its own SMA200
        return _make_series(symbol, start, closes)

    def test_results_ranked_by_score_descending(self):
        prices_by_symbol = {
            "VOO": self._steady_uptrend("VOO"),
            "SPY": self._flat_then_crash("SPY"),
        }
        screen = canslim.screen_universe(prices_by_symbol)
        scores = [r["score"] for r in screen["results"]]
        assert scores == sorted(scores, reverse=True)
        # The steady uptrend (N+M pass) should outrank the post-crash
        # symbol (below its own SMA200 -> M fails, and not near its own
        # high -> N fails).
        symbols_in_order = [r["symbol"] for r in screen["results"]]
        assert symbols_in_order.index("VOO") < symbols_in_order.index("SPY")

    def test_insufficient_history_symbols_are_excluded_and_listed(self):
        prices_by_symbol = {
            "VOO": self._steady_uptrend("VOO"),
            "GLD": _make_series("GLD", datetime.date(2024, 1, 2), [100.0] * 10),
        }
        screen = canslim.screen_universe(prices_by_symbol)
        result_symbols = [r["symbol"] for r in screen["results"]]
        assert "GLD" not in result_symbols
        assert "GLD" in screen["unavailable_symbols"]
        assert "VOO" in result_symbols

    def test_empty_universe_returns_empty_results(self):
        screen = canslim.screen_universe({})
        assert screen["results"] == []
        assert screen["unavailable_symbols"] == []
        assert screen["max_score"] == canslim.CANSLIM_MAX_SCORE
        assert screen["minimum_score"] == canslim.CANSLIM_DEFAULT_MINIMUM_SCORE

    def test_tie_broken_by_symbol_ascending(self):
        prices_by_symbol = {
            "SPY": self._steady_uptrend("SPY"),
            "QQQ": self._steady_uptrend("QQQ"),
        }
        screen = canslim.screen_universe(prices_by_symbol)
        # Both should score identically (same synthetic price shape) ->
        # alphabetical tiebreak.
        assert [r["symbol"] for r in screen["results"]] == ["QQQ", "SPY"]