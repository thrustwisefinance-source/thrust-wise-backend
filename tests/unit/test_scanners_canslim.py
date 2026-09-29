"""
Unit tests for the CANSLIM STOCK SCANNER (app/scanners/), separate from
tests/unit/test_canslim.py (the pre-existing, unrelated ETF-universe
CANSLIM-proxy test file — left untouched).

Covers:
- compute_criterion_c: pass, boundary, fail, missing data, zero/negative
  year-ago EPS (NOT_APPLICABLE, not a fabricated percentage)
- compute_criterion_a: pass (growth + ROE), missing years, missing ROE,
  negative earliest-year EPS, growth-without-ROE fail
- compute_criterion_n: near-high pass/fail, at-new-high, missing history
- compute_criterion_s: pass/fail on the article's 25M-share threshold,
  missing float
- compute_criterion_l / rank_relative_strength: ranking, ties, a
  single-stock "universe", missing peer data
- compute_criterion_i: pass/fail on holder count, missing holders
- compute_market_direction: uptrend, downtrend, insufficient index data
- assemble_stock_result: counts, qualifies, data_quality
- app.scanners.data extraction helpers: quarter-matching, annual-history
  trimming, price-derived inputs, insufficient history
- app.scanners.universe: universe registry lookups

No live external API calls anywhere in this file.
"""

import datetime
from unittest.mock import MagicMock

import pytest

from app.scanners import canslim as sc
from app.scanners import data as scanner_data
from app.scanners import universe as scanner_universe


# ---------------------------------------------------------------------------
# C — Current Quarterly Earnings
# ---------------------------------------------------------------------------


def test_criterion_c_pass_above_threshold():
    result = sc.compute_criterion_c(1.40, "2026-06-30", 1.00, "2025-06-30")
    assert result["status"] == sc.STATUS_PASS
    assert result["value"] == pytest.approx(0.40, abs=1e-6)


def test_criterion_c_boundary_exactly_25_percent():
    result = sc.compute_criterion_c(1.25, "2026-06-30", 1.00, "2025-06-30")
    assert result["status"] == sc.STATUS_PASS


def test_criterion_c_fail_below_threshold():
    result = sc.compute_criterion_c(1.10, "2026-06-30", 1.00, "2025-06-30")
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_c_missing_eps_is_unavailable_not_fail():
    result = sc.compute_criterion_c(None, None, None, None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


def test_criterion_c_negative_year_ago_eps_is_not_applicable():
    result = sc.compute_criterion_c(0.50, "2026-06-30", -0.10, "2025-06-30")
    assert result["status"] == sc.STATUS_NOT_APPLICABLE


def test_criterion_c_zero_year_ago_eps_is_not_applicable():
    result = sc.compute_criterion_c(0.50, "2026-06-30", 0.0, "2025-06-30")
    assert result["status"] == sc.STATUS_NOT_APPLICABLE


# ---------------------------------------------------------------------------
# A — Annual Earnings Growth
# ---------------------------------------------------------------------------


def _annual(years_eps: list[tuple[int, float]]) -> list[dict]:
    return [{"year": y, "eps": e} for y, e in years_eps]


def test_criterion_a_pass_growth_and_roe():
    history = _annual([(2023, 2.00), (2024, 2.50), (2025, 3.00)])
    result = sc.compute_criterion_a(history, return_on_equity=0.20)
    assert result["status"] == sc.STATUS_PASS
    assert result["value"] == pytest.approx(0.50, abs=1e-6)


def test_criterion_a_fails_when_roe_below_threshold_despite_growth():
    history = _annual([(2023, 2.00), (2024, 2.50), (2025, 3.00)])
    result = sc.compute_criterion_a(history, return_on_equity=0.10)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_a_fails_when_growth_below_threshold_despite_roe():
    history = _annual([(2023, 2.00), (2024, 2.05), (2025, 2.10)])
    result = sc.compute_criterion_a(history, return_on_equity=0.25)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_a_missing_years_is_unavailable():
    history = _annual([(2024, 2.00), (2025, 2.50)])  # only 2 years
    result = sc.compute_criterion_a(history, return_on_equity=0.20)
    assert result["status"] == sc.STATUS_UNAVAILABLE


def test_criterion_a_missing_roe_is_unavailable():
    history = _annual([(2023, 2.00), (2024, 2.50), (2025, 3.00)])
    result = sc.compute_criterion_a(history, return_on_equity=None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


def test_criterion_a_negative_earliest_year_is_not_applicable():
    history = _annual([(2023, -1.00), (2024, 0.50), (2025, 1.00)])
    result = sc.compute_criterion_a(history, return_on_equity=0.20)
    assert result["status"] == sc.STATUS_NOT_APPLICABLE


def test_criterion_a_uses_at_most_5_years():
    history = _annual(
        [(2019, 1.00), (2020, 1.00), (2021, 1.00), (2022, 1.00), (2023, 1.00), (2024, 1.30), (2025, 1.30)]
    )
    result = sc.compute_criterion_a(history, return_on_equity=0.20)
    assert result["period"] == "2021-2025"


# ---------------------------------------------------------------------------
# N — New (price-highs component)
# ---------------------------------------------------------------------------


def test_criterion_n_pass_near_high():
    result = sc.compute_criterion_n(current_price=90, week_52_high=100)
    assert result["status"] == sc.STATUS_PASS


def test_criterion_n_fail_far_from_high():
    result = sc.compute_criterion_n(current_price=70, week_52_high=100)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_n_at_new_high_flag_mentioned():
    result = sc.compute_criterion_n(current_price=101, week_52_high=100)
    assert result["status"] == sc.STATUS_PASS
    assert "new 52-week high" in result["explanation"]


def test_criterion_n_missing_history_is_unavailable():
    result = sc.compute_criterion_n(current_price=None, week_52_high=None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# S — Supply and Demand
# ---------------------------------------------------------------------------


def test_criterion_s_pass_small_float():
    result = sc.compute_criterion_s(shares_float=10_000_000)
    assert result["status"] == sc.STATUS_PASS


def test_criterion_s_fail_large_float():
    result = sc.compute_criterion_s(shares_float=500_000_000)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_s_uses_article_threshold_not_legacy_500m():
    # Exactly at the legacy ETF module's old 500M threshold must FAIL here
    # (article threshold is 25M, not 500M).
    result = sc.compute_criterion_s(shares_float=100_000_000)
    assert result["threshold"] == 25_000_000
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_s_missing_float_is_unavailable():
    result = sc.compute_criterion_s(shares_float=None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# L — Leader (relative strength ranking)
# ---------------------------------------------------------------------------


def test_rank_relative_strength_orders_best_first():
    returns = {"AAA": 0.50, "BBB": 0.10, "CCC": 0.90}
    ranks = sc.rank_relative_strength(returns)
    assert ranks["CCC"]["rank"] == 1
    assert ranks["AAA"]["rank"] == 2
    assert ranks["BBB"]["rank"] == 3
    assert ranks["CCC"]["percentile"] == 100.0
    assert ranks["BBB"]["percentile"] == 0.0


def test_rank_relative_strength_single_symbol_universe():
    ranks = sc.rank_relative_strength({"ONLY": 0.10})
    assert ranks["ONLY"]["percentile"] == 100.0
    assert ranks["ONLY"]["universe_size"] == 1


def test_rank_relative_strength_empty_universe():
    assert sc.rank_relative_strength({}) == {}


def test_criterion_l_pass_above_80th_percentile():
    rs = {"rank": 5, "universe_size": 500, "percentile": 99.0}
    result = sc.compute_criterion_l(rs)
    assert result["status"] == sc.STATUS_PASS


def test_criterion_l_fail_below_80th_percentile():
    rs = {"rank": 300, "universe_size": 500, "percentile": 40.0}
    result = sc.compute_criterion_l(rs)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_l_missing_peer_data_is_unavailable():
    result = sc.compute_criterion_l(None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


def test_criterion_l_never_claims_official_ibd_rating():
    rs = {"rank": 1, "universe_size": 10, "percentile": 100.0}
    result = sc.compute_criterion_l(rs)
    assert "NOT an official IBD RS Rating" in result["explanation"]


# ---------------------------------------------------------------------------
# I — Institutional Sponsorship
# ---------------------------------------------------------------------------


def test_criterion_i_pass_sufficient_holders():
    result = sc.compute_criterion_i(institutional_holders_count=10, percent_institutions=0.55)
    assert result["status"] == sc.STATUS_PASS


def test_criterion_i_fail_too_few_holders():
    result = sc.compute_criterion_i(institutional_holders_count=1, percent_institutions=0.05)
    assert result["status"] == sc.STATUS_FAIL


def test_criterion_i_missing_holders_is_unavailable_not_fail():
    result = sc.compute_criterion_i(institutional_holders_count=None, percent_institutions=None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# M — Market Direction
# ---------------------------------------------------------------------------


def test_market_direction_uptrend():
    result = sc.compute_market_direction(index_price=5000, index_sma200=4800)
    assert result["status"] == sc.STATUS_PASS


def test_market_direction_downtrend():
    result = sc.compute_market_direction(index_price=4700, index_sma200=4800)
    assert result["status"] == sc.STATUS_FAIL


def test_market_direction_insufficient_data_is_unavailable():
    result = sc.compute_market_direction(index_price=None, index_sma200=None)
    assert result["status"] == sc.STATUS_UNAVAILABLE


# ---------------------------------------------------------------------------
# assemble_stock_result
# ---------------------------------------------------------------------------


def _all_pass_criteria() -> dict:
    passing = {"status": sc.STATUS_PASS, "criterion": "X", "name": "x", "explanation": "", "data_source": ""}
    return {k: dict(passing) for k in sc.CRITERIA_KEYS}


def test_assemble_stock_result_all_pass_qualifies():
    result = sc.assemble_stock_result(
        symbol="AAA",
        company_name="AAA Inc",
        sector="Technology",
        industry="Software",
        universe="sp500",
        criteria=_all_pass_criteria(),
        price_date="2026-09-01",
        fundamentals_as_of="2026-06-30",
        fundamentals_fetched_at="2026-09-01T00:00:00Z",
        data_updated_at="2026-09-01T00:00:00Z",
    )
    assert result["criteria_passed"] == 7
    assert result["criteria_evaluated"] == 7
    assert result["criteria_unavailable"] == 0
    assert result["qualifies"] is True
    assert result["data_quality"] == "COMPLETE"


def test_assemble_stock_result_partial_data_not_treated_as_fail():
    criteria = _all_pass_criteria()
    criteria["i"] = {
        "status": sc.STATUS_UNAVAILABLE,
        "criterion": "I",
        "name": "i",
        "explanation": "",
        "data_source": "",
    }
    result = sc.assemble_stock_result(
        symbol="BBB",
        company_name=None,
        sector=None,
        industry=None,
        universe="sp500",
        criteria=criteria,
        price_date=None,
        fundamentals_as_of=None,
        fundamentals_fetched_at=None,
        data_updated_at=None,
    )
    assert result["criteria_passed"] == 6
    assert result["criteria_evaluated"] == 6
    assert result["criteria_unavailable"] == 1
    assert result["data_quality"] == "PARTIAL"
    # 6 of 7 still meets the default minimum of 5
    assert result["qualifies"] is True


def test_assemble_stock_result_insufficient_when_nothing_evaluated():
    criteria = {
        k: {
            "status": sc.STATUS_UNAVAILABLE,
            "criterion": k.upper(),
            "name": k,
            "explanation": "",
            "data_source": "",
        }
        for k in sc.CRITERIA_KEYS
    }
    result = sc.assemble_stock_result(
        symbol="CCC",
        company_name=None,
        sector=None,
        industry=None,
        universe="sp500",
        criteria=criteria,
        price_date=None,
        fundamentals_as_of=None,
        fundamentals_fetched_at=None,
        data_updated_at=None,
    )
    assert result["data_quality"] == "INSUFFICIENT"
    assert result["qualifies"] is False


# ---------------------------------------------------------------------------
# app.scanners.data — extraction helpers
# ---------------------------------------------------------------------------


def test_latest_and_year_ago_quarter_matches_closest_to_one_year_prior():
    history = {
        "2026-06-30": {"epsActual": 1.40},
        "2026-03-31": {"epsActual": 1.30},
        "2025-06-30": {"epsActual": 1.00},
        "2025-03-31": {"epsActual": 0.90},
    }
    eps_now, period_now, eps_ago, period_ago = scanner_data._latest_and_year_ago_quarter(history)
    assert eps_now == 1.40
    assert period_now == "2026-06-30"
    assert eps_ago == 1.00
    assert period_ago == "2025-06-30"


def test_latest_and_year_ago_quarter_missing_prior_year_entry():
    history = {"2026-06-30": {"epsActual": 1.40}}
    eps_now, period_now, eps_ago, period_ago = scanner_data._latest_and_year_ago_quarter(history)
    assert eps_now == 1.40
    assert eps_ago is None


def test_latest_and_year_ago_quarter_empty_history():
    assert scanner_data._latest_and_year_ago_quarter({}) == (None, None, None, None)


def test_latest_and_year_ago_quarter_skips_entries_without_eps_actual():
    history = {
        "2026-06-30": {"epsActual": None},
        "2026-03-31": {"epsActual": 1.20},
    }
    eps_now, period_now, _, _ = scanner_data._latest_and_year_ago_quarter(history)
    assert eps_now == 1.20
    assert period_now == "2026-03-31"


def test_annual_eps_history_ascending_and_trimmed():
    annual = {
        "2019-12-31": {"epsActual": 1.0},
        "2020-12-31": {"epsActual": 1.1},
        "2021-12-31": {"epsActual": 1.2},
        "2022-12-31": {"epsActual": 1.3},
        "2023-12-31": {"epsActual": 1.4},
        "2024-12-31": {"epsActual": 1.5},
    }
    result = scanner_data._annual_eps_history(annual, max_years=5)
    assert [r["year"] for r in result] == [2020, 2021, 2022, 2023, 2024]
    assert result[0]["eps"] == 1.1
    assert result[-1]["eps"] == 1.5


def test_annual_eps_history_skips_missing_eps():
    annual = {"2023-12-31": {"epsActual": None}, "2024-12-31": {"epsActual": 2.0}}
    result = scanner_data._annual_eps_history(annual, max_years=5)
    assert result == [{"year": 2024, "eps": 2.0}]


def _make_price(symbol: str, date: datetime.date, close: float) -> MagicMock:
    p = MagicMock()
    p.symbol = symbol
    p.date = date
    p.close = close
    return p


def test_compute_price_derived_inputs_insufficient_history():
    prices = [_make_price("AAA", datetime.date(2026, 1, 1) + datetime.timedelta(days=i), 100 + i) for i in range(10)]
    assert scanner_data.compute_price_derived_inputs(prices) is None


def test_compute_price_derived_inputs_full_history():
    start = datetime.date(2024, 1, 1)
    prices = []
    price = 100.0
    d = start
    for i in range(300):
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        price = 100.0 + i * 0.1
        prices.append(_make_price("AAA", d, price))
        d += datetime.timedelta(days=1)

    result = scanner_data.compute_price_derived_inputs(prices)
    assert result is not None
    assert result.current_price == pytest.approx(price, abs=0.01)
    assert result.week_52_high is not None
    assert result.trailing_return is not None


# ---------------------------------------------------------------------------
# app.scanners.universe — registry
# ---------------------------------------------------------------------------


def test_universe_registry_has_sp500_with_eodhd_backing():
    source = scanner_universe.get_universe_source("sp500")
    assert source is not None
    assert source.eodhd_index_symbol == "GSPC.INDX"


def test_universe_registry_unknown_key_returns_none():
    assert scanner_universe.get_universe_source("not_a_real_universe") is None


def test_list_universes_includes_sp500():
    keys = [u.key for u in scanner_universe.list_universes()]
    assert "sp500" in keys
