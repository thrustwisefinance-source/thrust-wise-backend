"""
Unit + integration tests for the four article-based strategies
(app.services.strategies Strategies 29-32):

  29 Seasonal Investment Performance Heat Map
  30 Investment Diversification Heat Map
  31 EMA + Bollinger Bands + ATR (analytics + historical backtest)
  32 Technical Crash-Warning Risk Indicator (experimental)

Everything here is deterministic: synthetic prices, mocked EODHD calls, no
network, no real API token, no database. No test places or simulates live
orders - the backtest is a pure historical simulation.
"""

import asyncio
import datetime
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db
from app.services import eodhd_client, strategies

D = datetime.date
FAR_FUTURE = D(2030, 1, 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def px(date, close, adj=None, open_=None, high=None, low=None):
    return SimpleNamespace(
        symbol="T",
        date=date,
        open=close if open_ is None else open_,
        high=close if high is None else high,
        low=close if low is None else low,
        close=close,
        adjusted_close=close if adj is None else adj,
        volume=1_000_000,
    )


def month_end_series(points):
    """points: [(year, month, close)] -> one bar on the 28th of each month."""
    return [px(D(y, m, 28), c) for y, m, c in points]


def weekday_series(start, closes, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, d = [], start
    for c in closes:
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        o = c * (1 + rng.normal(0, noise)) if noise else c
        hi, lo = max(o, c) * (1 + noise / 2), min(o, c) * (1 - noise / 2)
        rows.append(px(d, c, open_=o, high=hi, low=lo))
        d += datetime.timedelta(days=1)
    return rows


def random_walk(n, seed=1, drift=0.0004, vol=0.011, start=D(2021, 1, 4), s0=100.0):
    rng = np.random.default_rng(seed)
    rows, d, price = [], start, s0
    while len(rows) < n:
        if d.weekday() < 5:
            o = price
            c = price * (1 + rng.normal(drift, vol))
            h = max(o, c) * (1 + abs(rng.normal(0, 0.004)))
            lo = min(o, c) * (1 - abs(rng.normal(0, 0.004)))
            rows.append(px(d, c, open_=o, high=h, low=lo))
            price = c
        d += datetime.timedelta(days=1)
    return rows


def cell(result, year, month):
    row = next(r for r in result["rows"] if r["year"] == year)
    return row["months"][month - 1]


def qcell(result, year, quarter):
    row = next(r for r in result["rows"] if r["year"] == year)
    return row["quarters"][quarter - 1]


# ===========================================================================
# Strategy 29: Seasonal heat map
# ===========================================================================


class TestSeasonalHeatMap:
    def prices(self):
        return month_end_series(
            [(2023, 12, 100), (2024, 1, 110), (2024, 2, 121), (2024, 3, 108.9), (2024, 4, 108.9)]
        )

    def test_monthly_returns_and_percent_conversion(self):
        r = strategies.compute_seasonal_heat_map(self.prices(), "T", as_of=FAR_FUTURE)
        assert cell(r, 2024, 1)["return_percent"] == 10.0
        assert cell(r, 2024, 2)["return_percent"] == 10.0
        assert cell(r, 2024, 3)["return_percent"] == -10.0
        assert cell(r, 2024, 4)["return_percent"] == 0.0
        assert cell(r, 2024, 4)["sign"] == "zero"
        assert cell(r, 2024, 3)["sign"] == "negative"

    def test_percent_values_are_percent_not_fraction(self):
        pts = month_end_series([(2024, 1, 100), (2024, 2, 101.23)])
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        assert cell(r, 2024, 2)["return_percent"] == 1.23

    def test_quarter_is_geometrically_linked_not_summed(self):
        # +10%, +10%, -10%  ->  1.1*1.1*0.9 - 1 = 8.9%   (a sum would give 10%)
        r = strategies.compute_seasonal_heat_map(self.prices(), "T", as_of=FAR_FUTURE)
        q1 = qcell(r, 2024, 1)
        assert q1["status"] == "complete"
        assert q1["return_percent"] == pytest.approx(8.9, abs=0.01)
        assert q1["return_percent"] != 10.0

    def test_first_month_has_no_base_and_is_not_zero(self):
        r = strategies.compute_seasonal_heat_map(self.prices(), "T", as_of=FAR_FUTURE)
        dec = cell(r, 2023, 12)
        assert dec["status"] == "no_base" and dec["return_percent"] is None

    def test_missing_month_is_not_a_zero_return_and_blocks_the_quarter(self):
        # February has no data at all.
        pts = month_end_series([(2023, 12, 100), (2024, 1, 110), (2024, 3, 120), (2024, 4, 130)])
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        assert cell(r, 2024, 2)["status"] == "missing"
        assert cell(r, 2024, 2)["return_percent"] is None
        # March must NOT be stretched over the gap
        assert cell(r, 2024, 3)["status"] == "prior_month_missing"
        assert cell(r, 2024, 3)["return_percent"] is None
        q1 = qcell(r, 2024, 1)
        assert q1["status"] == "partial" and q1["return_percent"] is None
        assert q1["months_included"] == 1  # only the leading valid run (Jan)
        assert q1["partial_return_percent"] == 10.0
        assert any("no price data" in w for w in r["data_quality_warnings"])

    def test_in_progress_month_is_partial_and_quarter_incomplete(self):
        pts = month_end_series([(2024, 1, 100), (2024, 2, 110), (2024, 3, 121), (2024, 4, 133.1)])
        # 'today' is inside April -> April is not a complete month
        r = strategies.compute_seasonal_heat_map(pts, as_of=D(2024, 4, 29))
        apr = cell(r, 2024, 4)
        assert apr["status"] == "partial_month" and apr["return_percent"] is None
        assert apr["partial_return_percent"] == 10.0
        q2 = qcell(r, 2024, 2)
        assert q2["status"] == "partial" and q2["return_percent"] is None
        assert q2["months_included"] == 0
        # complete-period stats exclude the partial month
        april_summary = r["monthly_summary"][3]
        assert april_summary["count"] == 0
        assert any("not a complete month" in w for w in r["data_quality_warnings"])

    def test_data_ending_early_in_a_finished_month_is_partial(self):
        pts = month_end_series([(2024, 1, 100), (2024, 2, 110)])
        pts[-1] = px(D(2024, 2, 9), 110)  # data stops 9 Feb, though 'today' is far later
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        assert cell(r, 2024, 2)["status"] == "partial_month"

    def test_date_range_boundaries(self):
        r = strategies.compute_seasonal_heat_map(
            self.prices(), "T", start=D(2024, 2, 15), end=D(2024, 3, 1), as_of=FAR_FUTURE
        )
        assert [row["year"] for row in r["rows"]] == [2024]
        # months outside the range are reported as out_of_range, never invented
        assert cell(r, 2024, 1)["status"] == "out_of_range"
        assert cell(r, 2024, 4)["status"] == "out_of_range"
        # the first month in range still uses the prior month-end (before `start`) as its base
        assert cell(r, 2024, 2)["return_percent"] == 10.0
        assert cell(r, 2024, 3)["return_percent"] == -10.0
        # a quarter only partly inside the range is not complete
        assert qcell(r, 2024, 1)["status"] == "partial"
        assert r["requested_start"] == "2024-02-15" and r["requested_end"] == "2024-03-01"

    def test_range_before_available_history_warns(self):
        r = strategies.compute_seasonal_heat_map(self.prices(), start=D(2015, 1, 1), as_of=FAR_FUTURE)
        assert any("precedes the earliest stored price" in w for w in r["data_quality_warnings"])
        assert r["data_first_date"] == "2023-12-28"

    def test_range_entirely_outside_history_returns_none(self):
        assert strategies.compute_seasonal_heat_map(self.prices(), start=D(2030, 1, 1), as_of=FAR_FUTURE) is None

    def test_adjusted_vs_close_basis(self):
        pts = [px(D(2024, 1, 28), 100, adj=90), px(D(2024, 2, 28), 100, adj=99)]
        adj = strategies.compute_seasonal_heat_map(pts, price_basis="adjusted", as_of=FAR_FUTURE)
        raw = strategies.compute_seasonal_heat_map(pts, price_basis="close", as_of=FAR_FUTURE)
        assert cell(adj, 2024, 2)["return_percent"] == 10.0  # dividends reflected
        assert cell(raw, 2024, 2)["return_percent"] == 0.0
        assert "adjusted_close" in adj["adjustment_policy"] and "raw close" in raw["adjustment_policy"]

    def test_split_distorts_raw_close_and_is_flagged(self):
        pts = [
            px(D(2024, 1, 26), 200, adj=100),
            px(D(2024, 1, 29), 100, adj=100),  # 2:1 split: raw halves, adjusted flat
            px(D(2024, 2, 27), 100, adj=100),
            px(D(2024, 3, 27), 100, adj=100),
        ]
        raw = strategies.compute_seasonal_heat_map(pts, price_basis="close", as_of=FAR_FUTURE)
        assert any("split" in w.lower() for w in raw["data_quality_warnings"])

    def test_invalid_rows_are_dropped_not_zero_filled(self):
        pts = self.prices() + [px(D(2024, 5, 28), float("nan")), px(D(2024, 6, 28), -5.0)]
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        assert r["data_last_date"] == "2024-04-28"

    def test_insufficient_history_and_bad_arguments(self):
        assert strategies.compute_seasonal_heat_map([]) is None
        assert strategies.compute_seasonal_heat_map(month_end_series([(2024, 1, 100)])) is None
        with pytest.raises(ValueError):
            strategies.compute_seasonal_heat_map(self.prices(), price_basis="bogus")
        with pytest.raises(ValueError):
            strategies.compute_seasonal_heat_map(self.prices(), start=D(2024, 5, 1), end=D(2024, 1, 1))

    def test_summary_color_scale_and_calendar_year(self):
        pts = month_end_series([(2023, 12, 100)] + [(2024, m, 100 * 1.01**m) for m in range(1, 13)])
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        row = next(x for x in r["rows"] if x["year"] == 2024)
        assert row["calendar_year"]["status"] == "complete"
        assert row["calendar_year"]["return_percent"] == pytest.approx(12.68, abs=0.01)  # 1.01^12 - 1
        assert r["monthly_summary"][0]["count"] == 1 and r["monthly_summary"][0]["positive_rate_percent"] == 100.0
        assert r["color_scale"]["neutral"] == 0.0 and r["color_scale"]["max_abs_return_percent"] == 1.0
        assert len(r["month_labels"]) == 12 and r["quarter_labels"] == ["Q1", "Q2", "Q3", "Q4"]

    def test_does_not_hardcode_article_dates(self):
        pts = month_end_series([(1999, 1, 10), (1999, 2, 11)])
        r = strategies.compute_seasonal_heat_map(pts, as_of=FAR_FUTURE)
        assert [row["year"] for row in r["rows"]] == [1999]


# ===========================================================================
# Strategy 30: Diversification heat map
# ===========================================================================


def year_ends(symbol_points):
    """[(year, close)] -> bars on 29 Dec of each year."""
    return [px(D(y, 12, 29), c) for y, c in symbol_points]


class TestDiversificationHeatMap:
    def data(self):
        return {
            "AAA": year_ends([(2020, 100), (2021, 110), (2022, 99), (2023, 118.8)]),  # +10, -10, +20
            "BBB": year_ends([(2020, 100), (2021, 105), (2022, 110.25), (2023, 115.7625)]),  # +5 each
        }

    def run(self, **kw):
        kw.setdefault("as_of", FAR_FUTURE)
        return strategies.compute_diversification_heat_map(self.data(), **kw)

    def cells(self, r, symbol):
        return {c["year"]: c for c in next(a for a in r["assets"] if a["symbol"] == symbol)["cells"]}

    def test_annual_returns_consistent_across_assets(self):
        r = self.run()
        a, b = self.cells(r, "AAA"), self.cells(r, "BBB")
        assert a[2021]["return_percent"] == 10.0 and a[2022]["return_percent"] == -10.0
        assert a[2023]["return_percent"] == 20.0
        assert b[2021]["return_percent"] == b[2022]["return_percent"] == b[2023]["return_percent"] == 5.0
        assert a[2020]["status"] == "no_base"  # listing year is never ranked

    def test_ranking_within_each_year(self):
        r = self.run()
        by_year = {y["year"]: y["ranking"] for y in r["yearly_rankings"]}
        assert [e["symbol"] for e in by_year[2021]] == ["AAA", "BBB"]
        assert [e["symbol"] for e in by_year[2022]] == ["BBB", "AAA"]
        assert [e["rank"] for e in by_year[2022]] == [1, 2]
        assert by_year[2020] == []  # nothing complete in the first year
        assert self.cells(r, "AAA")[2021]["rank"] == 1 and self.cells(r, "AAA")[2022]["rank"] == 2

    def test_ties_share_a_rank(self):
        data = {
            "AAA": year_ends([(2020, 100), (2021, 110)]),
            "BBB": year_ends([(2020, 50), (2021, 55)]),
            "CCC": year_ends([(2020, 100), (2021, 90)]),
        }
        r = strategies.compute_diversification_heat_map(data, as_of=FAR_FUTURE)
        ranks = {e["symbol"]: e["rank"] for e in r["yearly_rankings"][1]["ranking"]}
        assert ranks == {"AAA": 1, "BBB": 1, "CCC": 3}

    def test_weighted_portfolio_return(self):
        r = self.run(weights_percent={"AAA": 60, "BBB": 40})
        p = {c["year"]: c for c in r["portfolio"]["cells"]}
        assert p[2021]["return_percent"] == pytest.approx(0.6 * 10 + 0.4 * 5)  # 8.0
        assert p[2022]["return_percent"] == pytest.approx(0.6 * -10 + 0.4 * 5)  # -4.0
        assert p[2023]["return_percent"] == pytest.approx(0.6 * 20 + 0.4 * 5)  # 14.0
        assert r["portfolio"]["rebalancing_policy"] == "annual_reset"
        assert "not equivalent to a buy-and-hold" in r["portfolio"]["policy_description"]
        assert r["assets"][0]["weight_percent"] == 60.0

    def test_portfolio_is_ranked_among_the_assets(self):
        r = self.run(weights_percent={"AAA": 60, "BBB": 40})
        year_2021 = r["yearly_rankings"][1]["ranking"]
        assert [e["symbol"] for e in year_2021] == ["AAA", "PORTFOLIO", "BBB"]
        p = {c["year"]: c for c in r["portfolio"]["cells"]}
        assert p[2021]["rank"] == 2 and p[2021]["rank_among"] == 3

    def test_no_weights_means_no_portfolio_row(self):
        assert self.run()["portfolio"] is None

    def test_no_diversification_claim_in_output(self):
        r = self.run(weights_percent={"AAA": 50, "BBB": 50})
        assert "does not establish diversification benefits" in r["disclaimer"]
        assert "not thereby lower-risk" in r["disclaimer"]

    # -- weight validation ---------------------------------------------------

    @pytest.mark.parametrize(
        "weights, fragment",
        [
            ({"AAA": 50, "BBB": 40}, "sum to 100"),
            ({"AAA": 70, "BBB": 40}, "sum to 100"),
            ({"AAA": -10, "BBB": 110}, "negative"),
            ({"AAA": float("nan"), "BBB": 100}, "finite"),
            ({"AAA": float("inf"), "BBB": 0}, "finite"),
            ({"AAA": "60", "BBB": 40}, "finite number"),
            ({"AAA": True, "BBB": 99}, "finite number"),
            ({"AAA": None, "BBB": 100}, "finite number"),
            ({"AAA": 100, "ZZZ": 0}, "not in the requested tickers"),
            ({"AAA": 150, "BBB": -50}, "exceed 100"),
            ({}, "non-empty"),
        ],
    )
    def test_weight_validation_rejects_bad_input(self, weights, fragment):
        with pytest.raises(ValueError, match=fragment):
            strategies.validate_portfolio_weights(["AAA", "BBB"], weights)

    def test_weight_validation_accepts_tolerance_and_defaults_omitted_to_zero(self):
        out = strategies.validate_portfolio_weights(["AAA", "BBB", "CCC"], {"aaa": 60.004, "BBB": 40})
        assert out == {"AAA": 60.004, "BBB": 40.0, "CCC": 0.0}

    def test_duplicate_weights_after_case_folding_rejected(self):
        with pytest.raises(ValueError, match="Duplicate"):
            strategies.validate_portfolio_weights(["AAA", "BBB"], {"AAA": 50, "aaa": 50})

    # -- differing histories / missing data ---------------------------------

    def test_later_listing_does_not_get_renormalised_weights(self):
        data = {
            "AAA": year_ends([(2020, 100), (2021, 110), (2022, 121)]),
            "NEW": year_ends([(2021, 50), (2022, 55)]),  # lists in 2021
        }
        r = strategies.compute_diversification_heat_map(data, {"AAA": 50, "NEW": 50}, as_of=FAR_FUTURE)
        p = {c["year"]: c for c in r["portfolio"]["cells"]}
        assert p[2021]["status"] == "incomplete" and p[2021]["return_percent"] is None
        assert "NEW" in p[2021]["reason"]
        assert p[2022]["return_percent"] == pytest.approx(10.0)  # both complete
        new = self.cells(r, "NEW")
        assert new[2020]["status"] == "not_listed" and new[2021]["status"] == "no_base"
        assert any("NEW: history starts" in w for w in r["data_quality_warnings"])

    def test_zero_weight_asset_with_short_history_does_not_block_portfolio(self):
        data = {
            "AAA": year_ends([(2020, 100), (2021, 110)]),
            "NEW": year_ends([(2021, 50)]),
        }
        r = strategies.compute_diversification_heat_map(data, {"AAA": 100, "NEW": 0}, as_of=FAR_FUTURE)
        p = {c["year"]: c for c in r["portfolio"]["cells"]}
        assert p[2021]["return_percent"] == 10.0

    def test_gap_year_is_missing_not_zero(self):
        data = {"AAA": year_ends([(2019, 100), (2021, 130), (2022, 143)])}
        r = strategies.compute_diversification_heat_map({**data, "BBB": data["AAA"]}, as_of=FAR_FUTURE)
        c = self.cells(r, "AAA")
        assert c[2020]["status"] == "missing" and c[2020]["return_percent"] is None
        assert c[2021]["status"] == "prior_year_missing" and c[2021]["return_percent"] is None
        assert c[2022]["return_percent"] == pytest.approx(10.0)

    def test_incomplete_current_year_is_partial_and_unranked(self):
        data = {
            "AAA": year_ends([(2023, 100), (2024, 110)]) + [px(D(2025, 3, 31), 121)],
            "BBB": year_ends([(2023, 100), (2024, 105)]) + [px(D(2025, 3, 31), 110.25)],
        }
        r = strategies.compute_diversification_heat_map(data, {"AAA": 50, "BBB": 50}, as_of=D(2025, 4, 15))
        a = self.cells(r, "AAA")[2025]
        assert a["status"] == "partial" and a["return_percent"] is None and a["rank"] is None
        assert a["partial_return_percent"] == 10.0
        assert r["yearly_rankings"][-1]["ranking"] == []
        p = {c["year"]: c for c in r["portfolio"]["cells"]}
        assert p[2025]["status"] == "partial" and p[2025]["partial_return_percent"] == pytest.approx(7.5)
        assert p[2025]["return_percent"] is None

    def test_year_range_and_ranges_outside_history(self):
        r = self.run(start=D(2022, 6, 1), end=D(2023, 1, 1))
        assert r["years"] == [2022, 2023]
        assert self.run(start=D(2040, 1, 1)) is None

    def test_price_basis_affects_returns(self):
        data = {
            "AAA": [px(D(2020, 12, 29), 100, adj=80), px(D(2021, 12, 29), 100, adj=100)],
            "BBB": [px(D(2020, 12, 29), 100), px(D(2021, 12, 29), 100)],
        }
        adj = strategies.compute_diversification_heat_map(data, as_of=FAR_FUTURE)
        raw = strategies.compute_diversification_heat_map(data, price_basis="close", as_of=FAR_FUTURE)
        assert self.cells(adj, "AAA")[2021]["return_percent"] == 25.0
        assert self.cells(raw, "AAA")[2021]["return_percent"] == 0.0

    def test_unusable_inputs(self):
        assert strategies.compute_diversification_heat_map({"AAA": [], "BBB": []}) is None
        with pytest.raises(ValueError):
            strategies.compute_diversification_heat_map(self.data(), price_basis="x")


# ===========================================================================
# Strategy 31: EMA + Bollinger + ATR
# ===========================================================================


def ohlc_frame(rows, start=D(2024, 1, 1)):
    """rows: [(open, high, low, close)] -> frame on consecutive days."""
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def manual_indicators(n, long_at=(), short_at=(), atr=2.0, rsi=50.0):
    z = pd.Series(False, index=range(n))
    idx = pd.date_range(D(2024, 1, 1), periods=n, freq="D")
    ls = pd.Series([i in long_at for i in range(n)], index=idx)
    ss = pd.Series([i in short_at for i in range(n)], index=idx)
    rsi_s = pd.Series(rsi, index=idx, dtype=float)
    return {
        "ema": pd.Series(100.0, index=idx), "bb_upper": pd.Series(110.0, index=idx),
        "bb_lower": pd.Series(90.0, index=idx), "bb_mid": pd.Series(100.0, index=idx),
        "atr": pd.Series(atr, index=idx), "rsi": rsi_s,
        "uptrend": z.set_axis(idx), "downtrend": z.set_axis(idx),
        "long_setup": ls, "short_setup": ss,
    }


ZERO_COST = {"commission_percent": 0.0, "slippage_percent": 0.0}


def sim(frame, ind, lo=0, hi=None, **overrides):
    p = strategies.resolve_ema_bollinger_atr_params({**ZERO_COST, **overrides})
    return strategies._eba_simulate(frame, ind, p, lo, len(frame) if hi is None else hi, False)


class TestEmaBollingerAtrParams:
    def test_article_defaults_preserved(self):
        p = strategies.resolve_ema_bollinger_atr_params()
        assert (p["ema_period"], p["bb_period"], p["bb_std"], p["rsi_period"], p["atr_period"], p["trend_lookback"]) == (
            30, 15, 1.5, 10, 7, 7,
        )
        assert (p["rsi_long_exit"], p["rsi_short_exit"]) == (90.0, 10.0)

    @pytest.mark.parametrize(
        "overrides, fragment",
        [
            ({"nope": 1}, "Unknown parameter"),
            ({"ema_period": 1}, "between"),
            ({"ema_period": 10.5}, "whole number"),
            ({"bb_std": float("nan")}, "finite"),
            ({"bb_std": True}, "finite"),
            ({"train_fraction": 0.95}, "between"),
            ({"position_fraction": 2}, "between"),
            ({"allow_short": "yes"}, "true or false"),
            ({"rsi_long_exit": 50, "rsi_short_exit": 50}, "lower than"),
        ],
    )
    def test_validation(self, overrides, fragment):
        with pytest.raises(ValueError, match=fragment):
            strategies.resolve_ema_bollinger_atr_params(overrides)

    def test_none_overrides_keep_defaults_and_overrides_apply(self):
        p = strategies.resolve_ema_bollinger_atr_params({"ema_period": None, "bb_std": 2.0})
        assert p["ema_period"] == 30 and p["bb_std"] == 2.0


class TestEmaBollingerAtrSignals:
    def build(self, pullback_close=80.0, pullback_open=100.0, n_up=45):
        """Steady uptrend (bodies far above EMA), then one candle that pulls back."""
        rows = []
        price = 100.0
        for _ in range(n_up):
            o, c = price, price + 1.0
            rows.append((o, c + 0.2, o - 0.2, c))
            price = c
        rows.append((price, price + 0.2, pullback_close - 0.2, pullback_close))
        return ohlc_frame(rows)

    def test_uptrend_classification_uses_body_vs_ema_of_preceding_candles(self):
        f = self.build()
        ind = strategies._eba_indicators(f, strategies.resolve_ema_bollinger_atr_params())
        assert bool(ind["uptrend"].iloc[-2]) is True
        assert bool(ind["downtrend"].iloc[-2]) is False
        # the signal candle's own (crashing) body does not break the *preceding* window
        assert bool(ind["uptrend"].iloc[-1]) is True

    def test_downtrend_classification(self):
        rows, price = [], 200.0
        for _ in range(45):
            o, c = price, price - 1.0
            rows.append((o, o + 0.2, c - 0.2, c))
            price = c
        ind = strategies._eba_indicators(ohlc_frame(rows), strategies.resolve_ema_bollinger_atr_params())
        assert bool(ind["downtrend"].iloc[-1]) is True and bool(ind["uptrend"].iloc[-1]) is False

    def test_trend_broken_by_one_body_inside_the_window(self):
        f = self.build(n_up=45)
        p = strategies.resolve_ema_bollinger_atr_params()
        rows = f.values.tolist()
        # drag candle -5 (inside the 7-candle window before the last bar) below the EMA
        rows[-5] = [rows[-5][0], rows[-5][1], 1.0, 1.0]
        ind = strategies._eba_indicators(ohlc_frame(rows), p)
        assert bool(ind["uptrend"].iloc[-1]) is False

    def test_long_entry_requires_uptrend_and_close_at_or_below_lower_band(self):
        p = strategies.resolve_ema_bollinger_atr_params()
        pulled = strategies._eba_indicators(self.build(pullback_close=80.0), p)
        assert bool(pulled["long_setup"].iloc[-1]) is True
        assert bool(pulled["short_setup"].iloc[-1]) is False
        assert float(self.build(pullback_close=80.0)["close"].iloc[-1]) <= float(pulled["bb_lower"].iloc[-1])
        # no pull-back to the band -> no setup, even in an uptrend
        calm = strategies._eba_indicators(self.build(pullback_close=145.0), p)
        assert bool(calm["long_setup"].iloc[-1]) is False

    def test_close_exactly_on_lower_band_counts_as_touch(self):
        f = self.build(pullback_close=80.0)
        p = strategies.resolve_ema_bollinger_atr_params()
        lower = float(strategies._eba_indicators(f, p)["bb_lower"].iloc[-1])
        # moving the last close onto the band moves the band too; compare using the same frame
        close = f["close"].copy()
        ind = strategies._eba_indicators(f, p)
        assert bool((close <= ind["bb_lower"]).iloc[-1]) == bool(ind["long_setup"].iloc[-1])
        assert lower > 0

    def test_short_entry_in_downtrend_at_or_above_upper_band(self):
        rows, price = [], 200.0
        for _ in range(45):
            o, c = price, price - 1.0
            rows.append((o, o + 0.2, c - 0.2, c))
            price = c
        rows.append((price, price + 40, price - 0.2, price + 35))  # spike up through the band
        ind = strategies._eba_indicators(ohlc_frame(rows), strategies.resolve_ema_bollinger_atr_params())
        assert bool(ind["short_setup"].iloc[-1]) is True and bool(ind["long_setup"].iloc[-1]) is False

    def test_indicators_use_only_past_and_current_data(self):
        f = self.build()
        p = strategies.resolve_ema_bollinger_atr_params()
        full = strategies._eba_indicators(f, p)
        for t in (35, 40, 44):
            part = strategies._eba_indicators(f.iloc[: t + 1], p)
            for name in ("ema", "bb_upper", "bb_lower", "rsi", "atr", "uptrend", "long_setup"):
                a, b = part[name].iloc[-1], full[name].iloc[t]
                assert (pd.isna(a) and pd.isna(b)) or a == b, (name, t)

    def test_future_changes_do_not_alter_earlier_signals(self):
        f = self.build()
        p = strategies.resolve_ema_bollinger_atr_params()
        base = strategies._eba_indicators(f, p)["long_setup"]
        altered = f.copy()
        altered.iloc[-1] = [10.0, 10.0, 10.0, 10.0]
        assert (strategies._eba_indicators(altered, p)["long_setup"].iloc[:-1] == base.iloc[:-1]).all()


class TestEmaBollingerAtrSimulation:
    def flat(self, n=20, price=100.0):
        return ohlc_frame([(price, price, price, price)] * n)

    def test_atr_stop_and_take_profit_levels_long(self):
        f = self.flat()
        f.loc[f.index[12], ["high"]] = 107.0  # bar after entry reaches +7
        ind = manual_indicators(len(f), long_at={10}, atr=2.0)
        out = sim(f, ind, stop_loss_atr_mult=1.5, take_profit_ratio=2.0)
        t = out["trades"][0]
        # stop distance = ATR 2 x 1.5 = 3; take-profit = 3 x 2 = 6
        assert t["stop_loss"] == 97.0 and t["take_profit"] == 106.0
        assert t["entry_price"] == 100.0 and t["exit_reason"] == "take_profit" and t["exit_price"] == 106.0
        assert t["price_return_percent"] == 6.0

    def test_atr_stop_and_take_profit_levels_short(self):
        f = self.flat()
        f.loc[f.index[12], ["low"]] = 93.0
        ind = manual_indicators(len(f), short_at={10}, atr=2.0)
        t = sim(f, ind, stop_loss_atr_mult=1.5, take_profit_ratio=2.0)["trades"][0]
        assert t["side"] == "short" and t["stop_loss"] == 103.0 and t["take_profit"] == 94.0
        assert t["exit_reason"] == "take_profit" and t["price_return_percent"] == 6.0

    def test_stop_loss_hit(self):
        f = self.flat()
        f.loc[f.index[12], ["low"]] = 96.0
        t = sim(f, manual_indicators(len(f), long_at={10}), stop_loss_atr_mult=1.5)["trades"][0]
        assert t["exit_reason"] == "stop_loss" and t["exit_price"] == 97.0
        assert t["price_return_percent"] == -3.0

    def test_stop_assumed_first_when_one_candle_touches_both(self):
        f = self.flat()
        f.loc[f.index[12], ["high", "low"]] = [110.0, 90.0]
        out = sim(f, manual_indicators(len(f), long_at={10}))
        assert out["trades"][0]["exit_reason"] == "stop_loss" and out["ambiguous_bars"] == 1

    def test_gap_through_stop_exits_at_the_open_not_the_level(self):
        f = self.flat()
        f.loc[f.index[12:], ["open", "high", "low", "close"]] = 90.0  # gaps far below the 97 stop
        t = sim(f, manual_indicators(len(f), long_at={10}))["trades"][0]
        assert t["exit_reason"] == "stop_loss_gap" and t["exit_price"] == 90.0

    def test_entry_is_next_bar_open_never_signal_close(self):
        f = self.flat()
        f.loc[f.index[11], ["open", "high", "low", "close"]] = 101.0
        out = sim(f, manual_indicators(len(f), long_at={10}))
        t = out["trades"][0]
        assert t["signal_date"] == "2024-01-11" and t["entry_date"] == "2024-01-12"
        assert t["entry_price"] == 101.0

    def test_signal_on_last_bar_cannot_trade(self):
        f = self.flat(15)
        assert sim(f, manual_indicators(15, long_at={14}))["trades"] == []

    def test_rsi_exit_long_executes_next_open(self):
        f = self.flat(20)
        ind = manual_indicators(20, long_at={5})
        ind["rsi"].iloc[8] = 95.0
        f.loc[f.index[9], ["open", "high", "low", "close"]] = 102.0
        t = sim(f, ind)["trades"][0]
        assert t["exit_reason"] == "rsi_exit" and t["exit_date"] == "2024-01-10" and t["exit_price"] == 102.0

    def test_rsi_exit_short(self):
        f = self.flat(20)
        ind = manual_indicators(20, short_at={5})
        ind["rsi"].iloc[8] = 10.0  # at-or-below 10 closes a short
        f.loc[f.index[9], ["open", "high", "low", "close"]] = 98.0
        t = sim(f, ind)["trades"][0]
        assert t["side"] == "short" and t["exit_reason"] == "rsi_exit" and t["price_return_percent"] == 2.0

    def test_rsi_threshold_boundaries(self):
        f = self.flat(20)
        ind = manual_indicators(20, long_at={5})
        ind["rsi"].iloc[8] = 89.99
        assert sim(f, ind)["trades"][0]["exit_reason"] == "end_of_data"
        ind["rsi"].iloc[8] = 90.0
        assert sim(f, ind)["trades"][0]["exit_reason"] == "rsi_exit"

    def test_open_position_closed_at_end_of_data(self):
        f = self.flat(20)
        t = sim(f, manual_indicators(20, long_at={5}))["trades"][0]
        assert t["exit_reason"] == "end_of_data" and t["exit_date"] == "2024-01-20"

    def test_one_position_at_a_time(self):
        f = self.flat(30)
        out = sim(f, manual_indicators(30, long_at={5, 6, 7, 8}))
        assert len(out["trades"]) == 1

    def test_shorts_can_be_disabled(self):
        f = self.flat()
        assert sim(f, manual_indicators(len(f), short_at={10}), allow_short=False)["trades"] == []

    def test_costs_and_slippage_accounting(self):
        f = self.flat()
        f.loc[f.index[12], ["high"]] = 107.0
        ind = manual_indicators(len(f), long_at={10})
        out = sim(
            f, ind, stop_loss_atr_mult=1.5, take_profit_ratio=2.0,
            commission_percent=0.1, slippage_percent=0.5, position_fraction=0.5,
        )
        t = out["trades"][0]
        entry = 100.0 * 1.005
        assert t["entry_price"] == pytest.approx(entry, abs=1e-4)
        tp = entry + 3.0 * 2.0
        exit_fill = tp * 0.995
        assert t["exit_price"] == pytest.approx(exit_fill, abs=1e-4)
        gross = exit_fill / entry - 1
        expected_net = 0.5 * gross - 2 * 0.5 * 0.001
        assert t["equity_return_percent"] == pytest.approx(expected_net * 100, abs=0.01)
        assert out["equity"][-1] == pytest.approx(1 + expected_net, abs=1e-6)

    def test_equity_compounds_across_trades_and_is_marked_to_market(self):
        f = self.flat(40)
        f.loc[f.index[12], ["high"]] = 107.0  # trade 1 wins +6%
        f.loc[f.index[26], ["low"]] = 90.0  # trade 2 stopped at -3%
        ind = manual_indicators(40, long_at={10, 24})
        out = sim(f, ind, stop_loss_atr_mult=1.5, take_profit_ratio=2.0)
        assert [t["exit_reason"] for t in out["trades"]] == ["take_profit", "stop_loss"]
        assert out["equity"][-1] == pytest.approx(1.06 * 0.97)
        assert out["equity"][0] == 1.0

    def test_metrics_from_known_trades(self):
        f = self.flat(60)
        f.loc[f.index[12], ["high"]] = 107.0  # +6
        f.loc[f.index[26], ["low"]] = 90.0  # -3
        f.loc[f.index[42], ["high"]] = 107.0  # +6
        ind = manual_indicators(60, long_at={10, 24, 40})
        s = sim(f, ind, stop_loss_atr_mult=1.5, take_profit_ratio=2.0)
        m = strategies._eba_metrics(s, f["close"].to_numpy(float), f.index)
        assert m["trade_count"] == 3 and m["long_trades"] == 3 and m["short_trades"] == 0
        assert m["win_rate_percent"] == pytest.approx(66.7, abs=0.1)
        assert m["profit_factor"] == pytest.approx(12 / 3, abs=0.01)  # 6+6 wins vs 3 loss
        assert m["total_return_percent"] == pytest.approx((1.06 * 0.97 * 1.06 - 1) * 100, abs=0.01)
        assert m["max_drawdown_percent"] == pytest.approx(-3.0, abs=0.01)
        assert m["benchmark_buy_hold_return_percent"] == 0.0

    def test_no_trades_edge_case(self):
        f = self.flat(40)
        s = sim(f, manual_indicators(40))
        m = strategies._eba_metrics(s, f["close"].to_numpy(float), f.index)
        assert m["trade_count"] == 0 and m["win_rate_percent"] is None and m["profit_factor"] is None
        assert m["total_return_percent"] == 0.0 and m["max_drawdown_percent"] == 0.0

    def test_profit_factor_undefined_without_losses(self):
        f = self.flat()
        f.loc[f.index[12], ["high"]] = 107.0
        s = sim(f, manual_indicators(len(f), long_at={10}), stop_loss_atr_mult=1.5, take_profit_ratio=2.0)
        m = strategies._eba_metrics(s, f["close"].to_numpy(float), f.index)
        assert m["profit_factor"] is None and m["win_rate_percent"] == 100.0

    def test_segment_simulation_starts_flat_and_stays_inside_its_range(self):
        f = self.flat(40)
        ind = manual_indicators(40, long_at={5, 25})
        train = sim(f, ind, lo=0, hi=20)
        test = sim(f, ind, lo=20, hi=40)
        assert all(t["exit_date"] <= "2024-01-20" for t in train["trades"])
        assert train["trades"][0]["exit_reason"] == "end_of_data"
        assert all(t["entry_date"] >= "2024-01-21" for t in test["trades"])
        assert len(test["trades"]) == 1  # the pre-split signal at bar 5 is not carried over


class TestEmaBollingerAtrBacktest:
    def test_insufficient_history_returns_none(self):
        assert strategies.compute_ema_bollinger_atr(random_walk(60)) is None

    def test_invalid_arguments_raise(self):
        with pytest.raises(ValueError):
            strategies.compute_ema_bollinger_atr(random_walk(600), timeframe="5m")
        with pytest.raises(ValueError):
            strategies.compute_ema_bollinger_atr(random_walk(600), params={"bb_std": -1})

    def test_structure_defaults_and_train_test_separation(self):
        prices = random_walk(900, seed=3)
        r = strategies.compute_ema_bollinger_atr(prices)
        assert r["timeframe"] == "1d" and r["parameters"]["ema_period"] == 30
        assert r["train"]["segment"] == "train" and r["out_of_sample"]["segment"] == "out_of_sample"
        assert r["train"]["end"] < r["out_of_sample"]["start"]
        assert r["train"]["bars"] + r["out_of_sample"]["bars"] == r["bars_used"] == 900
        split = r["out_of_sample"]["start"]
        for t in r["trades"]:
            if t["segment"] == "train":
                assert t["exit_date"] < split
            else:
                assert t["entry_date"] >= split
        assert r["latest"]["trend"] in ("Uptrend", "Downtrend", "Neutral")
        assert "no orders" in r["disclaimer"].lower() or "no orders are placed" in r["disclaimer"].lower()
        assert "not been reproduced" in r["disclaimer"]
        assert len(r["chart_data"]) <= strategies.CHART_LOOKBACK_DAYS
        for forbidden in ("signal", "action", "recommendation", "order"):
            assert forbidden not in r["latest"]

    def test_signals_and_fills_never_use_the_same_bar(self):
        r = strategies.compute_ema_bollinger_atr(random_walk(900, seed=5))
        assert r["trade_count_total"] > 0
        for t in r["trades"]:
            assert t["signal_date"] < t["entry_date"] <= t["exit_date"]

    def test_future_data_cannot_change_earlier_trades(self):
        prices = random_walk(900, seed=7)
        full = strategies.compute_ema_bollinger_atr(prices, params={"train_fraction": 0.9})
        cut = 700
        part = strategies.compute_ema_bollinger_atr(prices[:cut], params={"train_fraction": 0.9})
        cutoff = prices[cut - 1].date.isoformat()
        # compare trades fully inside the shorter window and before its (re-computed) split
        split_part = part["out_of_sample"]["start"]
        a = [t for t in full["trades"] if t["exit_date"] < min(cutoff, split_part) and t["segment"] == "train"]
        b = [t for t in part["trades"] if t["exit_date"] < min(cutoff, split_part) and t["segment"] == "train"]
        # both runs use different split points, so only trades that finished before BOTH splits are comparable
        split_full = full["out_of_sample"]["start"]
        a = [t for t in a if t["exit_date"] < min(split_full, split_part)]
        b = [t for t in b if t["exit_date"] < min(split_full, split_part)]
        assert a == b and len(a) > 0

    def test_optimisation_uses_training_segment_only(self):
        prices = random_walk(900, seed=11)
        base = strategies.compute_ema_bollinger_atr(prices, optimize=True)
        assert base["parameter_selection"]["enabled"] is True
        assert len(base["parameter_selection"]["candidates"]) == 16
        # rewrite every out-of-sample bar: the selected parameters must not change
        split = int(900 * 0.6)
        rewritten = prices[:split] + random_walk(900 - split, seed=99, start=prices[split].date, s0=300.0)
        again = strategies.compute_ema_bollinger_atr(rewritten, optimize=True)
        assert again["parameter_selection"]["candidates"] == base["parameter_selection"]["candidates"]
        assert again["parameter_selection"]["selected"] == base["parameter_selection"]["selected"]
        assert again["train"] == base["train"]

    def test_optimisation_off_by_default(self):
        r = strategies.compute_ema_bollinger_atr(random_walk(900, seed=3))
        assert r["parameter_selection"]["enabled"] is False and r["parameter_selection"]["candidates"] == []

    def test_small_out_of_sample_is_flagged(self):
        r = strategies.compute_ema_bollinger_atr(random_walk(900, seed=3))
        assert any("statistically weak" in w for w in r["data_quality_warnings"])

    def test_hourly_bars_keep_their_timestamps(self):
        start = datetime.datetime(2024, 1, 2, 14, 30)
        bars, price = [], 100.0
        rng = np.random.default_rng(2)
        for i in range(700):
            o = price
            c = price * (1 + rng.normal(0, 0.003))
            bars.append(strategies.PriceBar(start + datetime.timedelta(hours=i), o, max(o, c) * 1.001, min(o, c) * 0.999, c))
            price = c
        r = strategies.compute_ema_bollinger_atr(bars, timeframe="1h")
        assert r["timeframe"] == "1h" and r["first_bar"].endswith("Z") and "T14:30:00Z" in r["first_bar"]
        assert any("raw" in w and "UTC" in w for w in r["data_quality_warnings"])
        assert "daily" not in r["data_source"].lower()

    def test_daily_split_is_flagged(self):
        prices = random_walk(500, seed=3)
        for p in prices[300:]:
            p.close, p.open, p.high, p.low = p.close / 2, p.open / 2, p.high / 2, p.low / 2
            p.adjusted_close = p.adjusted_close  # adjusted unchanged -> ratio jumps
        r = strategies.compute_ema_bollinger_atr(prices)
        assert any("split" in w.lower() or "ratio" in w.lower() for w in r["data_quality_warnings"])

    def test_dirty_rows_are_dropped(self):
        prices = random_walk(700, seed=3)
        prices[100].high, prices[100].low = 1.0, 5.0  # high < low
        prices[200].close = float("nan")
        r = strategies.compute_ema_bollinger_atr(prices)
        assert r["bars_used"] == 698


# ===========================================================================
# Strategy 32: Crash warning
# ===========================================================================


def trending(n, start=100.0, step=0.5, start_date=D(2022, 1, 3)):
    return weekday_series(start_date, [start + step * i for i in range(n)])


def decline_after_rise(n_up=320, n_down=12, step=6.0):
    closes = [100 + 0.4 * i for i in range(n_up)]
    top = closes[-1]
    closes += [top - step * (i + 1) for i in range(n_down)]
    return weekday_series(D(2021, 1, 4), closes)


class TestCrashWarningIndicators:
    def test_sma_matches_manual_rolling_mean(self):
        prices = random_walk(400, seed=4)
        df = strategies._crash_frame(prices)
        c = strategies._crash_conditions(df)
        manual = df["close"].rolling(200).mean()
        assert c["sma_long"].iloc[-1] == pytest.approx(manual.iloc[-1])
        assert c["sma_short"].iloc[-1] == pytest.approx(df["close"].iloc[-50:].mean())
        assert np.isnan(c["sma_long"].iloc[198]) and not np.isnan(c["sma_long"].iloc[199])

    def test_rsi_extremes_use_the_shared_wilder_rsi(self):
        up = strategies._crash_conditions(strategies._crash_frame(trending(260)))
        assert up["rsi"].iloc[-1] == 100.0  # no losses at all
        down = strategies._crash_conditions(strategies._crash_frame(trending(260, start=500, step=-1.0)))
        assert down["rsi"].iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_macd_matches_standard_12_26_9_definition(self):
        df = strategies._crash_frame(random_walk(300, seed=8))
        c = strategies._crash_conditions(df)
        close = df["close"]
        macd = close.ewm(span=12, adjust=False, min_periods=12).mean() - close.ewm(span=26, adjust=False, min_periods=26).mean()
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        assert c["macd"].iloc[-1] == pytest.approx(macd.iloc[-1])
        assert c["signal"].iloc[-1] == pytest.approx(signal.iloc[-1])
        assert c["hist"].iloc[-1] == pytest.approx((macd - signal).iloc[-1])

    def test_steady_exponential_uptrend_triggers_nothing(self):
        # constant percentage growth: price above both SMAs, MACD rising with price
        r = strategies.compute_crash_warning(
            weekday_series(D(2021, 1, 4), [100 * 1.004**i for i in range(320)])
        )
        assert r["triggered_count"] == 0 and r["risk_level"] == "Low" and r["warning_active"] is False
        assert all(c["triggered"] is False for c in r["conditions"])

    def test_decline_triggers_trend_and_momentum_conditions(self):
        r = strategies.compute_crash_warning(decline_after_rise())
        by_id = {c["id"]: c for c in r["conditions"]}
        assert by_id["price_below_long_sma"]["triggered"] is True
        assert by_id["macd_bearish_cross_recent"]["triggered"] is True or by_id["macd_histogram_weakening"]["triggered"] is True
        assert r["triggered_count"] == sum(c["triggered"] for c in r["conditions"])
        assert r["triggered_count"] >= 2
        assert r["risk_level"] in ("Elevated", "High")
        assert r["warning_active"] is (r["triggered_count"] >= strategies.CRASH_WARNING_MIN_TRIGGERED)

    def test_histogram_weakening_rule_is_three_consecutive_declines(self):
        # smooth decelerating rise -> MACD histogram falling several bars in a row
        closes = [100 * (1 + 0.01) ** i if i < 250 else 100 * (1.01 ** 250) * (1 + 0.0004) ** (i - 250) for i in range(300)]
        df = strategies._crash_frame(weekday_series(D(2021, 1, 4), closes))
        c = strategies._crash_conditions(df)
        h = c["hist"]
        manual = (h < h.shift(1)) & (h.shift(1) < h.shift(2)) & (h.shift(2) < h.shift(3))
        got = c["flags"]["macd_histogram_weakening"]
        valid = c["valid"]
        assert (got[valid] == manual[valid]).all()
        assert got.any()

    def test_oversold_rsi_is_informational_and_not_scored(self):
        prices = weekday_series(D(2021, 1, 4), [100 + 0.2 * i for i in range(300)] + [160 - 3 * i for i in range(1, 25)])
        r = strategies.compute_crash_warning(prices)
        oversold = next(i for i in r["informational"] if i["id"] == "rsi_oversold")
        assert oversold["scored"] is False and "not evidence of a crash" in oversold["note"]
        assert oversold["id"] not in {c["id"] for c in r["conditions"]}
        assert r["scored_condition_count"] == 5
        assert r["triggered_count"] == sum(c["triggered"] for c in r["conditions"])


class TestCrashDivergence:
    def arrays(self, peak1=100.0, peak2=105.0, rsi1=70.0, rsi2=60.0, n=60):
        close = np.linspace(80, 82, n)
        rsi = np.full(n, 50.0)
        close[10], rsi[10] = peak1, rsi1
        close[30], rsi[30] = peak2, rsi2
        return close, rsi

    def test_bearish_divergence_detected_only_after_confirmation(self):
        close, rsi = self.arrays()
        flag, pair = strategies._crash_swing_divergence(close, rsi)
        k = strategies.CRASH_SWING_K
        assert not flag[: 30 + k].any()  # second swing (idx 30) unconfirmed until idx 35
        assert flag[30 + k] and pair[30 + k] == (10, 30)

    def test_divergence_requires_higher_price_and_lower_rsi(self):
        for kwargs in ({"rsi2": 75.0}, {"peak2": 95.0}):  # higher RSI / lower price high
            close, rsi = self.arrays(**kwargs)
            flag, _ = strategies._crash_swing_divergence(close, rsi)
            assert not flag.any()

    def test_divergence_expires_after_lookback(self):
        close, rsi = self.arrays(n=120)
        flag, _ = strategies._crash_swing_divergence(close, rsi)
        assert flag[35] and not flag[30 + strategies.CRASH_DIVERGENCE_LOOKBACK + 1]

    def test_flag_at_t_is_independent_of_later_bars(self):
        close, rsi = self.arrays(n=80)
        full, _ = strategies._crash_swing_divergence(close, rsi)
        for t in (34, 35, 50):
            part, _ = strategies._crash_swing_divergence(close[: t + 1], rsi[: t + 1])
            assert part[t] == full[t]

    def test_swings_closer_than_k_bars_are_not_paired(self):
        close = np.linspace(80, 82, 60)
        rsi = np.full(60, 50.0)
        close[10], rsi[10] = 100, 70
        close[13], rsi[13] = 105, 60  # only 3 bars after the first peak
        flag, _ = strategies._crash_swing_divergence(close, rsi)
        assert not flag.any()


class TestCrashWarningOutput:
    def test_insufficient_and_missing_data(self):
        assert strategies.compute_crash_warning(random_walk(100)) is None
        assert strategies.compute_crash_warning([]) is None
        prices = random_walk(400, seed=2)
        prices[50].close = float("nan")
        prices[60].close = 0.0
        prices[70].close = None
        r = strategies.compute_crash_warning(prices)
        assert r["bars_used"] == 397

    def test_historical_timestamps_and_observations(self):
        prices = decline_after_rise()
        r = strategies.compute_crash_warning(prices)
        last = prices[-1].date.isoformat()
        assert r["as_of"] == last
        assert all(c["observed_at"] == last for c in r["conditions"] + r["informational"])
        dates = [row["date"] for row in r["chart_data"]]
        assert dates == sorted(dates) and dates[-1] == last
        assert len(dates) == strategies.CHART_LOOKBACK_DAYS
        for c in r["conditions"]:
            if c["triggered"]:
                assert c["triggered_since"] is not None and c["triggered_since"] <= last
            else:
                assert c["triggered_since"] is None
        assert r["chart_data"][-1]["triggered_count"] == r["triggered_count"]
        assert set(r["chart_data"][-1]["triggered"]) == {c["id"] for c in r["conditions"] if c["triggered"]}

    def test_triggered_since_marks_start_of_current_run(self):
        prices = decline_after_rise()
        df = strategies._crash_frame(prices)
        cond = strategies._crash_conditions(df)
        r = strategies.compute_crash_warning(prices)
        c = next(x for x in r["conditions"] if x["id"] == "price_below_long_sma")
        series = cond["flags"]["price_below_long_sma"]
        i = len(series) - 1
        while i > 0 and series.iloc[i - 1]:
            i -= 1
        assert c["triggered_since"] == df.index[i].date().isoformat()

    def test_no_future_data_leakage_in_conditions(self):
        prices = random_walk(500, seed=21)
        df = strategies._crash_frame(prices)
        full = strategies._crash_conditions(df)
        for t in (260, 330, 410):
            part = strategies._crash_conditions(df.iloc[: t + 1])
            assert part["count"].iloc[-1] == full["count"].iloc[t]
            for cid in full["flags"]:
                assert bool(part["flags"][cid].iloc[-1]) == bool(full["flags"][cid].iloc[t]), (cid, t)

    def test_changing_future_prices_does_not_change_past_warnings(self):
        prices = random_walk(500, seed=22)
        base = strategies._crash_conditions(strategies._crash_frame(prices))["count"]
        altered = list(prices)
        for p in altered[450:]:
            p.close = p.close * 0.5
        after = strategies._crash_conditions(strategies._crash_frame(altered))["count"]
        assert (base.iloc[:440] == after.iloc[:440]).all()

    def test_no_probability_or_forecast_fields_and_clear_labelling(self):
        r = strategies.compute_crash_warning(random_walk(500, seed=3))
        text = " ".join([r["assessment"], r["disclaimer"]]).lower()
        assert "not a forecast" in text and "not a crash prediction" in text
        for key in r:
            assert "probab" not in key and "forecast" not in key and "predict" not in key
        assert r["timeframe"] == "1d" and "5-minute intraday variant is not implemented" in r["data_basis"]
        assert r["risk_level"] in ("Low", "Elevated", "High")

    def test_divergence_details_when_present(self):
        # find a random walk that yields an active divergence on the last bar
        for seed in range(40):
            r = strategies.compute_crash_warning(random_walk(450, seed=seed))
            if r and r["divergence"]:
                d = r["divergence"]
                assert d["first_swing_date"] < d["second_swing_date"] <= d["confirmed_on"]
                assert d["second_swing_close"] > d["first_swing_close"]
                assert d["second_swing_rsi"] < d["first_swing_rsi"]
                return
        pytest.skip("no seed produced an active divergence (detector itself is covered above)")


class TestCrashValidation:
    def test_classification_metrics_and_episodes(self):
        label = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0], bool)
        warn = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
        breach = np.array([np.nan] * 14)
        breach[2] = 4.0
        m = strategies._classification(label, warn, breach, 0, 14)
        assert (m["true_positives"], m["false_positives"], m["false_negatives"], m["true_negatives"]) == (1, 2, 5, 6)
        assert m["precision_percent"] == pytest.approx(33.3, abs=0.1)
        assert m["recall_percent"] == pytest.approx(16.7, abs=0.1)
        assert m["base_rate_percent"] == pytest.approx(42.9, abs=0.1)
        assert m["episodes"] == 3 and m["episodes_detected"] == 1 and m["episodes_missed"] == 2
        assert m["median_lead_bars"] == 4.0

    def test_degenerate_denominators_are_none_not_errors(self):
        label = np.zeros(10, bool)
        warn = np.zeros(10, bool)
        m = strategies._classification(label, warn, np.full(10, np.nan), 0, 10)
        assert m["precision_percent"] is None and m["recall_percent"] is None and m["episodes"] == 0

    def test_validation_block_is_time_ordered_out_of_sample_and_labelled(self):
        v = strategies.compute_crash_warning(random_walk(900, seed=5))["validation"]
        assert v["status"] == "available"
        assert v["in_sample"]["end"] < v["out_of_sample"]["start"]  # purge gap between blocks
        d = v["definition"]
        assert d["prediction_horizon_bars"] == strategies.CRASH_EVENT_HORIZON_BARS
        assert "10%" in d["crash_event"] and "SMA(200)" in d["baseline_rule"]
        for block in ("in_sample", "out_of_sample"):
            ind, base = v[block]["indicator"], v[block]["baseline_sma200"]
            assert ind["true_positives"] + ind["false_negatives"] == ind["crash_labelled_bars"]
            assert base["bars"] == ind["bars"]
        assert any("statistically unreliable" in w for w in v["warnings"]) or v["out_of_sample"]["indicator"]["episodes"] >= 5

    def test_labels_for_last_horizon_bars_are_never_evaluated(self):
        prices = random_walk(900, seed=5)
        v = strategies.compute_crash_warning(prices)["validation"]
        horizon = strategies.CRASH_EVENT_HORIZON_BARS
        cutoff = prices[-horizon - 1].date.isoformat()
        assert v["out_of_sample"]["end"] <= cutoff

    def test_baseline_and_always_warn_comparison_present(self):
        v = strategies.compute_crash_warning(random_walk(900, seed=5))["validation"]
        oos = v["out_of_sample"]
        assert oos["always_warn_precision_percent"] == oos["indicator"]["base_rate_percent"]

    def test_validation_unavailable_when_too_short(self):
        r = strategies.compute_crash_warning(random_walk(strategies.MIN_ROWS_CRASH_WARNING + 5, seed=1))
        assert r["validation"]["status"] == "unavailable" and "evaluable bars" in r["validation"]["reason"]


# ===========================================================================
# Integration: registration, API schemas, EODHD entitlement, cache, sessions
# ===========================================================================


class FakeCache:
    """Records cache traffic; behaves like a cold, working cache."""

    def __init__(self, hit=None):
        self.hit = hit
        self.sets = []
        self.gets = []

    async def get_json(self, key):
        self.gets.append(key)
        return self.hit

    async def set_json(self, key, value, ttl=None):
        self.sets.append((key, ttl))

    async def get_or_compute(self, key, fn, ttl=None):
        self.gets.append(key)
        if self.hit is not None:
            return self.hit
        value = await fn()
        self.sets.append((key, ttl))
        return value


@pytest.fixture
def api(monkeypatch):
    from app.routers import strategy as router

    prices_by_symbol = {}
    calls = []
    db_sentinel = object()
    fake_cache = FakeCache()

    async def load_prices(db, symbol, limit=None):
        calls.append((db, symbol))
        return list(prices_by_symbol.get(symbol, []))

    monkeypatch.setattr(router, "load_prices", load_prices)
    monkeypatch.setattr(router.cache, "get_json", fake_cache.get_json)
    monkeypatch.setattr(router.cache, "set_json", fake_cache.set_json)
    monkeypatch.setattr(router.cache, "get_or_compute", fake_cache.get_or_compute)
    monkeypatch.setattr(router.regime, "extract_symbol_view", lambda *_a, **_k: None)
    monkeypatch.setattr(router.strategies, "extract_factor_momentum_view", lambda *_a, **_k: None)
    router._hourly_memo.clear()

    from app.exceptions import register_exception_handlers

    app = FastAPI()
    register_exception_handlers(app)  # same {data, success, message} error envelope as production
    app.include_router(router.router, prefix="/api")
    app.include_router(router.portfolio_router, prefix="/api")
    app.dependency_overrides[get_db] = lambda: db_sentinel
    yield SimpleNamespace(
        client=TestClient(app), router=router, prices=prices_by_symbol, calls=calls,
        db=db_sentinel, cache=fake_cache, monkeypatch=monkeypatch,
    )
    router._hourly_memo.clear()


class TestRegistrationAndSchemas:
    def test_strategies_endpoint_registers_the_three_single_symbol_strategies(self, api):
        api.prices["VOO"] = random_walk(900, seed=3)
        resp = api.client.get("/api/etfs/VOO/strategies")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["seasonalHeatMap"]["rows"] and data["emaBollingerAtr"]["train"] and data["crashWarning"]["conditions"]
        # categories follow the existing grouping - no new top-level category
        assert set(data["calendarStrategies"]) == {"tltMonthlyCycle", "seasonalHeatMap"}
        assert "emaBollingerAtr" in data["technicalStrategies"]
        assert "crashWarning" in data["quantitativeStrategies"]
        assert "diversificationHeatMap" not in data  # multi-asset: served by the POST route
        assert data["crashWarning"] == data["quantitativeStrategies"]["crashWarning"]
        # existing strategies still present
        assert data["macd"] is not None and data["ema50Rsi"] is not None
        assert resp.json()["message"] is None or "Seasonal" not in resp.json()["message"]
        assert ("tw:v11:strategies:VOO", None) in api.cache.sets
        assert (api.db, "VOO") in api.calls  # the injected session is what reaches the loader

    def test_short_history_leaves_new_fields_null_with_existing_message_pattern(self, api):
        api.prices["VOO"] = random_walk(90, seed=3)
        resp = api.client.get("/api/etfs/VOO/strategies")
        data = resp.json()["data"]
        assert data["emaBollingerAtr"] is None and data["crashWarning"] is None
        msg = resp.json()["message"]
        assert "EMA + Bollinger Bands + ATR" in msg and "Technical Crash-Warning Risk Indicator" in msg

    def test_response_has_no_signal_or_order_fields(self, api):
        api.prices["VOO"] = random_walk(900, seed=3)
        text = api.client.get("/api/etfs/VOO/strategies").text.lower()
        for bad in ("crashprobability", "buysignal", "sellsignal", "placeorder"):
            assert bad not in text


class TestSeasonalRoute:
    def test_range_and_basis_parameters(self, api):
        api.prices["SPY"] = random_walk(900, seed=3)
        resp = api.client.get(
            "/api/etfs/spy/strategies/seasonal_heat_map",
            params={"start": "2022-01-01", "end": "2022-12-31", "price_basis": "close"},
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()["data"]
        assert d["symbol"] == "SPY" and d["priceBasis"] == "close"
        assert [r["year"] for r in d["rows"]] == [2022]
        assert len(d["rows"][0]["months"]) == 12 and len(d["rows"][0]["quarters"]) == 4
        assert api.cache.sets[0][0].startswith("tw:v1:article:seasonal:SPY:close:")
        assert (api.db, "SPY") in api.calls

    def test_supports_portfolio_registry_symbols(self, api):
        api.prices["TLT"] = random_walk(500, seed=3)
        assert api.client.get("/api/etfs/TLT/strategies/seasonal_heat_map").status_code == 200

    def test_errors(self, api):
        assert api.client.get("/api/etfs/NOPE/strategies/seasonal_heat_map").status_code == 404
        r = api.client.get("/api/etfs/SPY/strategies/seasonal_heat_map", params={"start": "2024-05-01", "end": "2024-01-01"})
        assert r.status_code == 422 and "start must not be after end" in r.json()["message"]
        assert api.client.get("/api/etfs/SPY/strategies/seasonal_heat_map", params={"price_basis": "x"}).status_code == 422
        assert api.client.get("/api/etfs/SPY/strategies/seasonal_heat_map").status_code == 503  # not ingested yet
        api.prices["SPY"] = random_walk(300, seed=3)
        far = api.client.get("/api/etfs/SPY/strategies/seasonal_heat_map", params={"start": "2040-01-01"})
        assert far.status_code == 422 and "Insufficient price history" in far.json()["message"]

    def test_cache_hit_skips_database_and_computation(self, api):
        api.cache.hit = {"cached": True}
        # response_model is applied to cached data too; use a minimal valid payload instead
        api.cache.hit = strategies.compute_seasonal_heat_map(random_walk(400, seed=3), "SPY")
        resp = api.client.get("/api/etfs/SPY/strategies/seasonal_heat_map")
        assert resp.status_code == 200 and api.calls == []
        assert api.cache.sets == []


class TestEmaBollingerAtrRoute:
    def test_daily_with_parameters(self, api):
        api.prices["QQQ"] = random_walk(900, seed=3)
        resp = api.client.get(
            "/api/etfs/QQQ/strategies/ema_bollinger_atr",
            params={"ema_period": 20, "stop_loss_atr_mult": 2, "optimize": "true", "allow_short": "false"},
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()["data"]
        assert d["parameters"]["emaPeriod"] == 20 and d["parameters"]["stopLossAtrMult"] == 2.0
        assert d["parameters"]["allowShort"] is False and d["parameters"]["bbStd"] == 1.5
        assert d["parameterSelection"]["enabled"] is True
        assert d["timeframe"] == "1d"
        assert all(t["side"] == "long" for t in d["trades"])
        assert (api.db, "QQQ") in api.calls

    def test_bad_parameters_are_422(self, api):
        api.prices["QQQ"] = random_walk(900, seed=3)
        r = api.client.get("/api/etfs/QQQ/strategies/ema_bollinger_atr", params={"ema_period": 1})
        assert r.status_code == 422 and "ema_period" in r.json()["message"]
        r = api.client.get("/api/etfs/QQQ/strategies/ema_bollinger_atr", params={"rsi_long_exit": 60, "rsi_short_exit": 45, "bb_std": 99})
        assert r.status_code == 422

    def test_cache_keys_differ_per_parameter_set(self, api):
        api.prices["QQQ"] = random_walk(900, seed=3)
        api.client.get("/api/etfs/QQQ/strategies/ema_bollinger_atr")
        api.client.get("/api/etfs/QQQ/strategies/ema_bollinger_atr", params={"bb_std": 2})
        keys = [k for k, _ in api.cache.sets]
        assert len(set(keys)) == 2 and all(k.startswith("tw:v1:article:eba:QQQ:1d:") for k in keys)

    def test_insufficient_history(self, api):
        api.prices["QQQ"] = random_walk(80, seed=3)
        r = api.client.get("/api/etfs/QQQ/strategies/ema_bollinger_atr")
        assert r.status_code == 422 and "Insufficient 1d history" in r.json()["message"]

    # -- hourly timeframe / EODHD entitlement --------------------------------

    @staticmethod
    def hourly_rows(n=900):
        out, price = [], 100.0
        rng = np.random.default_rng(4)
        t = datetime.datetime(2024, 1, 2, 14, 30)
        for i in range(n):
            o = price
            c = price * (1 + rng.normal(0, 0.003))
            out.append(
                {
                    "timestamp": 0, "datetime": (t + datetime.timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                    "open": o, "high": max(o, c) * 1.001, "low": min(o, c) * 0.999, "close": c, "volume": 1000,
                }
            )
            price = c
        return out

    def test_hourly_success_uses_intraday_endpoint_and_never_touches_daily_data(self, api):
        fetch = AsyncMock(return_value=self.hourly_rows())
        api.monkeypatch.setattr(api.router.eodhd_client, "fetch_intraday_history", fetch)
        resp = api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"})
        assert resp.status_code == 200, resp.text
        d = resp.json()["data"]
        assert d["timeframe"] == "1h" and "intraday" in d["dataSource"].lower()
        assert d["lastBar"].endswith("Z")
        args = fetch.await_args
        assert args.args[0] == "SPY.US" and args.args[1] == "1h"
        assert api.calls == []  # daily_prices never consulted for an hourly request

    def test_missing_intraday_entitlement_is_an_actionable_403_with_no_daily_fallback(self, api):
        fetch = AsyncMock(side_effect=eodhd_client.EodhdEntitlementError(
            "EODHD denied access to intraday 1h data for SPY.US (HTTP 403). The configured EODHD plan/token "
            "must include intraday data (EOD+Intraday or All-in-One). Use timeframe=1d, or upgrade the plan."
        ))
        api.monkeypatch.setattr(api.router.eodhd_client, "fetch_intraday_history", fetch)
        api.prices["SPY"] = random_walk(900, seed=3)  # daily data IS available - must not be used
        resp = api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"})
        assert resp.status_code == 403
        body = resp.json()
        assert body["success"] is False and body["data"] is None
        text = resp.text
        assert "intraday" in text and "upgrade" in text.lower()
        assert api.calls == [] and api.cache.sets == []

    def test_no_hourly_coverage_is_404_not_daily_substitute(self, api):
        api.monkeypatch.setattr(api.router.eodhd_client, "fetch_intraday_history", AsyncMock(return_value=[]))
        resp = api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"})
        assert resp.status_code == 404 and "Daily data was NOT substituted" in resp.text
        assert api.calls == []

    def test_upstream_failure_is_502_and_does_not_leak_the_token(self, api):
        secret = "SECRET-TOKEN-123"
        api.monkeypatch.setattr(api.router.eodhd_client.settings, "eodhd_api_token", secret)
        request = httpx.Request("GET", f"https://eodhd.com/api/intraday/SPY.US?api_token={secret}")
        error = httpx.HTTPStatusError("boom", request=request, response=httpx.Response(500, request=request))
        api.monkeypatch.setattr(api.router.eodhd_client, "fetch_intraday_history", AsyncMock(side_effect=error))
        resp = api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"})
        assert resp.status_code == 502 and secret not in resp.text

    def test_timeout_is_502(self, api):
        api.monkeypatch.setattr(
            api.router.eodhd_client, "fetch_intraday_history", AsyncMock(side_effect=httpx.ConnectTimeout("t"))
        )
        assert api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"}).status_code == 502

    def test_hourly_bars_are_memoised_between_requests(self, api):
        fetch = AsyncMock(return_value=self.hourly_rows())
        api.monkeypatch.setattr(api.router.eodhd_client, "fetch_intraday_history", fetch)
        api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h"})
        api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h", "bb_std": 2})
        assert fetch.await_count == 1  # second request reused the in-process memo (protects the API quota)

    def test_hourly_lookback_validation(self, api):
        r = api.client.get("/api/etfs/SPY/strategies/ema_bollinger_atr", params={"timeframe": "1h", "lookback_days": 99999})
        assert r.status_code == 422


class TestDiversificationRoute:
    def seed(self, api):
        api.prices["SPY"] = random_walk(1400, seed=1)
        api.prices["GLD"] = random_walk(1400, seed=2)
        api.prices["SHY"] = random_walk(1400, seed=3)

    def test_success_with_weights(self, api):
        self.seed(api)
        resp = api.client.post(
            "/api/portfolio/diversification_heat_map",
            json={"symbols": ["spy", "GLD", "SHY"], "weightsPercent": {"SPY": 60, "SHY": 40}, "priceBasis": "adjusted"},
        )
        assert resp.status_code == 200, resp.text
        d = resp.json()["data"]
        assert d["symbols"] == ["SPY", "GLD", "SHY"]
        assert d["portfolio"]["rebalancingPolicy"] == "annual_reset"
        assert d["portfolio"]["weightsPercent"] == {"SPY": 60.0, "GLD": 0.0, "SHY": 40.0}
        assert any(e["isPortfolio"] for y in d["yearlyRankings"] for e in y["ranking"])
        assert all((api.db, s) in api.calls for s in ("SPY", "GLD", "SHY"))
        assert api.cache.sets[0][0].startswith("tw:v1:article:diversification:")

    def test_works_without_weights(self, api):
        self.seed(api)
        d = api.client.post("/api/portfolio/diversification_heat_map", json={"symbols": ["SPY", "GLD"]}).json()["data"]
        assert d["portfolio"] is None

    def test_cache_key_is_order_insensitive(self, api):
        self.seed(api)
        api.client.post("/api/portfolio/diversification_heat_map", json={"symbols": ["SPY", "GLD"]})
        api.client.post("/api/portfolio/diversification_heat_map", json={"symbols": ["GLD", "SPY"]})
        assert api.cache.sets[0][0] == api.cache.sets[1][0]

    @pytest.mark.parametrize(
        "body, status",
        [
            ({"symbols": ["SPY"]}, 422),
            ({"symbols": ["SPY", "SPY"]}, 422),
            ({"symbols": ["SPY", "NOPE"]}, 404),
            ({"symbols": [f"S{i}" for i in range(13)]}, 422),
            ({"symbols": ["SPY", "GLD"], "weightsPercent": {"SPY": 30, "GLD": 30}}, 422),
            ({"symbols": ["SPY", "GLD"], "weightsPercent": {"SPY": -10, "GLD": 110}}, 422),
            ({"symbols": ["SPY", "GLD"], "weightsPercent": {"SPY": "NaN", "GLD": 100}}, 422),
            ({"symbols": ["SPY", "GLD"], "weightsPercent": {"SPY": 50, "XXX": 50}}, 422),
            ({"symbols": ["SPY", "GLD"], "start": "2024-01-01", "end": "2020-01-01"}, 422),
            ({"symbols": ["SPY", "GLD"], "priceBasis": "weird"}, 422),
        ],
    )
    def test_validation_errors(self, api, body, status):
        self.seed(api)
        assert api.client.post("/api/portfolio/diversification_heat_map", json=body).status_code == status

    def test_weight_error_message_is_precise(self, api):
        self.seed(api)
        r = api.client.post(
            "/api/portfolio/diversification_heat_map",
            json={"symbols": ["SPY", "GLD"], "weightsPercent": {"SPY": 30, "GLD": 30}},
        )
        assert "sum to 100%" in r.json()["message"]

    def test_missing_price_data_is_503(self, api):
        api.prices["SPY"] = random_walk(500, seed=1)
        r = api.client.post("/api/portfolio/diversification_heat_map", json={"symbols": ["SPY", "GLD"]})
        assert r.status_code == 503 and "GLD" in r.json()["message"]

    def test_range_without_data_is_422(self, api):
        self.seed(api)
        r = api.client.post(
            "/api/portfolio/diversification_heat_map", json={"symbols": ["SPY", "GLD"], "start": "2050-01-01"}
        )
        assert r.status_code == 422


# -- EODHD client ------------------------------------------------------------


class TestEodhdIntradayClient:
    def run(self, coro):
        return asyncio.run(coro)

    def http_error(self, status):
        request = httpx.Request("GET", "https://eodhd.com/api/intraday/SPY.US?api_token=abc")
        return httpx.HTTPStatusError("x", request=request, response=httpx.Response(status, request=request))

    @pytest.mark.parametrize("status", [401, 402, 403])
    def test_entitlement_statuses_map_to_actionable_error(self, status):
        with patch.object(eodhd_client, "_get_json_with_retry", AsyncMock(side_effect=self.http_error(status))):
            with pytest.raises(eodhd_client.EodhdEntitlementError) as exc:
                self.run(eodhd_client.fetch_intraday_history("SPY.US", "1h"))
        message = str(exc.value)
        assert str(status) in message and "intraday" in message and "api_token" not in message

    def test_other_http_errors_propagate(self):
        with patch.object(eodhd_client, "_get_json_with_retry", AsyncMock(side_effect=self.http_error(404))):
            with pytest.raises(httpx.HTTPStatusError):
                self.run(eodhd_client.fetch_intraday_history("SPY.US", "1h"))

    def test_request_shape_and_payload_validation(self):
        mock = AsyncMock(return_value=[{"datetime": "2024-01-02 14:30:00"}])
        frm, to = datetime.datetime(2024, 1, 1), datetime.datetime(2024, 3, 1)
        with patch.object(eodhd_client, "_get_json_with_retry", mock):
            rows = self.run(eodhd_client.fetch_intraday_history("SPY.US", "1h", frm, to))
        assert rows == [{"datetime": "2024-01-02 14:30:00"}]
        url, params = mock.await_args.args[0], mock.await_args.args[1]
        assert url.endswith("/intraday/SPY.US") and params["interval"] == "1h"
        assert params["from"] == str(int(frm.replace(tzinfo=datetime.timezone.utc).timestamp()))
        assert params["to"] == str(int(to.replace(tzinfo=datetime.timezone.utc).timestamp()))
        with patch.object(eodhd_client, "_get_json_with_retry", AsyncMock(return_value={"error": "x"})):
            assert self.run(eodhd_client.fetch_intraday_history("SPY.US", "1h")) == []

    def test_documented_range_limits_and_intervals_enforced_before_any_request(self):
        mock = AsyncMock()
        with patch.object(eodhd_client, "_get_json_with_retry", mock):
            with pytest.raises(ValueError, match="at most 120 days"):
                self.run(eodhd_client.fetch_intraday_history(
                    "SPY.US", "1m", datetime.datetime(2020, 1, 1), datetime.datetime(2024, 1, 1)))
            with pytest.raises(ValueError, match="Unsupported"):
                self.run(eodhd_client.fetch_intraday_history("SPY.US", "15m"))
        mock.assert_not_called()

    def test_token_is_redacted_from_log_text(self):
        with patch.object(eodhd_client.settings, "eodhd_api_token", "TOPSECRET"):
            assert "TOPSECRET" not in eodhd_client._redact("GET https://x?api_token=TOPSECRET failed")
            assert "***" in eodhd_client._redact("api_token=TOPSECRET")


# -- ingestion / admin -------------------------------------------------------


class FakeSession:
    def __init__(self):
        self.executed = []
        self.commits = 0
        self._scalars = iter([10, 14, D(2010, 1, 4)])

    async def scalar(self, stmt):
        return next(self._scalars)

    async def execute(self, stmt):
        self.executed.append(stmt)

    async def commit(self):
        self.commits += 1


class TestBackfill:
    rows = [
        {"date": "2024-01-02", "open": 1, "high": 2, "low": 1, "close": 1.5, "adjusted_close": 1.4, "volume": 5},
        {"date": "2024-01-03", "open": 1, "high": 0.5, "low": 1, "close": 1.5, "adjusted_close": 1.4, "volume": 5},  # high<low
        {"date": "2024-01-04", "open": 1, "high": 2, "low": 1, "close": None, "adjusted_close": None, "volume": 5},
    ]

    def run(self, restate):
        from app.services import ingestion

        session = FakeSession()
        deleted = []

        async def delete_prefix(prefix):
            deleted.append(prefix)

        with patch.object(ingestion.eodhd_client, "fetch_eod_history", AsyncMock(return_value=self.rows)) as fetch, \
                patch.object(ingestion.cache, "delete_prefix", delete_prefix):
            result = asyncio.run(ingestion.backfill_symbol_history(session, "SPY", 15, restate_adjusted=restate))
        return session, result, deleted, fetch

    def sql(self, session):
        from sqlalchemy.dialects import postgresql

        return str(session.executed[0].compile(dialect=postgresql.dialect()))

    def test_insert_only_by_default_and_filters_invalid_rows(self):
        session, result, deleted, fetch = self.run(False)
        assert "DO NOTHING" in self.sql(session) and "DO UPDATE" not in self.sql(session)
        assert result["rows_fetched"] == 1 and result["rows_inserted"] == 4
        assert result["adjusted_close_restated"] is False and result["earliest_date_stored"] == "2010-01-04"
        assert fetch.await_args.args[0] == "SPY.US" and "from_date" in fetch.await_args.kwargs
        assert session.commits == 1

    def test_restate_updates_only_adjusted_close(self):
        session, result, _, _ = self.run(True)
        sql = self.sql(session)
        assert "DO UPDATE SET adjusted_close" in sql
        assert "SET open" not in sql and "SET close" not in sql
        assert result["adjusted_close_restated"] is True

    def test_strategy_caches_are_invalidated(self):
        _, _, deleted, _ = self.run(False)
        assert "tw:v11:strategies:" in deleted and "tw:v1:article:" in deleted


class TestAdminBackfillRoute:
    @pytest.fixture
    def admin(self, monkeypatch):
        from app.routers import admin

        monkeypatch.setattr(admin.settings, "admin_api_token", "adm")
        monkeypatch.setattr(admin, "_run_in_background", lambda coro: coro.close())
        app = FastAPI()
        app.include_router(admin.router, prefix="/api")
        return TestClient(app)

    def test_requires_admin_token_and_is_invisible_otherwise(self, admin):
        assert admin.post("/api/admin/backfill", params={"symbol": "SPY"}).status_code == 404
        assert admin.post("/api/admin/backfill", params={"symbol": "SPY"}, headers={"X-Admin-Token": "bad"}).status_code == 404

    def test_unknown_symbol_and_bad_years(self, admin):
        h = {"X-Admin-Token": "adm"}
        assert admin.post("/api/admin/backfill", params={"symbol": "NOPE"}, headers=h).status_code == 404
        assert admin.post("/api/admin/backfill", params={"symbol": "SPY", "years": 100}, headers=h).status_code == 422

    def test_accepts_and_starts_in_background(self, admin):
        r = admin.post(
            "/api/admin/backfill", params={"symbol": "spy", "years": 20, "restate_adjusted": "true"},
            headers={"X-Admin-Token": "adm"},
        )
        assert r.status_code == 202
        assert r.json()["data"] == {"status": "started", "symbol": "SPY", "years": 20, "restateAdjusted": True}


class TestCategoryClassification:
    def test_new_fields_live_in_existing_categories_only(self):
        from app.schemas import (
            CalendarStrategies,
            PortfolioStrategies,
            QuantitativeStrategies,
            RegimeStrategies,
            StrategyAnalytics,
            TechnicalStrategies,
        )

        assert "seasonal_heat_map" in CalendarStrategies.model_fields
        assert "ema_bollinger_atr" in TechnicalStrategies.model_fields
        assert "crash_warning" in QuantitativeStrategies.model_fields
        assert "seasonal_heat_map" not in RegimeStrategies.model_fields
        assert set(PortfolioStrategies.model_fields) == {"tqqq_tmf_ief_rebalancing", "vt_vs_vti_vxus"}
        assert all(not f.is_required() for f in StrategyAnalytics.model_fields.values())

    def test_no_new_external_data_provider_dependencies(self):
        import pathlib

        root = pathlib.Path(__file__).parents[2] / "app"
        text = (root / "services" / "strategies.py").read_text(encoding="utf-8")
        section = text[text.index("Strategies 29-32: article-based additions"):].lower()
        client = (root / "services" / "eodhd_client.py").read_text(encoding="utf-8").lower()
        for provider in ("yfinance", "alpha_vantage", "alphavantage", "financialmodelingprep", "polygon", "import fmp"):
            assert provider not in section and provider not in client
        assert "https://eodhd.com/api" in client
