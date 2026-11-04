"""
Tests for Volatility Harvesting (app.services.volatility_harvesting) and
POST /api/portfolio/volatility_harvesting. Deterministic synthetic prices; no
network, no database.
"""

import datetime
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db
from app.schemas import VolatilityHarvestingStrategy
from app.services import strategies as vh

D0 = datetime.date(2022, 1, 3)


def weekdays(n, start=D0):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def bars(values, dates=None, start=D0):
    dates = dates or weekdays(len(values), start)
    return [
        SimpleNamespace(symbol="T", date=d, open=v, high=v, low=v, close=v, adjusted_close=v, volume=1)
        for d, v in zip(dates, values)
    ]


def zigzag(n, amp=0.02, base=100.0, phase=0):
    """Mean-reverting oscillation: ideal for rebalancing."""
    return [base * (1 + amp * (1 if (i + phase) % 20 < 10 else -1)) for i in range(n)]


def two_assets(n=300):
    a = bars(zigzag(n))
    b = bars(zigzag(n, phase=10))
    return {"AAA": a, "BBB": b}


W = {"AAA": 50.0, "BBB": 50.0}


def run(prices=None, weights=W, **kw):
    return vh.compute_volatility_harvesting(prices or two_assets(), weights, **kw)


class TestValidation:
    def v(self, **over):
        args = dict(
            symbols=["AAA", "BBB"], weights_percent=W, rebalance_frequency="quarterly", initial_value=1000.0,
            transaction_cost_percent=0.0, risk_free_rate_percent=0.0, start=None, end=None,
        )
        args.update(over)
        return vh.validate_request(**args)

    def test_valid_request(self):
        symbols, weights = self.v(symbols=["aaa", " bbb "], weights_percent={"aaa": 60, "BBB": 40})
        assert symbols == ["AAA", "BBB"] and weights == {"AAA": 60.0, "BBB": 40.0}

    @pytest.mark.parametrize(
        "over,msg",
        [
            (dict(symbols=[]), "empty"),
            (dict(symbols=["AAA"], weights_percent={"AAA": 100}), "between 2 and 12"),
            (dict(symbols=["AAA", "aaa"]), "Duplicate symbol"),
            (dict(symbols=["AAA", ""]), "non-empty"),
            (dict(weights_percent=None), "weights_percent is required"),
            (dict(weights_percent={"AAA": 50, "BBB": 40}), "sum to 100"),
            (dict(weights_percent={"AAA": 100.0, "BBB": 0.0}), "greater than 0"),
            (dict(weights_percent={"AAA": 60, "BBB": -10, }), "negative"),
            (dict(weights_percent={"AAA": 100}), "Missing target weight"),
            (dict(weights_percent={"AAA": 50, "BBB": 25, "CCC": 25}), "not in the requested"),
            (dict(weights_percent={"AAA": float("nan"), "BBB": 50}), "finite"),
            (dict(weights_percent={"AAA": "50", "BBB": 50}), "finite"),
            (dict(rebalance_frequency="weekly"), "rebalance_frequency"),
            (dict(initial_value=0), "initial_value"),
            (dict(initial_value=float("inf")), "initial_value"),
            (dict(transaction_cost_percent=-1), "transaction_cost_percent"),
            (dict(transaction_cost_percent=50), "transaction_cost_percent"),
            (dict(risk_free_rate_percent=-1), "risk_free_rate_percent"),
            (dict(start=datetime.date(2024, 5, 1), end=datetime.date(2024, 1, 1)), "start must not be after end"),
            (dict(cost_sensitivity=[]), "non-empty"),
            (dict(cost_sensitivity=[0.1, 99]), "between 0 and"),
        ],
    )
    def test_invalid_requests_raise(self, over, msg):
        with pytest.raises(ValueError, match=msg):
            self.v(**over)

    def test_insufficient_history_and_bad_basis(self):
        with pytest.raises(ValueError, match="Insufficient aligned price history"):
            run(two_assets(30))
        with pytest.raises(ValueError, match="price_basis"):
            run(price_basis="nope")
        with pytest.raises(ValueError, match="No usable"):
            run({"AAA": bars(zigzag(300)), "BBB": []})

    def test_window_filter_applied(self):
        res = run(start=datetime.date(2022, 6, 1), end=datetime.date(2022, 12, 31))
        assert res["analysis_period"]["start"] >= "2022-06-01" and res["analysis_period"]["end"] <= "2022-12-31"
        with pytest.raises(ValueError, match="Insufficient"):
            run(start=datetime.date(2022, 12, 1), end=datetime.date(2022, 12, 20))


class TestSimulationMechanics:
    def setup_method(self):
        self.idx = pd.DatetimeIndex(weekdays(120))

    def paths(self, a, b, freq="monthly", cost=0.0, w=(0.5, 0.5)):
        prices = np.column_stack([a, b]).astype(float)
        target = np.array(w, dtype=float)
        bh = vh._simulate(prices, self.idx, target, freq, 1000.0, cost, rebalance=False)
        rb = vh._simulate(prices, self.idx, target, freq, 1000.0, cost, rebalance=True)
        return prices, bh, rb

    def test_buy_and_hold_weights_drift_with_prices(self):
        a = np.linspace(100, 200, 120)  # doubles
        b = np.full(120, 100.0)
        _, bh, _ = self.paths(a, b)
        assert bh["weights"][0].tolist() == [0.5, 0.5]
        assert bh["weights"][-1][0] == pytest.approx(2 / 3, abs=1e-9)  # 200/(200+100)
        assert bh["values"][-1] == pytest.approx(1000 * (0.5 * 2 + 0.5 * 1))
        assert bh["events"] == []

    def test_rebalanced_weights_return_to_target_on_rebalance_dates(self):
        a = np.linspace(100, 200, 120)
        b = np.full(120, 100.0)
        _, _, rb = self.paths(a, b)
        assert rb["events"], "monthly rebalancing should have fired"
        dates = {e["date"]: i for i, e in enumerate(rb["events"])}
        for t, ts in enumerate(self.idx):
            if ts.date().isoformat() in dates:
                assert rb["weights"][t].tolist() == pytest.approx([0.5, 0.5])
        # between rebalances weights drift away from target
        assert not np.allclose(rb["weights"][-1], [0.5, 0.5])

    def test_rebalancing_happens_on_first_trading_day_of_each_new_month(self):
        _, _, rb = self.paths(np.linspace(100, 150, 120), np.full(120, 100.0))
        months = {(e["date"][:7]) for e in rb["events"]}
        assert all(
            (pd.Timestamp(e["date"]) - pd.Timedelta(days=1)).month != pd.Timestamp(e["date"]).month
            or pd.Timestamp(e["date"]).weekday() == 0
            for e in rb["events"]
        )
        first_idx = self.idx[0].strftime("%Y-%m")
        assert first_idx not in months  # initial allocation is not a rebalance event
        assert self.idx[-1].date().isoformat() not in {e["date"] for e in rb["events"]}

    def test_frequencies_produce_fewer_events_when_less_frequent(self):
        a, b = np.linspace(100, 150, 120), np.full(120, 100.0)
        counts = {f: len(self.paths(a, b, f)[2]["events"]) for f in vh.REBALANCE_FREQUENCIES}
        assert counts["monthly"] > counts["quarterly"] >= counts["semiannual"] >= counts["annual"]

    def test_transaction_cost_reduces_value_exactly_by_cost(self):
        a, b = np.linspace(100, 150, 120), np.full(120, 100.0)
        _, _, free = self.paths(a, b, cost=0.0)
        _, _, paid = self.paths(a, b, cost=0.01)
        assert paid["total_cost"] > 0 and paid["values"][-1] < free["values"][-1]
        first = paid["events"][0]
        assert first["transaction_cost"] == pytest.approx(0.01 * first["traded_notional"])

    def test_constant_prices_rebalanced_equals_buy_and_hold(self):
        _, bh, rb = self.paths(np.full(120, 50.0), np.full(120, 80.0))
        assert np.allclose(bh["values"], rb["values"])

    def test_no_lookahead_prefix_run_matches_full_run(self):
        rng = np.random.default_rng(5)
        a = 100 * np.cumprod(1 + rng.normal(0, 0.01, 120))
        b = 100 * np.cumprod(1 + rng.normal(0, 0.012, 120))
        prices = np.column_stack([a, b])
        target = np.array([0.6, 0.4])
        full = vh._simulate(prices, self.idx, target, "monthly", 1000.0, 0.001, True)
        k = 70
        prefix = vh._simulate(prices[:k], self.idx[:k], target, "monthly", 1000.0, 0.001, True)
        # every value before the prefix's last bar is identical: later prices cannot influence earlier results
        assert np.allclose(full["values"][: k - 1], prefix["values"][: k - 1])
        assert np.allclose(full["weights"][: k - 1], prefix["weights"][: k - 1])
        assert [e["date"] for e in full["events"] if e["date"] < self.idx[k - 1].date().isoformat()] == [
            e["date"] for e in prefix["events"] if e["date"] < self.idx[k - 1].date().isoformat()
        ]


class TestMetrics:
    def idx(self, n):
        return pd.DatetimeIndex(weekdays(n))

    def test_cumulative_return_cagr_vol_sharpe_drawdown_against_manual(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0004, 0.01, 520)
        values = 1000 * np.cumprod(np.r_[1.0, 1 + r])
        idx = self.idx(len(values))
        m = vh._metrics(values, idx, 2.0)
        assert m["cumulative_return_percent"] == pytest.approx((values[-1] / values[0] - 1) * 100, abs=1e-3)
        years = (idx[-1] - idx[0]).days / 365.25
        assert m["cagr_percent"] == pytest.approx(((values[-1] / values[0]) ** (1 / years) - 1) * 100, abs=1e-3)
        dr = values[1:] / values[:-1] - 1
        vol = dr.std(ddof=1) * math.sqrt(252)
        assert m["annualized_volatility_percent"] == pytest.approx(vol * 100, abs=1e-3)
        assert m["sharpe_ratio"] == pytest.approx((dr.mean() * 252 - 0.02) / vol, abs=1e-3)
        dd = (values / np.maximum.accumulate(values) - 1).min()
        assert m["max_drawdown_percent"] == pytest.approx(dd * 100, abs=1e-3)
        assert m["max_drawdown_peak_date"] < m["max_drawdown_trough_date"]

    def test_zero_volatility_gives_null_sharpe_and_vol(self):
        values = np.full(300, 1000.0)
        m = vh._metrics(values, self.idx(300), 1.0)
        assert m["annualized_volatility_percent"] is None and m["sharpe_ratio"] is None
        assert m["max_drawdown_percent"] == 0.0 and m["cumulative_return_percent"] == 0.0

    def test_cagr_not_reported_for_windows_shorter_than_a_year(self):
        values = 1000 * np.cumprod(np.r_[1.0, np.full(100, 1.001)])
        assert vh._metrics(values, self.idx(101), 0.0)["cagr_percent"] is None

    def test_max_drawdown_known_value(self):
        values = np.array([100, 120, 90, 110, 80, 130], dtype=float)
        m = vh._metrics(values, self.idx(6), 0.0)
        assert m["max_drawdown_percent"] == pytest.approx((80 / 120 - 1) * 100, abs=1e-3)
        assert m["max_drawdown_peak_date"] == self.idx(6)[1].date().isoformat()
        assert m["max_drawdown_trough_date"] == self.idx(6)[4].date().isoformat()


class TestComputeVolatilityHarvesting:
    def test_rebalancing_beats_buy_and_hold_on_mean_reverting_pair(self):
        res = run(rebalance_frequency="monthly")
        assert res["comparison"]["cumulative_return_difference_pp"] > 0
        assert res["comparison"]["return_difference_direction"] == "Rebalanced ahead"
        assert res["rebalancing"]["events"] == res["rebalanced"]["rebalance_events"] > 0
        assert res["buy_and_hold"]["rebalance_events"] == 0
        assert res["strategy_id"] == "volatility_harvesting" and res["category"] == "Quantitative Strategies"
        assert "not isolated" in res["comparison"]["attribution_note"]
        assert "guaranteed" in res["interpretation"].lower() or "not evidence" in res["interpretation"].lower()

    def test_missing_dates_are_aligned_not_filled(self):
        a = bars(zigzag(300))
        dates_b = [b.date for i, b in enumerate(bars(zigzag(300, phase=10))) if i % 25 != 0]
        b = bars([p.close for i, p in enumerate(bars(zigzag(300, phase=10))) if i % 25 != 0], dates=dates_b)
        res = run({"AAA": a, "BBB": b})
        assert res["data_quality"]["misaligned_dates_excluded"] == 12
        assert res["analysis_period"]["observations"] == 288
        assert any("aligned on common trading days" in w for w in res["data_quality_warnings"])

    def test_no_event_in_window_is_flagged_and_identical(self):
        res = run(rebalance_frequency="annual", start=D0, end=datetime.date(2022, 9, 1))
        assert res["rebalancing"]["events"] == 0
        assert res["comparison"]["cumulative_return_difference_pp"] == 0
        assert "identical to buy-and-hold" in res["interpretation"]

    def test_optional_frequency_comparison_and_cost_sensitivity(self):
        res = run(
            rebalance_frequency="monthly", include_frequency_comparison=True,
            transaction_cost_sensitivity_percent=[0.0, 0.5, 2.0],
        )
        assert [r["frequency"] for r in res["frequency_comparison"]] == list(vh.REBALANCE_FREQUENCIES)
        sens = res["transaction_cost_sensitivity"]
        values = [r["final_value"] for r in sens]
        assert values == sorted(values, reverse=True)  # more cost, lower final value
        assert sens[0]["cumulative_return_difference_pp"] > sens[-1]["cumulative_return_difference_pp"]
        plain = run(rebalance_frequency="monthly")
        assert plain["frequency_comparison"] is None and plain["transaction_cost_sensitivity"] is None

    def test_output_is_json_safe_and_matches_schema(self):
        res = run(rebalance_frequency="monthly", include_frequency_comparison=True,
                  transaction_cost_sensitivity_percent=[0.1])
        import json

        json.dumps(res, allow_nan=False)  # raises on NaN / Infinity
        model = VolatilityHarvestingStrategy.model_validate(res)
        dumped = model.model_dump(by_alias=True)
        assert dumped["buyAndHold"]["rebalanceEvents"] == 0
        assert set(dumped["correlationMatrix"]["symbols"]) == {"AAA", "BBB"}
        assert dumped["weightEvolution"][0]["rebalancedWeightsPercent"]["AAA"] == 50.0

    def test_weight_evolution_and_chart_are_downsampled(self):
        res = run(two_assets(1500), rebalance_frequency="monthly")
        assert len(res["weight_evolution"]) <= vh.MAX_WEIGHT_POINTS + 1
        assert len(res["chart_data"]) <= vh.MAX_CHART_POINTS + 1
        assert res["chart_data"][-1]["date"] == res["analysis_period"]["end"]

    def test_identical_assets_have_unit_correlation(self):
        a = bars(zigzag(300))
        res = run({"AAA": a, "BBB": list(a)})
        assert res["average_pairwise_correlation"] == pytest.approx(1.0, abs=1e-6)

    def test_flat_prices_do_not_produce_nan(self):
        flat = {"AAA": bars([100.0] * 300), "BBB": bars([50.0] * 300)}
        res = run(flat)
        assert res["rebalanced"]["sharpe_ratio"] is None and res["rebalanced"]["annualized_volatility_percent"] is None
        assert any("Sharpe" in w for w in res["data_quality_warnings"])

    def test_extreme_move_in_close_basis_triggers_warning(self):
        vals = zigzag(300)
        vals[150:] = [v / 2 for v in vals[150:]]  # un-adjusted 2:1 split
        res = run({"AAA": bars(vals), "BBB": bars(zigzag(300, phase=10))}, price_basis="close")
        assert any("possible unadjusted split" in w or "split" in w.lower() for w in res["data_quality_warnings"])


class TestRoute:
    @staticmethod
    def call(body, prices_by_symbol=None):
        from app.routers import strategy as router_module

        prices_by_symbol = prices_by_symbol or {"VOO": bars(zigzag(300)), "GLD": bars(zigzag(300, phase=10))}

        async def fake_load(db, symbol, limit=None):
            return prices_by_symbol.get(symbol, [])

        app = FastAPI()
        app.include_router(router_module.portfolio_router, prefix="/api")
        app.dependency_overrides[get_db] = lambda: None
        patches = [
            patch.object(router_module, "load_prices", fake_load),
            patch.object(router_module.cache, "get_json", AsyncMock(return_value=None)),
            patch.object(router_module.cache, "set_json", AsyncMock(return_value=None)),
        ]
        for p in patches:
            p.start()
        try:
            return TestClient(app).post("/api/portfolio/volatility_harvesting", json=body)
        finally:
            for p in patches:
                p.stop()

    BODY = {"symbols": ["VOO", "GLD"], "weightsPercent": {"VOO": 60, "GLD": 40}, "rebalanceFrequency": "monthly"}

    def test_success_camel_case_response(self):
        res = self.call(self.BODY)
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["strategyName"] == "Volatility Harvesting"
        assert data["portfolio"]["weightsPercent"] == {"VOO": 60.0, "GLD": 40.0}
        assert data["rebalanced"]["rebalanceEvents"] > 0
        assert data["assumptions"]["riskFreeRatePercent"] == pytest.approx(4.5)  # backend RISK_FREE_RATE default

    @pytest.mark.parametrize(
        "patch_body,fragment",
        [
            ({"weightsPercent": None}, "weights_percent is required"),
            ({"weightsPercent": {"VOO": 70, "GLD": 40}}, "sum to 100"),
            ({"symbols": ["VOO", "voo"]}, "Duplicate symbol"),
            ({"symbols": []}, "empty"),
            ({"start": "2024-05-01", "end": "2024-01-01"}, "start must not be after end"),
            ({"rebalanceFrequency": "daily"}, "rebalanceFrequency"),
            ({"transactionCostPercent": 99}, "transaction_cost_percent"),
        ],
    )
    def test_validation_errors_are_422_in_error_envelope(self, patch_body, fragment):
        res = self.call({**self.BODY, **patch_body})
        assert res.status_code == 422
        assert fragment in res.json()["detail"] or fragment in str(res.json())

    def test_unknown_symbol_404_and_short_history_422(self):
        assert self.call({**self.BODY, "symbols": ["VOO", "ZZZZ"], "weightsPercent": {"VOO": 50, "ZZZZ": 50}}).status_code == 404
        short = {"VOO": bars(zigzag(20)), "GLD": bars(zigzag(20))}
        res = self.call(self.BODY, short)
        assert res.status_code == 422 and "Insufficient" in res.json()["detail"]