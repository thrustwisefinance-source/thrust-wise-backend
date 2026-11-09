"""
Unit tests for the isolated Set 3 architecture (app.strategies.set_3_strategies,
app.routers.strategy_set3, app.schemas.strategy_set3).

No database / Redis needed: the router is mounted on a throwaway FastAPI app
with load_prices, the newest-date query and the cache replaced by fakes.
"""

import datetime
import os
import types

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("EODHD_API_TOKEN", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.database import get_db  # noqa: E402
from app.exceptions import register_exception_handlers  # noqa: E402
from app.routers import strategy_set3  # noqa: E402
from app.schemas.strategy_set3 import (  # noqa: E402
    QuantitativeSet3Strategies,
    RandomizedSimulationParams,
)
from app.strategies.set_3_strategies import CATEGORY_REGISTRIES  # noqa: E402
from app.strategies.set_3_strategies import common  # noqa: E402
from app.strategies.set_3_strategies.quantitative import (  # noqa: E402
    buy_and_hold,
    compute_randomized_investment_simulation,
    simulate_random_portfolios,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_rows(n=300, start=datetime.date(2023, 1, 2), drift=0.0004, vol=0.01, seed=7, base=100.0):
    rng = np.random.default_rng(seed)
    price = base * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    dates = pd.bdate_range(start, periods=n)
    return [
        types.SimpleNamespace(date=d.date(), close=float(p), adjusted_close=float(p), volume=1000)
        for d, p in zip(dates, price)
    ]


def make_frame(**kw):
    return common.prices_to_frame(make_rows(**kw))


def ctx(frame=None, params=None, symbol="VOO", extra=None):
    return common.Set3Context(
        symbol=symbol,
        prices=frame if frame is not None else make_frame(),
        extra_prices=extra or {},
        params=params,
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_params_defaults_valid():
    p = RandomizedSimulationParams()
    assert p.buy_probability + p.sell_probability == 1
    assert p.benchmark_symbol == "SPY"


@pytest.mark.parametrize(
    "kw",
    [
        {"buy_probability": 0.7, "sell_probability": 0.5},  # sums above 1
        {"buy_probability": 0.0, "sell_probability": 0.0},  # sums to 0
        {"buy_probability": -0.1},
        {"max_purchase_allocation_percent": 0},
        {"max_sell_percent": 101},
        {"iterations": 0},
        {"iterations": 5000},
        {"initial_capital": 0},
        {"seed": -1},
        {"commission_percent": -1},
        {"start_date": datetime.date(2024, 1, 1), "end_date": datetime.date(2023, 1, 1)},
        {"benchmark_symbol": "BAD SYMBOL!"},
    ],
)
def test_params_rejected(kw):
    with pytest.raises(ValidationError):
        RandomizedSimulationParams(**kw)


def test_params_hold_probability_allowed_and_symbol_uppercased():
    p = RandomizedSimulationParams(buy_probability=0.3, sell_probability=0.3, benchmark_symbol="voo")
    assert p.benchmark_symbol == "VOO"


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------

CORE = dict(
    initial_capital=10_000.0,
    buy_probability=0.5,
    sell_probability=0.5,
    max_purchase_fraction=1.0,
    max_sell_fraction=1.0,
    iterations=200,
    seed=123,
    commission_rate=0.001,
    commission_flat=1.0,
)


def test_never_negative_cash_or_holdings_aggressive_settings():
    prices = make_frame(n=400, vol=0.03)["price"].to_numpy()
    out = simulate_random_portfolios(prices, **CORE)
    assert out["min_cash"].min() >= 0
    assert out["min_shares"].min() >= 0
    assert (out["equity"] >= 0).all()


def test_never_negative_with_flat_fee_larger_than_trades():
    prices = make_frame(n=120, base=2.0)["price"].to_numpy()  # tiny prices
    kw = {**CORE, "commission_flat": 50.0, "initial_capital": 500.0, "iterations": 100}
    out = simulate_random_portfolios(prices, **kw)
    assert out["min_cash"].min() >= 0 and out["min_shares"].min() >= 0


def test_unaffordable_price_means_no_trades_and_cash_untouched():
    prices = np.full(50, 1_000_000.0)
    out = simulate_random_portfolios(prices, **{**CORE, "initial_capital": 100.0, "iterations": 10})
    assert out["shares"].sum() == 0 and np.allclose(out["cash"], 100.0)


def test_reproducible_with_same_seed_and_different_for_other_seed():
    prices = make_frame()["price"].to_numpy()
    a = simulate_random_portfolios(prices, **CORE)
    b = simulate_random_portfolios(prices, **CORE)
    c = simulate_random_portfolios(prices, **{**CORE, "seed": 124})
    assert np.array_equal(a["equity"], b["equity"])
    assert not np.array_equal(a["equity"], c["equity"])


def test_iterations_use_independent_streams_and_are_stable_across_counts():
    prices = make_frame()["price"].to_numpy()
    many = simulate_random_portfolios(prices, **{**CORE, "iterations": 50})
    few = simulate_random_portfolios(prices, **{**CORE, "iterations": 10})
    # iteration k is identical regardless of how many iterations were requested
    assert np.array_equal(many["equity"][:10], few["equity"])
    # and iterations differ from each other
    assert len({tuple(r[-5:]) for r in many["equity"]}) > 40


def test_no_look_ahead_future_prices_do_not_change_past_equity():
    prices = make_frame(n=200)["price"].to_numpy()
    altered = prices.copy()
    altered[150:] *= 3.0
    a = simulate_random_portfolios(prices, **CORE)["equity"]
    b = simulate_random_portfolios(altered, **CORE)["equity"]
    assert np.array_equal(a[:, :150], b[:, :150])


def test_buy_probability_one_only_buys_sell_probability_zero():
    prices = make_frame()["price"].to_numpy()
    out = simulate_random_portfolios(
        prices, **{**CORE, "buy_probability": 1.0, "sell_probability": 0.0, "iterations": 20}
    )
    assert out["sells"].sum() == 0 and out["buys"].min() > 0


def test_buy_and_hold_uses_same_cost_model():
    prices = np.array([100.0, 110.0, 121.0])
    bh = buy_and_hold(prices, initial_capital=10_000.0, commission_rate=0.01, commission_flat=5.0)
    # shares = floor((10000-5)/(100*1.01)) = 98 ; fee = 98*100*.01 + 5 = 103
    assert bh["shares"] == 98 and bh["commission"] == pytest.approx(103.0)
    assert bh["equity"][0] == pytest.approx(10_000.0 - 103.0)  # only costs lost on day 1
    assert bh["equity"][-1] == pytest.approx(10_000.0 - 98 * 100 - 103 + 98 * 121)


def test_zero_cost_buy_and_hold_matches_price_ratio_for_divisible_prices():
    prices = np.array([100.0, 150.0])
    bh = buy_and_hold(prices, initial_capital=10_000.0, commission_rate=0.0, commission_flat=0.0)
    assert bh["equity"][-1] / 10_000.0 - 1 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Strategy entry point
# ---------------------------------------------------------------------------


def run(params=None, **kw):
    return compute_randomized_investment_simulation(ctx(params=params, **kw))


def test_result_shape_and_metrics():
    params = RandomizedSimulationParams(iterations=60, seed=5)
    bench = make_frame(seed=99, base=400.0)
    out = run(params, extra={"SPY": bench})
    assert out["name"].startswith("Randomized Investment Simulation")
    agg = out["aggregate"]
    for k in (
        "average_return_percent", "median_return_percent", "best_return_percent",
        "worst_return_percent", "std_dev_return_percent", "percent_outperforming_buy_and_hold",
        "average_max_drawdown_percent", "average_final_value",
    ):
        assert agg[k] is not None
    assert agg["worst_return_percent"] <= agg["median_return_percent"] <= agg["best_return_percent"]
    assert len(out["iteration_results"]) == 60
    assert out["iteration_results"][0]["iteration"] == 1
    assert out["buy_and_hold"]["symbol"] == "VOO" and out["benchmark"]["symbol"] == "SPY"
    assert out["parameters"]["initial_capital"] == 10_000.0
    assert out["parameters"]["reproducible"] is True and out["parameters"]["seed_used"] == 5
    assert out["chart_data"][0]["date"] == out["period"]["start_date"]
    assert out["chart_data"][-1]["date"] == out["period"]["end_date"]
    assert out["chart_data"][0]["p5"] <= out["chart_data"][0]["p95"]
    assert any("Antifragil" in n for n in out["notes"])
    # percent outperforming agrees with the iteration-level flags
    flags = [r["beat_buy_and_hold"] for r in out["iteration_results"]]
    assert agg["percent_outperforming_buy_and_hold"] == pytest.approx(100 * sum(flags) / 60, abs=0.01)
    # final values & final balances are internally consistent per iteration
    last_price = ctx().prices["price"].iloc[-1]
    r0 = out["iteration_results"][0]
    assert r0["final_value"] == pytest.approx(r0["final_cash"] + r0["final_shares"] * last_price, abs=0.02)


def test_same_period_and_capital_for_strategy_and_buy_and_hold_and_benchmark():
    params = RandomizedSimulationParams(iterations=10, seed=1, initial_capital=25_000)
    out = run(params, extra={"SPY": make_frame(seed=3, base=300.0)})
    assert out["buy_and_hold"]["initial_capital"] == out["benchmark"]["initial_capital"] == 25_000
    assert out["aggregate"]["initial_capital"] == 25_000
    # both benchmark and B&H chart columns start at ~initial capital (costs only)
    first = out["chart_data"][0]
    assert abs(first["buy_and_hold"] - 25_000) < 25_000 * 0.02
    assert abs(first["benchmark"] - 25_000) < 25_000 * 0.02


def test_benchmark_equal_to_symbol_reuses_buy_and_hold():
    out = run(RandomizedSimulationParams(iterations=5, seed=1, benchmark_symbol="VOO"))
    assert out["benchmark"] == out["buy_and_hold"]


def test_missing_benchmark_is_omitted_with_warning():
    out = run(RandomizedSimulationParams(iterations=5, seed=1))
    assert out["benchmark"] is None
    assert out["aggregate"]["percent_outperforming_benchmark"] is None
    assert any("Benchmark SPY" in w for w in out["warnings"])


def test_benchmark_starting_late_is_omitted():
    late = make_frame(n=100, start=datetime.date(2023, 9, 1))
    out = run(RandomizedSimulationParams(iterations=5, seed=1), extra={"SPY": late})
    assert out["benchmark"] is None and any("starts after" in w for w in out["warnings"])


def test_date_window_is_applied_and_clamped_with_warning():
    frame = make_frame(n=400)
    start = frame.index[100].date()
    end = frame.index[250].date()
    out = run(RandomizedSimulationParams(iterations=5, seed=1, start_date=start, end_date=end), frame=frame)
    assert out["period"]["start_date"] == start.isoformat()
    assert out["period"]["end_date"] == end.isoformat()
    assert out["period"]["trading_days"] == 151
    out2 = run(RandomizedSimulationParams(iterations=5, seed=1, start_date=datetime.date(2000, 1, 1)), frame=frame)
    assert any("before the first available" in w for w in out2["warnings"])


def test_insufficient_history_raises_unavailable():
    with pytest.raises(common.Set3StrategyUnavailable):
        run(RandomizedSimulationParams(seed=1), frame=make_frame(n=10))


def test_unseeded_run_is_flagged_non_reproducible_and_reports_seed_to_replay():
    out = run(RandomizedSimulationParams(iterations=5))
    assert out["parameters"]["reproducible"] is False
    replay = run(RandomizedSimulationParams(iterations=5, seed=out["parameters"]["seed_used"]))
    assert replay["aggregate"] == out["aggregate"]


def test_output_is_json_safe():
    import json

    json.dumps(run(RandomizedSimulationParams(iterations=5, seed=2)), allow_nan=False)


def test_chart_downsampling_keeps_endpoints():
    out = run(RandomizedSimulationParams(iterations=3, seed=2, chart_max_points=25))
    assert len(out["chart_data"]) <= 25
    assert out["chart_data"][-1]["date"] == out["period"]["end_date"]


def test_cacheable_only_with_seed():
    spec = CATEGORY_REGISTRIES["quantitative"]["randomized_investment_simulation"]
    assert spec.cacheable(RandomizedSimulationParams(seed=1)) is True
    assert spec.cacheable(RandomizedSimulationParams()) is False


# ---------------------------------------------------------------------------
# Registry / architecture
# ---------------------------------------------------------------------------


def test_every_registered_strategy_has_a_schema_field():
    from app.schemas import strategy_set3 as s

    containers = {
        "technical": s.TechnicalSet3Strategies,
        "quantitative": s.QuantitativeSet3Strategies,
        "regime": s.RegimeSet3Strategies,
        "calendar": s.CalendarSet3Strategies,
        "portfolio": s.PortfolioSet3Strategies,
    }
    assert set(CATEGORY_REGISTRIES) == set(common.CATEGORIES) == set(containers)
    for category, registry in CATEGORY_REGISTRIES.items():
        for key, spec in registry.items():
            assert spec.key == key and spec.category == category
            assert key in containers[category].model_fields
    assert "randomized_investment_simulation" in QuantitativeSet3Strategies.model_fields


def test_cache_namespaces_are_independent_and_not_legacy():
    keys = {
        c: common.build_cache_key(c, "voo", data_version="2026-01-02", fingerprint="x")
        for c in common.CATEGORIES
    }
    assert keys["quantitative"] == "tw:set3:quantitative:VOO:2026-01-02:x"
    assert len(set(keys.values())) == 5
    for c, k in keys.items():
        assert k.startswith(f"tw:set3:{c}:VOO")
        assert not k.startswith("tw:v")
    # none of the legacy purge prefixes used by ingestion can match a Set 3 key
    for legacy in ("tw:v12:strategies:", "tw:v11:strategies:", "tw:v3:strategies:", "tw:v1:article:"):
        assert not any(k.startswith(legacy) for k in keys.values())


def test_fingerprint_depends_on_seed_and_params():
    a = common.params_fingerprint(RandomizedSimulationParams(seed=1))
    assert a == common.params_fingerprint(RandomizedSimulationParams(seed=1))
    assert a != common.params_fingerprint(RandomizedSimulationParams(seed=2))
    assert a != common.params_fingerprint(RandomizedSimulationParams(seed=1, iterations=99))


# ---------------------------------------------------------------------------
# Router (fake DB / cache)
# ---------------------------------------------------------------------------


class FakeCache:
    def __init__(self):
        self.store = {}
        self.sets = []

    async def get_json(self, key):
        return self.store.get(key)

    async def set_json(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append(key)


@pytest.fixture
def client(monkeypatch):
    rows = {"VOO": make_rows(n=300, seed=1), "SPY": make_rows(n=300, seed=2, base=400.0)}
    calls = {"load": 0}

    async def fake_load_prices(db, symbol, limit=None):
        calls["load"] += 1
        return rows.get(symbol, [])

    async def fake_latest(db, symbol):
        r = rows.get(symbol)
        return r[-1].date if r else None

    fake_cache = FakeCache()
    monkeypatch.setattr(strategy_set3, "load_prices", fake_load_prices)
    monkeypatch.setattr(strategy_set3, "_latest_price_date", fake_latest)
    monkeypatch.setattr(strategy_set3, "cache", fake_cache)

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(strategy_set3.router, prefix="/api")

    async def fake_db():
        yield None

    app.dependency_overrides[get_db] = fake_db
    c = TestClient(app, raise_server_exceptions=False)
    c.fake_cache, c.calls, c.rows = fake_cache, calls, rows
    return c


CATEGORIES = ["technical", "quantitative", "regime", "calendar", "portfolio"]


def test_all_five_endpoints_registered():
    paths = {r.path for r in strategy_set3.router.routes}
    for c in CATEGORIES:
        assert f"/etfs/{{symbol}}/strategies/{c}_set3" in paths
    assert len(paths) == 5


@pytest.mark.parametrize("category", ["technical", "regime", "calendar", "portfolio"])
def test_empty_categories_return_envelope_with_no_strategies(client, category):
    r = client.get(f"/api/etfs/VOO/strategies/{category}_set3")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["data"]["symbol"] == "VOO"
    assert body["data"]["category"] == category
    assert body["data"]["setName"] == "set3"
    assert body["data"]["strategies"] == {}
    assert "No Set 3" in body["message"]


def test_quantitative_endpoint_returns_only_set3_quantitative(client):
    r = client.get("/api/etfs/VOO/strategies/quantitative_set3?seed=11&iterations=25")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["category"] == "quantitative"
    assert list(data["strategies"]) == ["randomizedInvestmentSimulation"]
    sim = data["strategies"]["randomizedInvestmentSimulation"]
    assert sim["parameters"]["iterations"] == 25
    assert sim["aggregate"]["iterations"] == 25
    assert len(sim["iterationResults"]) == 25
    assert sim["benchmark"]["symbol"] == "SPY"
    assert "percentOutperformingBuyAndHold" in sim["aggregate"]
    assert data["registeredStrategies"][0]["key"] == "randomized_investment_simulation"


def test_seeded_requests_are_cached_unseeded_are_not(client):
    url = "/api/etfs/VOO/strategies/quantitative_set3?iterations=10"
    first = client.get(url + "&seed=3").json()
    n_loads = client.calls["load"]
    second = client.get(url + "&seed=3").json()
    assert first == second
    assert client.calls["load"] == n_loads  # served from cache, no DB price load
    assert len(client.fake_cache.sets) == 1
    assert client.fake_cache.sets[0].startswith("tw:set3:quantitative:VOO:")
    # different seed -> different key
    client.get(url + "&seed=4")
    assert len(set(client.fake_cache.sets)) == 2
    # no seed -> never cached, fresh draw each call
    before = list(client.fake_cache.sets)
    a = client.get(url).json()["data"]["strategies"]["randomizedInvestmentSimulation"]
    b = client.get(url).json()["data"]["strategies"]["randomizedInvestmentSimulation"]
    assert client.fake_cache.sets == before
    assert a["parameters"]["seedSupplied"] is False
    assert a["parameters"]["seedUsed"] != b["parameters"]["seedUsed"]


def test_cache_key_changes_when_new_price_data_arrives(client):
    url = "/api/etfs/VOO/strategies/quantitative_set3?iterations=5&seed=1"
    client.get(url)
    extra = types.SimpleNamespace(
        date=client.rows["VOO"][-1].date + datetime.timedelta(days=1),
        close=100.0, adjusted_close=100.0, volume=1,
    )
    client.rows["VOO"].append(extra)
    client.get(url)
    assert len(set(client.fake_cache.sets)) == 2


def test_reproducible_across_requests(client):
    url = "/api/etfs/VOO/strategies/quantitative_set3?iterations=15&seed=99"
    a = client.get(url).json()["data"]["strategies"]["randomizedInvestmentSimulation"]
    client.fake_cache.store.clear()
    b = client.get(url).json()["data"]["strategies"]["randomizedInvestmentSimulation"]
    assert a == b


def test_validation_errors_use_the_error_envelope(client):
    r = client.get("/api/etfs/VOO/strategies/quantitative_set3?buy_probability=0.8&sell_probability=0.8")
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False and body["data"] is None and "must not exceed 1" in body["message"]
    r = client.get("/api/etfs/VOO/strategies/quantitative_set3?iterations=100000")
    assert r.status_code == 422 and r.json()["success"] is False


def test_unknown_symbol_404_and_unknown_strategy_422(client):
    assert client.get("/api/etfs/NOPE/strategies/quantitative_set3").status_code == 404
    r = client.get("/api/etfs/VOO/strategies/quantitative_set3?strategies=bogus")
    assert r.status_code == 422 and "bogus" in r.json()["message"]


def test_no_price_data_is_503_like_existing_api(client):
    client.rows["VOO"] = []
    assert client.get("/api/etfs/VOO/strategies/quantitative_set3?seed=1").status_code == 503


def test_insufficient_history_returns_null_strategy_with_reason(client):
    client.rows["VOO"] = make_rows(n=8)
    body = client.get("/api/etfs/VOO/strategies/quantitative_set3?seed=1").json()
    assert body["success"] is True
    assert body["data"]["strategies"]["randomizedInvestmentSimulation"] is None
    assert "randomized_investment_simulation" in body["data"]["unavailable"]
    assert "trading day" in body["message"]


def test_strategy_failure_is_isolated_and_not_cached(client, monkeypatch):
    spec = CATEGORY_REGISTRIES["quantitative"]["randomized_investment_simulation"]
    boom = common.Set3Strategy(**{**spec.__dict__, "compute": lambda c: 1 / 0})
    monkeypatch.setitem(CATEGORY_REGISTRIES["quantitative"], spec.key, boom)
    r = client.get("/api/etfs/VOO/strategies/quantitative_set3?seed=1")
    assert r.status_code == 200
    assert r.json()["data"]["strategies"]["randomizedInvestmentSimulation"] is None
    assert client.fake_cache.sets == []


def test_existing_strategies_route_is_untouched_by_set3_router():
    """Set 3 must not define (or shadow) GET /etfs/{symbol}/strategies."""
    assert "/etfs/{symbol}/strategies" not in {r.path for r in strategy_set3.router.routes}
