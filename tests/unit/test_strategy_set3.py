"""
API tests for the additive Set 3 endpoints:

    GET  /api/etfs/{symbol}/strategies/quantitative_set3   (+ asymmetric_volatility_persistence,
                                                             fed_model_relative_valuation)
    POST /api/scanners/quantitative_set3                    (technical_fundamental_composite_scoring)

The real routers are mounted on a throwaway FastAPI app in the same order as
app/main.py. DB access, yfinance and Redis are replaced by fakes.
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

from app.database import get_db  # noqa: E402
from app.exceptions import register_exception_handlers  # noqa: E402
from app.routers import scanners_set3, strategies_set3  # noqa: E402
from app.routers import strategy_set3 as registry_router  # noqa: E402
from app.scanners.set_3_scanners.services import cache_service as scanner_cache  # noqa: E402
from app.scanners.set_3_scanners.services import stock_data_service as sds  # noqa: E402
from app.schemas.common import ApiResponse  # noqa: E402
from app.schemas.scanners_set3 import QuantitativeScannerSet3Response  # noqa: E402
from app.schemas.strategies_set3 import QuantitativeSet3ExtendedResponse  # noqa: E402
from app.strategies.set_3_strategies.quantitative_models import (  # noqa: E402
    asymmetric_volatility_persistence as avp,
)
from app.strategies.set_3_strategies.services import cache_service as strat_cache  # noqa: E402
from app.strategies.set_3_strategies.services import data_service  # noqa: E402
from app.strategies.set_3_strategies.services.data_service import (  # noqa: E402
    EarningsYieldInput,
    Set3DataUnavailable,
    TreasuryYieldInput,
)

AVP_ID = "asymmetric_volatility_persistence"
FED_ID = "fed_model_relative_valuation"
RAND_ID = "randomized_investment_simulation"
Q = "/api/etfs/VOO/strategies/quantitative_set3"
FAST = "avp_garch_enabled=false&iterations=5&seed=1&"  # keep the suite quick & deterministic


def make_rows(n=600, start=datetime.date(2023, 1, 2), seed=7, base=100.0):
    rng = np.random.default_rng(seed)
    price = base * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    dates = pd.bdate_range(start, periods=n)
    return [
        types.SimpleNamespace(date=d.date(), close=float(p), adjusted_close=float(p), volume=1000)
        for d, p in zip(dates, price, strict=True)
    ]


class FakeCache:
    def __init__(self, broken=False):
        self.store, self.sets, self.gets, self.broken = {}, [], [], broken

    async def get_json(self, key):
        self.gets.append(key)
        if self.broken:
            raise ConnectionError("redis down")
        return self.store.get(key)

    async def set_json(self, key, value, ttl=None):
        if self.broken:
            raise ConnectionError("redis down")
        self.store[key] = value
        self.sets.append(key)


def make_treasury(n=300):
    end = data_service.utc_today()
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    s = pd.Series(np.linspace(3.6, 4.2, n), index=idx)
    return TreasuryYieldInput(series=s, source="fake ^TNX", dropped_observations=0,
                              as_of=s.index[-1].date())


@pytest.fixture
def env(monkeypatch):
    rows = {"VOO": make_rows(seed=1), "SHY": make_rows(seed=2, base=80.0), "SPY": make_rows(seed=3, base=400.0)}
    cache = FakeCache()
    calls = {"avp": 0, "ey": 0, "ty": 0, "stock": 0}

    async def newest(db, symbol):
        r = rows.get(symbol)
        return r[-1].date if r else None

    async def load_frame(db, symbol):
        from app.strategies.set_3_strategies.common import prices_to_frame
        return prices_to_frame(rows.get(symbol, []))

    # ETF strategy side
    monkeypatch.setattr(strategies_set3, "_newest_price_date", newest)
    monkeypatch.setattr(data_service, "load_etf_price_frame", load_frame)
    monkeypatch.setattr(strat_cache, "cache", cache)
    real_compute = avp.compute_asymmetric_volatility_persistence

    def counting_compute(*a, **k):
        calls["avp"] += 1
        return real_compute(*a, **k)

    monkeypatch.setattr(avp, "compute_asymmetric_volatility_persistence", counting_compute)

    async def fake_ey(symbol, basis):
        calls["ey"] += 1
        return EarningsYieldInput(pe_ratio=25.0, pe_basis=basis, field_name="trailingPE",
                                  source="fake yfinance", as_of=data_service.utc_today())

    async def fake_ty(years):
        calls["ty"] += 1
        return make_treasury()

    monkeypatch.setattr(data_service, "fetch_etf_earnings_yield", fake_ey)
    monkeypatch.setattr(data_service, "fetch_treasury_yield_history", fake_ty)

    # existing registry router (randomized simulation) - same fakes the existing tests use
    async def fake_load_prices(db, symbol, limit=None):
        return rows.get(symbol, [])

    async def fake_latest(db, symbol):
        r = rows.get(symbol)
        return r[-1].date if r else None

    monkeypatch.setattr(registry_router, "load_prices", fake_load_prices)
    monkeypatch.setattr(registry_router, "_latest_price_date", fake_latest)
    monkeypatch.setattr(registry_router, "cache", cache)

    # scanner side
    monkeypatch.setattr(scanner_cache, "cache", cache)
    stocks = {}

    async def fake_stock(symbol, ma, rsi):
        calls["stock"] += 1
        behaviour = stocks[symbol]
        if isinstance(behaviour, Exception):
            raise behaviour
        close, pe, de = behaviour
        return sds.StockInputs(symbol, close, pe, de, None if de is None else de * 100)

    monkeypatch.setattr(sds, "fetch_stock_inputs", fake_stock)

    app = FastAPI()
    register_exception_handlers(app)
    # same relative order as app/main.py
    app.include_router(strategies_set3.router, prefix="/api")
    app.include_router(scanners_set3.router, prefix="/api")
    app.include_router(registry_router.router, prefix="/api")

    async def fake_db():
        yield None

    app.dependency_overrides[get_db] = fake_db
    client = TestClient(app, raise_server_exceptions=False)
    client.rows, client.cache, client.calls, client.stocks, client.app_ = rows, cache, calls, stocks, app
    client.monkeypatch = monkeypatch
    return client


def zigzag_up(n=120):
    base = 100 + 0.2 * np.arange(n)
    idx = pd.bdate_range(end=pd.Timestamp("2026-11-10"), periods=n)
    return pd.Series(base + np.where(np.arange(n) % 2 == 0, 1.5, -1.5), index=idx)


# ===========================================================================
# ETF strategies: quantitative_set3
# ===========================================================================


def test_quantitative_set3_returns_the_three_independent_strategies(env):
    r = env.get(Q + "?" + FAST)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["symbol"] == "VOO" and data["category"] == "quantitative" and data["setName"] == "set3"
    assert [s["key"] for s in data["registeredStrategies"]] == [RAND_ID, AVP_ID, FED_ID]
    st = data["strategies"]
    assert st["randomizedInvestmentSimulation"] is not None  # existing strategy preserved
    for key, sid in (("asymmetricVolatilityPersistence", AVP_ID), ("fedModelRelativeValuation", FED_ID)):
        s = st[key]
        for field in ("strategyId", "strategyName", "category", "description", "parameters",
                      "calculatedValues", "historicalSeries", "chartData", "interpretation",
                      "dataAvailability", "calculationStatus", "error"):
            assert field in s, (key, field)
        assert s["strategyId"] == sid and s["category"] == "quantitative"
        assert s["error"] is None


def test_response_validates_against_the_declared_schema(env):
    parsed = ApiResponse[QuantitativeSet3ExtendedResponse].model_validate(env.get(Q + "?" + FAST).json())
    s = parsed.data.strategies
    assert s.asymmetric_volatility_persistence.calculation_status == "ok"
    assert s.fed_model_relative_valuation.calculation_status == "ok"
    assert s.fed_model_relative_valuation.calculated_values.earnings_yield_pct == pytest.approx(4.0)
    assert s.asymmetric_volatility_persistence.parameters.vol_window == 21


def test_avp_content_and_parameters_are_wired_through(env):
    r = env.get(Q + "?strategies=" + AVP_ID + "&avp_garch_enabled=false&avp_vol_window=10"
                "&avp_acf_max_lag=7&avp_return_type=simple&avp_series_points=50")
    s = r.json()["data"]["strategies"]["asymmetricVolatilityPersistence"]
    assert s["parameters"]["volWindow"] == 10 and s["parameters"]["returnType"] == "simple"
    assert len(s["calculatedValues"]["autocorrelation"]["values"]) == 7
    assert s["calculatedValues"]["rollingVolatility"]["window"] == 10
    assert len(s["historicalSeries"]["dates"]) == 50
    assert s["calculatedValues"]["garch"]["status"] == "disabled"
    assert s["dataAvailability"]["status"] == "available"


def test_fed_content_and_parameters_are_wired_through(env):
    r = env.get(Q + "?strategies=" + FED_ID + "&fed_pe_basis=forward&fed_neutral_band_pct=3&fed_lookback_years=2")
    s = r.json()["data"]["strategies"]["fedModelRelativeValuation"]
    cv = s["calculatedValues"]
    assert s["parameters"] == {"lookbackYears": 2, "peBasis": "forward", "neutralBandPct": 3.0,
                               "chartMaxPoints": 400, "includeGranger": True}
    assert cv["peBasis"] == "forward" and cv["relativeValuation"] == "approximately_equal"  # |4.0-4.2| < 3
    assert s["historicalSeries"]["earningsYieldHistoryAvailable"] is False
    assert cv["grangerCausality"]["status"] == "unavailable"


def test_strategies_filter_runs_only_the_selected_ones(env):
    r = env.get(Q + "?strategies=" + FED_ID)
    st = r.json()["data"]["strategies"]
    assert st["fedModelRelativeValuation"] is not None
    assert st["asymmetricVolatilityPersistence"] is None and st["randomizedInvestmentSimulation"] is None
    assert env.calls["avp"] == 0


def test_unknown_strategy_key_is_rejected(env):
    r = env.get(Q + "?strategies=nope")
    assert r.status_code == 422 and r.json()["success"] is False
    assert "Unknown quantitative_set3 strategies: nope" in r.json()["message"]


@pytest.mark.parametrize("qs", [
    "avp_vol_window=1", "avp_acf_max_lag=0", "avp_acf_max_lag=500", "avp_garch_model=egarch",
    "avp_return_type=arith", "avp_garch_max_observations=10", "avp_shock_threshold_sigma=-1",
    "fed_lookback_years=0", "fed_lookback_years=99", "fed_pe_basis=peg",
    "fed_neutral_band_pct=-1", "avp_vol_window=abc",
])
def test_invalid_query_parameters_return_422_envelope(env, qs):
    r = env.get(Q + "?" + qs)
    assert r.status_code == 422
    assert r.json()["success"] is False and r.json()["data"] is None and r.json()["message"]
    assert env.calls["avp"] == 0 and env.calls["ey"] == 0  # nothing computed for bad input


def test_unknown_etf_is_404_like_the_existing_api(env):
    r = env.get("/api/etfs/NOPE/strategies/quantitative_set3")
    assert r.status_code == 404
    assert r.json() == {"data": None, "success": False, "message": "Unknown ETF symbol: NOPE"}


def test_new_response_is_a_superset_of_the_existing_quantitative_response(env):
    """Same envelope + identical randomizedInvestmentSimulation as the registry router."""
    legacy_app = FastAPI()
    register_exception_handlers(legacy_app)
    legacy_app.include_router(registry_router.router, prefix="/api")
    legacy_app.dependency_overrides[get_db] = env.app_.dependency_overrides[get_db]
    legacy = TestClient(legacy_app, raise_server_exceptions=False).get(
        Q + "?iterations=5&seed=1").json()["data"]
    new = env.get(Q + "?" + FAST).json()["data"]
    assert set(legacy) <= set(new)
    assert set(legacy["strategies"]) <= set(new["strategies"])
    assert new["strategies"]["randomizedInvestmentSimulation"] == legacy["strategies"]["randomizedInvestmentSimulation"]


def test_other_four_set3_categories_are_still_served_by_the_registry_router(env):
    for cat in ("technical", "regime", "calendar", "portfolio"):
        r = env.get(f"/api/etfs/VOO/strategies/{cat}_set3")
        assert r.status_code == 200 and r.json()["data"]["category"] == cat
        assert r.json()["data"]["registeredStrategies"] == []


def test_composite_scoring_is_not_exposed_through_etf_strategy_apis(env):
    for cat in ("technical", "quantitative", "regime", "calendar", "portfolio"):
        text = env.get(f"/api/etfs/VOO/strategies/{cat}_set3?{FAST}").text
        assert "technical_fundamental_composite" not in text
        assert "technicalFundamentalComposite" not in text


# ---- applicability / data availability -------------------------------------


def test_bond_etf_fed_model_is_not_applicable_without_provider_calls(env):
    r = env.get("/api/etfs/SHY/strategies/quantitative_set3?strategies=" + FED_ID + "," + AVP_ID +
                "&avp_garch_enabled=false")
    st = r.json()["data"]["strategies"]
    assert st["fedModelRelativeValuation"]["calculationStatus"] == "not_applicable"
    assert st["fedModelRelativeValuation"]["calculatedValues"] is None
    assert env.calls["ey"] == 0 and env.calls["ty"] == 0
    assert st["asymmetricVolatilityPersistence"]["calculationStatus"] == "ok"  # independent
    assert FED_ID in r.json()["data"]["unavailable"]
    assert "not an equity ETF" in r.json()["message"]


def test_missing_earnings_yield_is_reported_not_fabricated(env):
    async def no_ey(symbol, basis):
        raise Set3DataUnavailable("No valid positive trailing P/E (trailingPE) is available for VOO")

    env.monkeypatch.setattr(data_service, "fetch_etf_earnings_yield", no_ey)
    r = env.get(Q + "?strategies=" + FED_ID)
    fedm = r.json()["data"]["strategies"]["fedModelRelativeValuation"]
    assert fedm["calculationStatus"] == "partial" and fedm["calculatedValues"] is None
    assert fedm["historicalSeries"]["earningsYield"] == []
    assert "trailingPE" in r.json()["data"]["unavailable"][FED_ID]


def test_avp_reports_not_ingested_when_no_prices_exist(env):
    env.rows["VOO"] = []
    env.monkeypatch.setattr(strategies_set3, "_newest_price_date", lambda db, s: _none())
    r = env.get(Q + "?strategies=" + AVP_ID + "," + FED_ID)
    st = r.json()["data"]["strategies"]
    assert st["asymmetricVolatilityPersistence"]["calculationStatus"] == "unavailable"
    assert "not been ingested" in st["asymmetricVolatilityPersistence"]["dataAvailability"]["message"]
    assert st["fedModelRelativeValuation"]["calculationStatus"] == "ok"  # independent of prices


async def _none():
    return None


def test_short_history_gives_unavailable_with_message(env):
    env.rows["VOO"] = make_rows(n=30)
    r = env.get(Q + "?strategies=" + AVP_ID)
    s = r.json()["data"]["strategies"]["asymmetricVolatilityPersistence"]
    assert s["calculationStatus"] == "unavailable" and s["calculatedValues"] is None
    assert "Insufficient history" in s["dataAvailability"]["message"]
    assert r.json()["data"]["unavailable"][AVP_ID]


# ---- error isolation -------------------------------------------------------


def test_avp_failure_does_not_break_the_other_strategies(env):
    def boom(*a, **k):
        raise RuntimeError("numpy exploded: secret path /srv/x")

    env.monkeypatch.setattr(avp, "compute_asymmetric_volatility_persistence", boom)
    r = env.get(Q + "?" + FAST)
    assert r.status_code == 200
    st = r.json()["data"]["strategies"]
    assert st["asymmetricVolatilityPersistence"]["calculationStatus"] == "error"
    assert st["asymmetricVolatilityPersistence"]["error"]
    assert "secret" not in r.text
    assert st["fedModelRelativeValuation"]["calculationStatus"] == "ok"
    assert st["randomizedInvestmentSimulation"] is not None
    assert AVP_ID in r.json()["data"]["unavailable"]


def test_fed_provider_failure_does_not_break_the_other_strategies(env):
    async def boom(symbol, basis):
        raise RuntimeError("yahoo 500")

    async def boom2(years):
        raise RuntimeError("yahoo 500")

    env.monkeypatch.setattr(data_service, "fetch_etf_earnings_yield", boom)
    env.monkeypatch.setattr(data_service, "fetch_treasury_yield_history", boom2)
    r = env.get(Q + "?" + FAST)
    st = r.json()["data"]["strategies"]
    assert st["fedModelRelativeValuation"]["calculationStatus"] == "unavailable"
    assert st["asymmetricVolatilityPersistence"]["calculationStatus"] == "ok"
    assert st["randomizedInvestmentSimulation"] is not None


def test_randomized_failure_does_not_break_the_new_strategies(env):
    async def boom(*a, **k):
        raise RuntimeError("sim exploded")

    env.monkeypatch.setattr(registry_router, "_run_category", boom)
    r = env.get(Q + "?avp_garch_enabled=false")
    d = r.json()["data"]
    assert r.status_code == 200
    assert d["strategies"]["randomizedInvestmentSimulation"] is None
    assert RAND_ID in d["unavailable"]
    assert d["strategies"]["asymmetricVolatilityPersistence"]["calculationStatus"] == "ok"
    assert d["strategies"]["fedModelRelativeValuation"]["calculationStatus"] == "ok"


def test_failed_results_are_never_cached(env):
    def boom(*a, **k):
        raise RuntimeError("x")

    env.monkeypatch.setattr(avp, "compute_asymmetric_volatility_persistence", boom)
    env.get(Q + "?strategies=" + AVP_ID)
    assert env.cache.sets == []


# ---- cache isolation -------------------------------------------------------


def test_cache_keys_live_in_the_set3_namespace_only(env):
    env.get(Q + "?" + FAST)
    newest = env.rows["VOO"][-1].date.isoformat()
    assert env.cache.sets, "expected cache writes"
    assert all(k.startswith("tw:set3:") for k in env.cache.sets + env.cache.gets)
    assert not any(k.startswith("tw:v") for k in env.cache.sets + env.cache.gets)
    mine = [k for k in env.cache.sets if k.startswith(f"tw:set3:quantitative:VOO:{newest}:")]
    assert len(mine) >= 2  # AVP + Fed (the existing simulation writes its own Set 3 key)
    for k in mine:
        assert len(k.split(":")[-1]) == 16  # fingerprint


def test_second_identical_request_is_served_from_cache(env):
    url = Q + "?strategies=" + AVP_ID + "," + FED_ID + "&avp_garch_enabled=false"
    first = env.get(url).json()["data"]["strategies"]
    assert env.calls["avp"] == 1 and env.calls["ey"] == 1
    second = env.get(url).json()["data"]["strategies"]
    assert env.calls["avp"] == 1 and env.calls["ey"] == 1 and env.calls["ty"] == 1
    assert second == first


def test_different_parameters_never_reuse_each_others_cache_entries(env):
    base = Q + "?strategies=" + AVP_ID + "," + FED_ID + "&avp_garch_enabled=false"
    env.get(base)
    env.get(base + "&avp_vol_window=10")
    assert env.calls["avp"] == 2 and env.calls["ey"] == 1  # fed params unchanged -> cached
    env.get(base + "&fed_pe_basis=forward")
    assert env.calls["ey"] == 2 and env.calls["avp"] == 2
    keys = [k for k in env.cache.sets if ":VOO:" in k]
    assert len(set(keys)) == len(keys) == 4


def test_new_price_date_produces_a_new_cache_key(env):
    url = Q + "?strategies=" + AVP_ID + "&avp_garch_enabled=false"
    env.get(url)
    env.rows["VOO"] = make_rows(n=601, seed=1)  # one more trading day ingested
    env.get(url)
    assert env.calls["avp"] == 2
    assert len({k.split(":")[3 + 1] for k in env.cache.sets if ":VOO:" in k}) == 2


def test_redis_failure_falls_back_to_computation(env):
    env.monkeypatch.setattr(strat_cache, "cache", FakeCache(broken=True))
    r = env.get(Q + "?strategies=" + AVP_ID + "," + FED_ID + "&avp_garch_enabled=false")
    st = r.json()["data"]["strategies"]
    assert r.status_code == 200
    assert st["asymmetricVolatilityPersistence"]["calculationStatus"] == "ok"
    assert st["fedModelRelativeValuation"]["calculationStatus"] == "ok"


def test_corrupt_cache_entry_is_ignored(env):
    url = Q + "?strategies=" + AVP_ID + "&avp_garch_enabled=false"
    env.get(url)
    key = next(k for k in env.cache.sets if ":VOO:" in k)
    env.cache.store[key] = {"strategyId": "garbage"}
    r = env.get(url)
    assert r.json()["data"]["strategies"]["asymmetricVolatilityPersistence"]["calculationStatus"] == "ok"


# ===========================================================================
# Stock scanner: POST /api/scanners/quantitative_set3
# ===========================================================================

S = "/api/scanners/quantitative_set3"


def test_scanner_returns_scores_with_full_breakdown(env):
    env.stocks["AAPL"] = (zigzag_up(), 12.0, 0.4)
    r = env.post(S, json={"symbols": ["aapl"]})
    assert r.status_code == 200, r.text
    parsed = ApiResponse[QuantitativeScannerSet3Response].model_validate(r.json())
    res = parsed.data.results[0]
    assert res.symbol == "AAPL" and res.total_score == 4 and res.max_score == 4
    assert res.calculation_status == "complete" and res.error is None
    raw = r.json()["data"]["results"][0]
    for field in ("symbol", "currentPrice", "movingAverage", "priceAboveMa", "rsi", "rsiBelowUpper",
                  "forwardPe", "forwardPeBelowThreshold", "debtToEquity", "debtToEquityBelowThreshold",
                  "individualScores", "totalScore", "maxScore", "rsiOpportunity", "dataAvailability",
                  "calculationStatus", "error"):
        assert field in raw, field
    assert raw["individualScores"] == {"priceAboveMa": 1, "rsiBelowUpper": 1,
                                       "forwardPeBelowThreshold": 1, "debtToEquityBelowThreshold": 1}
    assert raw["rsiOpportunity"]["threshold"] == 30.0
    assert r.json()["data"]["parameters"]["maPeriod"] == 50


def test_scanner_configurable_thresholds_via_camel_case_body(env):
    env.stocks["AAPL"] = (zigzag_up(), 12.0, 0.4)
    r = env.post(S, json={"symbols": ["AAPL"], "forwardPeThreshold": 10, "debtToEquityThreshold": 0.3,
                          "maPeriod": 30, "rsiPeriod": 10, "rsiUpperThreshold": 40,
                          "rsiOpportunityThreshold": 20})
    res = r.json()["data"]["results"][0]
    assert res["forwardPeBelowThreshold"]["status"] == "failed"
    assert res["debtToEquityBelowThreshold"]["status"] == "failed"
    assert res["rsiBelowUpper"]["status"] == "failed"  # RSI > 40
    assert res["totalScore"] == 1  # only price > 30d MA
    assert r.json()["data"]["parameters"] == {
        "maPeriod": 30, "rsiPeriod": 10, "rsiUpperThreshold": 40.0, "rsiOpportunityThreshold": 20.0,
        "forwardPeThreshold": 10.0, "debtToEquityThreshold": 0.3}


def test_scanner_missing_fundamentals_are_unavailable_not_zero(env):
    env.stocks["XYZ"] = (zigzag_up(), None, None)
    res = env.post(S, json={"symbols": ["XYZ"]}).json()["data"]["results"][0]
    assert res["forwardPe"] is None and res["debtToEquity"] is None
    assert res["forwardPeBelowThreshold"]["status"] == "unavailable"
    assert res["individualScores"]["forwardPeBelowThreshold"] is None
    assert res["calculationStatus"] == "partial" and res["scoreComplete"] is False
    assert res["totalScore"] == 2


def test_scanner_isolates_failures_per_symbol(env):
    env.stocks.update({"GOOD": (zigzag_up(), 12.0, 0.4), "BOOM": RuntimeError("token=abc123"),
                       "EMPTY": (None, None, None)})
    r = env.post(S, json={"symbols": ["GOOD", "BOOM", "EMPTY"]})
    assert r.status_code == 200
    by = {x["symbol"]: x for x in r.json()["data"]["results"]}
    assert by["GOOD"]["totalScore"] == 4
    assert by["BOOM"]["calculationStatus"] == "error" and by["BOOM"]["error"]
    assert "abc123" not in r.text
    assert by["EMPTY"]["calculationStatus"] == "unavailable"
    s = r.json()["data"]["summary"]
    assert (s["requested"], s["complete"], s["error"], s["unavailable"]) == (3, 1, 1, 1)
    assert "could not be scored" in r.json()["message"]


@pytest.mark.parametrize("body", [
    {}, {"symbols": []}, {"symbols": ["BAD SYMBOL"]}, {"symbols": ["AAPL"], "maPeriod": 1},
    {"symbols": ["AAPL"], "rsiOpportunityThreshold": 75},
    {"symbols": ["AAPL"], "forwardPeThreshold": -5},
    {"symbols": [f"S{i}" for i in range(101)]},
    {"symbols": ["AAPL"], "rsiPeriod": "x"},
])
def test_scanner_request_validation_returns_422_envelope(env, body):
    r = env.post(S, json=body)
    assert r.status_code == 422
    assert r.json()["success"] is False and r.json()["data"] is None
    assert env.calls["stock"] == 0


def test_scanner_rejects_get_and_is_separate_from_canslim_routes(env):
    assert env.get(S).status_code == 405


def test_scanner_cache_namespace_and_reuse(env):
    env.stocks["AAPL"] = (zigzag_up(), 12.0, 0.4)
    env.post(S, json={"symbols": ["AAPL"]})
    env.post(S, json={"symbols": ["AAPL"]})
    assert env.calls["stock"] == 1
    assert env.cache.sets and all(k.startswith("tw:set3:scanner:quantitative:AAPL:") for k in env.cache.sets)
    assert not any(k.startswith("tw:v") for k in env.cache.sets + env.cache.gets)
    env.post(S, json={"symbols": ["AAPL"], "forwardPeThreshold": 11})
    assert env.calls["stock"] == 2  # different parameters -> different key


def test_scanner_survives_redis_failure(env):
    env.monkeypatch.setattr(scanner_cache, "cache", FakeCache(broken=True))
    env.stocks["AAPL"] = (zigzag_up(), 12.0, 0.4)
    assert env.post(S, json={"symbols": ["AAPL"]}).json()["data"]["results"][0]["totalScore"] == 4


# ---- universe handling -----------------------------------------------------


def _patch_universe(env, members):
    async def get_universe(db, key):
        return members

    env.monkeypatch.setattr(scanners_set3.scanner_universe, "get_stock_universe", get_universe)


def test_universe_members_are_scanned_up_to_the_limit(env):
    members = [types.SimpleNamespace(symbol=s) for s in ("AAA", "BBB", "CCC", "DDD")]
    _patch_universe(env, members)
    for s in ("AAA", "BBB", "CCC", "DDD", "ZZZ"):
        env.stocks[s] = (zigzag_up(), 12.0, 0.4)
    r = env.post(S, json={"universe": "sp500", "universeLimit": 2, "symbols": ["ZZZ"]})
    d = r.json()["data"]
    assert [x["symbol"] for x in d["results"]] == ["ZZZ", "AAA", "BBB"]
    assert d["universe"] == "sp500"
    assert any("first 2" in n for n in d["notes"])


def test_unknown_universe_is_404(env):
    r = env.post(S, json={"universe": "nope"})
    assert r.status_code == 404 and "Unknown universe" in r.json()["message"]


def test_empty_universe_without_symbols_is_503_with_clear_message(env):
    _patch_universe(env, [])
    r = env.post(S, json={"universe": "sp500"})
    assert r.status_code == 503 and "not been ingested" in r.json()["message"]


def test_empty_universe_with_explicit_symbols_still_scans_them(env):
    _patch_universe(env, [])
    env.stocks["AAPL"] = (zigzag_up(), 12.0, 0.4)
    r = env.post(S, json={"universe": "sp500", "symbols": ["AAPL"]})
    assert r.status_code == 200 and r.json()["data"]["results"][0]["symbol"] == "AAPL"
    assert any("not ingested" in n for n in r.json()["data"]["notes"])


# ===========================================================================
# Existing application unchanged
# ===========================================================================


def test_only_additive_routes_were_added_to_the_application():
    import app.main as main
    from tests.unit.test_valuation_strategies import ORIGINAL_ROUTES

    current = {(m.upper(), p) for p, v in main.app.openapi()["paths"].items() for m in v}
    assert ORIGINAL_ROUTES <= current, "an original route disappeared"
    added = current - ORIGINAL_ROUTES
    assert added == {
        ("GET", "/api/etfs/{symbol}/strategies/technical_set3"),
        ("GET", "/api/etfs/{symbol}/strategies/quantitative_set3"),
        ("GET", "/api/etfs/{symbol}/strategies/regime_set3"),
        ("GET", "/api/etfs/{symbol}/strategies/calendar_set3"),
        ("GET", "/api/etfs/{symbol}/strategies/portfolio_set3"),
        ("POST", "/api/scanners/quantitative_set3"),
    }


def test_existing_strategies_endpoint_is_still_served_by_the_original_router():
    import app.main as main

    routes = [r for r in main.app.routes
              if getattr(r, "path", "") == "/api/etfs/{symbol}/strategies"]
    assert len(routes) == 1 and routes[0].methods == {"GET"}
    assert routes[0].endpoint.__module__ == "app.routers.strategy"


def test_new_quantitative_route_takes_precedence_over_the_registry_route():
    import app.main as main

    matches = [r for r in main.app.routes
               if getattr(r, "path", "") == "/api/etfs/{symbol}/strategies/quantitative_set3"]
    assert [r.endpoint.__module__ for r in matches] == [
        "app.routers.strategies_set3", "app.routers.strategy_set3"]


def test_existing_cache_prefixes_are_untouched_by_set3_modules():
    import inspect

    for mod in (strat_cache, scanner_cache):
        src = inspect.getsource(mod)
        assert "delete_prefix" not in src and "scan_iter" not in src and "tw:v" not in src.replace("tw:v*", "")
