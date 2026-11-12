"""
Tests for the Technical-Fundamental Composite Scoring scanner
(app.scanners.set_3_scanners). Providers and Redis are replaced by fakes.
"""

import asyncio
import datetime
import itertools
import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("EODHD_API_TOKEN", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.scanners.set_3_scanners.quantitative import (  # noqa: E402
    technical_fundamental_composite as tfcs,
)
from app.scanners.set_3_scanners.schemas.quantitative import (  # noqa: E402
    EffectiveParameters,
    ScannerRequest,
)
from app.scanners.set_3_scanners.services import (  # noqa: E402
    cache_service,
    scoring_service as sc,
    stock_data_service as sds,
)

P = EffectiveParameters(
    ma_period=50, rsi_period=14, rsi_upper_threshold=70.0, rsi_opportunity_threshold=30.0,
    forward_pe_threshold=20.0, debt_to_equity_threshold=1.0,
)


def closes(values, end="2026-11-10"):
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(values))
    return pd.Series(np.asarray(values, dtype=float), index=idx)


def rising(n=120, start=100.0, step=0.5):
    return closes(start + step * np.arange(n))


def falling(n=120, start=200.0, step=0.8):
    return closes(start - step * np.arange(n))


def zigzag_up(n=120):
    """Uptrend with alternating +/- days -> RSI well below 70, price above MA."""
    base = 100 + 0.2 * np.arange(n)
    return closes(base + np.where(np.arange(n) % 2 == 0, 1.5, -1.5))


def zigzag_down(n=120):
    """Downtrend with alternating +/- days -> price below its MA, RSI mid-range (not 0)."""
    base = 200 - 0.4 * np.arange(n)
    return closes(base + np.where(np.arange(n) % 2 == 0, 1.5, -1.5))


def score(close, pe=15.0, de=0.5, params=P, **kw):
    return sc.score_stock("TEST", close, pe, de, params, **kw)


# ---------------------------------------------------------------------------
# 1 / 2. Price above / below the 50-day MA
# ---------------------------------------------------------------------------


def test_price_above_50_day_ma_passes():
    c = rising()
    r = score(c)
    ma = float(c.iloc[-50:].mean())
    assert r.moving_average == pytest.approx(ma, abs=1e-3)
    assert r.current_price == pytest.approx(float(c.iloc[-1]))
    assert r.price_above_ma.status == "passed" and r.price_above_ma.score == 1
    assert r.individual_scores.price_above_ma == 1


def test_price_below_50_day_ma_fails_with_zero_points():
    r = score(falling())
    assert r.price_above_ma.status == "failed"
    assert r.price_above_ma.passed is False and r.price_above_ma.score == 0


def test_price_equal_to_ma_is_not_above():
    r = score(closes([100.0] * 60))
    assert r.price_above_ma.status == "failed"  # strict "above"


def test_ma_period_is_configurable():
    c = closes([100.0] * 40 + [110.0] * 10)
    p20 = P.model_copy(update={"ma_period": 20})
    assert score(c, params=p20).moving_average == pytest.approx(float(c.iloc[-20:].mean()))


# ---------------------------------------------------------------------------
# 3. RSI below 70 (and RSI correctness)
# ---------------------------------------------------------------------------

WILDER = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
          46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]


def reference_rsi(values, period):
    """Independent brute-force Wilder RSI."""
    d = [b - a for a, b in zip(values[:-1], values[1:], strict=True)]
    ag = sum(max(x, 0) for x in d[:period]) / period
    al = sum(max(-x, 0) for x in d[:period]) / period
    for x in d[period:]:
        ag = (ag * (period - 1) + max(x, 0)) / period
        al = (al * (period - 1) + max(-x, 0)) / period
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


@pytest.mark.parametrize("n", [15, 16, 18, 20])
def test_wilder_rsi_matches_independent_implementation(n):
    assert sc.wilder_rsi(closes(WILDER[:n]), 14) == pytest.approx(reference_rsi(WILDER[:n], 14), abs=1e-9)


def test_wilder_rsi_close_to_published_textbook_values():
    # StockCharts' worked example (their sheet rounds the first averages, hence +/-0.1)
    for n, published in [(15, 70.53), (16, 66.32), (18, 69.41), (20, 57.97)]:
        assert sc.wilder_rsi(closes(WILDER[:n]), 14) == pytest.approx(published, abs=0.1)


def test_rsi_below_70_passes_and_above_fails():
    ok = score(zigzag_up())
    assert ok.rsi < 70 and ok.rsi_below_upper.status == "passed"
    hot = score(rising())  # monotone rise -> RSI 100
    assert hot.rsi == pytest.approx(100.0)
    assert hot.rsi_below_upper.status == "failed" and hot.rsi_below_upper.score == 0


def test_rsi_upper_threshold_and_period_are_configurable():
    c = zigzag_up()
    r = score(c)
    strict = P.model_copy(update={"rsi_upper_threshold": max(r.rsi - 1, 1.0)})
    assert score(c, params=strict).rsi_below_upper.status == "failed"
    p5 = P.model_copy(update={"rsi_period": 5})
    assert score(c, params=p5).rsi == pytest.approx(reference_rsi(list(c.values), 5), abs=1e-3)


def test_flat_prices_leave_rsi_undefined_not_fabricated():
    r = score(closes([50.0] * 80))
    assert r.rsi is None and r.rsi_below_upper.status == "unavailable"
    assert r.rsi_opportunity.flagged is None


# ---------------------------------------------------------------------------
# 4. RSI below 30 opportunity flag (separate from the score)
# ---------------------------------------------------------------------------


def test_rsi_below_30_sets_opportunity_flag_without_adding_a_point():
    r = score(falling(), pe=15.0, de=0.5)
    assert r.rsi < 30
    assert r.rsi_opportunity.flagged is True and r.rsi_opportunity.threshold == 30.0
    # RSI < 70 point is earned, MA point is not, fundamentals earn 2 -> 3, and the
    # opportunity flag adds nothing on top.
    assert r.individual_scores.rsi_below_upper == 1
    assert r.individual_scores.price_above_ma == 0
    assert r.total_score == 3
    assert not hasattr(r.individual_scores, "rsi_below_opportunity_threshold")


def test_opportunity_flag_false_when_rsi_not_that_low():
    r = score(zigzag_up())
    assert r.rsi_opportunity.flagged is False


def test_opportunity_threshold_configurable():
    p = P.model_copy(update={"rsi_opportunity_threshold": 80.0, "rsi_upper_threshold": 90.0})
    assert score(zigzag_up(), params=p).rsi_opportunity.flagged is True


# ---------------------------------------------------------------------------
# 5. Forward P/E threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pe,status,points", [(19.99, "passed", 1), (20.0, "failed", 0),
                                              (35.0, "failed", 0), (5.0, "passed", 1)])
def test_forward_pe_threshold(pe, status, points):
    c = score(rising(), pe=pe).forward_pe_below_threshold
    assert (c.status, c.score) == (status, points)


def test_forward_pe_threshold_is_configurable():
    p = P.model_copy(update={"forward_pe_threshold": 30.0})
    assert score(rising(), pe=25.0, params=p).forward_pe_below_threshold.status == "passed"


@pytest.mark.parametrize("pe", [0.0, -12.0])
def test_non_positive_forward_pe_is_not_meaningful_and_earns_no_point(pe):
    c = score(rising(), pe=pe).forward_pe_below_threshold
    assert c.status == "not_meaningful" and c.score == 0 and c.passed is False
    assert "no point" in c.message


# ---------------------------------------------------------------------------
# 6. Debt-to-equity threshold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("de,status,points", [(0.99, "passed", 1), (1.0, "failed", 0),
                                              (2.5, "failed", 0), (0.0, "passed", 1)])
def test_debt_to_equity_threshold(de, status, points):
    c = score(rising(), de=de).debt_to_equity_below_threshold
    assert (c.status, c.score) == (status, points)


def test_negative_debt_to_equity_is_not_meaningful():
    c = score(rising(), de=-3.2).debt_to_equity_below_threshold
    assert c.status == "not_meaningful" and c.score == 0


def test_provider_percentage_is_converted_to_a_ratio():
    # Yahoo reports debtToEquity as a percentage: 154.6 means 1.546x
    fpe, de, raw = sds.extract_fundamentals({"forwardPE": 18.0, "debtToEquity": 154.6})
    assert de == pytest.approx(1.546) and raw == 154.6 and fpe == 18.0
    r = score(rising(), de=de, debt_to_equity_provider_value=raw)
    assert r.debt_to_equity_below_threshold.status == "failed"
    assert r.debt_to_equity_provider_value == 154.6
    _, low, _ = sds.extract_fundamentals({"debtToEquity": 45.0})
    assert low == pytest.approx(0.45)
    assert score(rising(), de=low).debt_to_equity_below_threshold.status == "passed"


# ---------------------------------------------------------------------------
# 7. Total score 0..4
# ---------------------------------------------------------------------------


def test_total_score_is_sum_of_independent_points_for_all_16_combinations():
    """Every pass/fail combination of the four conditions yields total = number of
    passes (0..4). A rising / falling series sets the MA outcome; the (configurable)
    RSI threshold is placed just above / below the measured RSI to set the RSI one."""
    seen = set()
    for ma_ok, rsi_ok, pe_ok, de_ok in itertools.product([True, False], repeat=4):
        c = zigzag_up() if ma_ok else zigzag_down()
        rsi = score(c).rsi
        thr = min(rsi + 5.0, 100.0) if rsi_ok else max(rsi - 5.0, 1.0)
        params = P.model_copy(update={"rsi_upper_threshold": thr})
        r = score(c, pe=10.0 if pe_ok else 40.0, de=0.4 if de_ok else 2.0, params=params)
        assert r.price_above_ma.passed is ma_ok
        assert r.rsi_below_upper.passed is rsi_ok
        assert r.forward_pe_below_threshold.passed is pe_ok
        assert r.debt_to_equity_below_threshold.passed is de_ok
        expected = int(ma_ok) + int(rsi_ok) + int(pe_ok) + int(de_ok)
        assert r.total_score == expected and 0 <= r.total_score <= 4
        assert r.max_score == 4 and r.score_complete and r.calculation_status == "complete"
        seen.add(r.total_score)
    assert seen == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# 8. Missing fundamental metrics
# ---------------------------------------------------------------------------


def test_missing_forward_pe_is_unavailable_not_a_pass_or_zero():
    r = score(rising(), pe=None, de=0.5)
    c = r.forward_pe_below_threshold
    assert c.status == "unavailable" and c.passed is None and c.score is None
    assert r.individual_scores.forward_pe_below_threshold is None
    assert r.forward_pe is None  # never defaulted to 0
    assert r.total_score == 2  # MA + D/E only (RSI of a monotone rise fails)
    assert r.evaluated_conditions == 3 and r.score_complete is False
    assert r.calculation_status == "partial"
    assert r.data_availability.forward_pe_available is False
    assert any("forward_pe_below_threshold" in m for m in r.data_availability.messages)


def test_missing_debt_to_equity_is_unavailable():
    r = score(zigzag_up(), pe=10.0, de=None)
    assert r.debt_to_equity_below_threshold.status == "unavailable"
    assert r.debt_to_equity is None and r.total_score == 3 and r.calculation_status == "partial"


def test_all_fundamentals_missing_still_scores_technicals():
    r = score(zigzag_up(), pe=None, de=None)
    assert r.total_score == 2 and r.evaluated_conditions == 2 and r.calculation_status == "partial"


def test_everything_missing_gives_null_total_and_unavailable_status():
    r = score(None, pe=None, de=None)
    assert r.total_score is None and r.calculation_status == "unavailable"
    assert r.current_price is None and r.evaluated_conditions == 0
    assert any("No price history" in m for m in r.data_availability.messages)


@pytest.mark.parametrize("info,expected", [
    (None, (None, None, None)), ({}, (None, None, None)),
    ({"forwardPE": "Infinity", "debtToEquity": "N/A"}, (None, None, None)),
    ({"forwardPE": float("nan")}, (None, None, None)),
    ({"forwardPE": True}, (None, None, None)),
    ({"forwardPE": "21.5"}, (21.5, None, None)),
])
def test_fundamental_extraction_never_fabricates_values(info, expected):
    assert sds.extract_fundamentals(info) == expected


# ---------------------------------------------------------------------------
# 9. Insufficient historical observations
# ---------------------------------------------------------------------------


def test_fewer_than_ma_period_observations_makes_ma_unavailable_but_rsi_works():
    r = score(zigzag_up(30))
    assert r.moving_average is None and r.price_above_ma.status == "unavailable"
    assert r.rsi is not None and r.rsi_below_upper.score is not None
    assert r.data_availability.price_history_observations == 30
    assert r.data_availability.required_price_observations == 50
    assert r.data_availability.price_history_available is False
    assert any("Only 30 price observations" in m for m in r.data_availability.messages)


def test_five_days_of_history_cannot_produce_a_50_day_average():
    """Regression test for the article's bug (5 days of data, 50-day MA)."""
    r = score(rising(5))
    assert r.moving_average is None and r.price_above_ma.score is None
    assert r.rsi is None and r.rsi_below_upper.status == "unavailable"


def test_exactly_ma_period_observations_is_enough():
    assert score(rising(50)).moving_average is not None


def test_fetch_requests_far_more_than_five_days_of_history(monkeypatch):
    seen = {}

    async def fake_hist(symbol, start, end):
        seen.update(start=start, end=end)
        return None

    async def fake_info(symbol):
        return {"forwardPE": 15, "debtToEquity": 50}

    monkeypatch.setattr(sds.yfinance_client, "fetch_price_history_range", fake_hist)
    monkeypatch.setattr(sds.yfinance_client, "fetch_info", fake_info)
    out = asyncio.run(sds.fetch_stock_inputs("AAPL", 50, 14))
    assert sc.fetch_target_observations(50, 14) >= 100 > 50
    span = (seen["end"] - seen["start"]).days
    assert span >= 50 * 1.4  # calendar days covering >= 50 trading days
    assert out.close is None and any("no price history" in m for m in out.messages)
    assert out.forward_pe == 15 and out.debt_to_equity == pytest.approx(0.5)


def test_extract_close_series_cleans_provider_frame():
    idx = pd.date_range("2026-01-01", periods=5, freq="B", tz="America/New_York")
    df = pd.DataFrame({"Close": [10.0, np.nan, -2.0, 11.0, 12.0]}, index=idx)
    s = sds.extract_close_series(df)
    assert list(s.values) == [10.0, 11.0, 12.0] and s.index.tz is None
    assert sds.extract_close_series(None) is None
    assert sds.extract_close_series(pd.DataFrame({"Open": [1.0]})) is None


# ---------------------------------------------------------------------------
# 10. Multiple-stock error isolation
# ---------------------------------------------------------------------------


class FakeCache:
    def __init__(self):
        self.store, self.sets = {}, []

    async def get_json(self, key):
        return self.store.get(key)

    async def set_json(self, key, value, ttl=None):
        self.store[key] = value
        self.sets.append((key, ttl))


@pytest.fixture
def fake_cache(monkeypatch):
    fc = FakeCache()
    monkeypatch.setattr(cache_service, "cache", fc)
    return fc


def fetcher_for(table):
    async def fetch(symbol, ma, rsi):
        behaviour = table[symbol]
        if isinstance(behaviour, Exception):
            raise behaviour
        close, pe, de = behaviour
        return sds.StockInputs(symbol, close, pe, de, None if de is None else de * 100)
    return fetch


def test_one_failing_symbol_does_not_stop_the_others(fake_cache):
    table = {
        "GOOD": (zigzag_up(), 12.0, 0.4),
        "BOOM": RuntimeError("provider exploded: secret-token-123"),
        "NODATA": (None, None, None),
        "PART": (rising(), None, 0.4),
    }
    req = ScannerRequest(symbols=list(table))
    resp = asyncio.run(tfcs.scan_symbols(list(table), req, fetcher=fetcher_for(table)))
    by = {r.symbol: r for r in resp.results}
    assert [r.symbol for r in resp.results] == list(table)  # request order kept
    assert by["GOOD"].calculation_status == "complete" and by["GOOD"].total_score == 4
    assert by["BOOM"].calculation_status == "error" and by["BOOM"].total_score is None
    assert "secret-token" not in (by["BOOM"].error or "")
    assert by["NODATA"].calculation_status == "unavailable"
    assert by["PART"].calculation_status == "partial"
    s = resp.summary
    assert (s.requested, s.complete, s.partial, s.unavailable, s.error) == (4, 1, 1, 1, 1)
    assert s.score_distribution["4"] == 1
    assert sum(s.score_distribution.values()) == 2  # GOOD + PART have a total score


def test_summary_counts_opportunity_flags(fake_cache):
    table = {"DIP": (falling(), 15.0, 0.5), "UP": (zigzag_up(), 15.0, 0.5)}
    resp = asyncio.run(tfcs.scan_symbols(list(table), ScannerRequest(symbols=list(table)),
                                         fetcher=fetcher_for(table)))
    assert resp.summary.rsi_opportunity_count == 1
    assert resp.parameters.ma_period == 50 and resp.scanner_id == "technical_fundamental_composite_scoring"


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_symbols_are_normalised_deduplicated_and_validated():
    r = ScannerRequest(symbols=[" aapl ", "AAPL", "brk-b", "", "msft"])
    assert r.symbols == ["AAPL", "BRK-B", "MSFT"]
    assert ScannerRequest(symbols="aapl, msft").symbols == ["AAPL", "MSFT"]
    assert ScannerRequest(universe="sp500").symbols == []


@pytest.mark.parametrize("kw", [
    {}, {"symbols": []}, {"symbols": ["BAD SYMBOL"]}, {"symbols": ["^GSPC"]},
    {"symbols": ["A"], "ma_period": 1}, {"symbols": ["A"], "rsi_period": 1},
    {"symbols": ["A"], "rsi_upper_threshold": 0}, {"symbols": ["A"], "rsi_upper_threshold": 101},
    {"symbols": ["A"], "rsi_opportunity_threshold": 70.0},
    {"symbols": ["A"], "rsi_opportunity_threshold": 80.0},
    {"symbols": ["A"], "forward_pe_threshold": 0}, {"symbols": ["A"], "debt_to_equity_threshold": -1},
    {"symbols": [f"S{i}" for i in range(101)]},
])
def test_invalid_requests_rejected(kw):
    with pytest.raises(ValidationError):
        ScannerRequest(**kw)


def test_request_defaults_match_the_article_and_accept_camel_case():
    r = ScannerRequest(symbols=["AAPL"])
    assert (r.ma_period, r.rsi_period, r.rsi_upper_threshold, r.forward_pe_threshold,
            r.debt_to_equity_threshold, r.rsi_opportunity_threshold) == (50, 14, 70.0, 20.0, 1.0, 30.0)
    assert ScannerRequest(symbols=["A"], maPeriod=30, forwardPeThreshold=25).forward_pe_threshold == 25


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


def test_cache_keys_use_scanner_namespace_and_separate_parameter_sets():
    fp1 = cache_service.scanner_fingerprint("sid", P, "v1")
    fp2 = cache_service.scanner_fingerprint("sid", P.model_copy(update={"forward_pe_threshold": 25.0}), "v1")
    fp3 = cache_service.scanner_fingerprint("sid", P, "v2")
    assert len({fp1, fp2, fp3}) == 3
    key = cache_service.scanner_cache_key("quantitative", "aapl", "2026-11-10", fp1)
    assert key == f"tw:set3:scanner:quantitative:AAPL:2026-11-10:{fp1}"
    assert not key.startswith("tw:v")


def test_second_identical_scan_is_served_from_cache(fake_cache):
    calls = {"n": 0}
    base = fetcher_for({"AAPL": (zigzag_up(), 12.0, 0.4)})

    async def counting(symbol, ma, rsi):
        calls["n"] += 1
        return await base(symbol, ma, rsi)

    req = ScannerRequest(symbols=["AAPL"])
    a = asyncio.run(tfcs.scan_symbols(["AAPL"], req, fetcher=counting))
    b = asyncio.run(tfcs.scan_symbols(["AAPL"], req, fetcher=counting))
    assert calls["n"] == 1 and a.results[0].total_score == b.results[0].total_score == 4
    assert all(k.startswith("tw:set3:scanner:") for k, _ in fake_cache.sets)
    # different thresholds must NOT reuse the entry
    req2 = ScannerRequest(symbols=["AAPL"], forward_pe_threshold=10.0)
    c = asyncio.run(tfcs.scan_symbols(["AAPL"], req2, fetcher=counting))
    assert calls["n"] == 2 and c.results[0].forward_pe_below_threshold.status == "failed"
    # use_cache=False bypasses reads
    asyncio.run(tfcs.scan_symbols(["AAPL"], ScannerRequest(symbols=["AAPL"], use_cache=False), fetcher=counting))
    assert calls["n"] == 3


def test_errors_and_unavailable_results_are_not_cached(fake_cache):
    table = {"BOOM": RuntimeError("x"), "NODATA": (None, None, None)}
    asyncio.run(tfcs.scan_symbols(list(table), ScannerRequest(symbols=list(table)),
                                  fetcher=fetcher_for(table)))
    assert fake_cache.sets == []


def test_cache_failures_fall_back_safely(monkeypatch):
    class Broken:
        async def get_json(self, key):
            raise ConnectionError("redis down")

        async def set_json(self, key, value, ttl=None):
            raise ConnectionError("redis down")

    monkeypatch.setattr(cache_service, "cache", Broken())
    table = {"AAPL": (zigzag_up(), 12.0, 0.4)}
    resp = asyncio.run(tfcs.scan_symbols(["AAPL"], ScannerRequest(symbols=["AAPL"]),
                                         fetcher=fetcher_for(table)))
    assert resp.results[0].calculation_status == "complete"


def test_unreadable_cache_entry_is_ignored_and_recomputed(fake_cache):
    req = ScannerRequest(symbols=["AAPL"])
    params = tfcs.effective_parameters(req)
    as_of = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    key = cache_service.scanner_cache_key(
        "quantitative", "AAPL", as_of,
        cache_service.scanner_fingerprint(tfcs.SCANNER_ID, params, tfcs.CALCULATION_VERSION),
    )
    fake_cache.store[key] = {"calculation_status": "not-a-status"}
    resp = asyncio.run(tfcs.scan_symbols(["AAPL"], req, fetcher=fetcher_for({"AAPL": (zigzag_up(), 12.0, 0.4)})))
    assert resp.results[0].calculation_status == "complete"
