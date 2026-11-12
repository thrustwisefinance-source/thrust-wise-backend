"""
Tests for the Fed Model Relative Valuation strategy
(app.strategies.set_3_strategies.quantitative_models.fed_model_relative_valuation)
and its data helpers. Providers are replaced by fakes - no network, DB or Redis.
"""

import asyncio
import datetime
import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("EODHD_API_TOKEN", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.constants import ETF_REGISTRY  # noqa: E402
from app.strategies.set_3_strategies.quantitative_models import (  # noqa: E402
    fed_model_relative_valuation as fed,
)
from app.strategies.set_3_strategies.schemas.quantitative import FedModelParams  # noqa: E402
from app.strategies.set_3_strategies.services import data_service  # noqa: E402
from app.strategies.set_3_strategies.services.data_service import (  # noqa: E402
    EarningsYieldInput,
    Set3DataUnavailable,
    TreasuryYieldInput,
)

TODAY = datetime.date(2026, 11, 10)


def make_treasury(values=None, end=TODAY, n=300):
    if values is None:
        values = np.linspace(3.5, 4.25, n)
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=len(values))
    s = pd.Series(np.asarray(values, dtype=float), index=idx)
    return TreasuryYieldInput(series=s, source="fake ^TNX", dropped_observations=0,
                              as_of=s.index[-1].date())


def make_earnings(pe=25.0, basis="trailing", as_of=TODAY):
    return EarningsYieldInput(pe_ratio=pe, pe_basis=basis,
                              field_name="trailingPE" if basis == "trailing" else "forwardPE",
                              source="fake yfinance", as_of=as_of)


def profile(symbol="VOO"):
    m = ETF_REGISTRY[symbol]
    return fed.EtfProfile(symbol, m.name, tuple(m.tags), m.asset_class)


def run(symbol="VOO", params=None, earnings=None, treasury=None, ey_exc=None, ty_exc=None):
    calls = {"ey": 0, "ty": 0}

    async def fake_ey(sym, basis):
        calls["ey"] += 1
        if ey_exc:
            raise ey_exc
        return earnings or make_earnings(basis=basis)

    async def fake_ty(years):
        calls["ty"] += 1
        if ty_exc:
            raise ty_exc
        return treasury or make_treasury()

    res = asyncio.run(
        fed.run_fed_model_relative_valuation(
            symbol, profile(symbol), params or FedModelParams(),
            fetch_earnings=fake_ey, fetch_treasury=fake_ty,
        )
    )
    return res, calls


# ---------------------------------------------------------------------------
# 5. Earnings-yield calculation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pe,expected", [(25.0, 4.0), (20.0, 5.0), (12.5, 8.0), (40.0, 2.5)])
def test_earnings_yield_is_inverse_of_pe_in_percent(pe, expected):
    res, _ = run(earnings=make_earnings(pe=pe))
    cv = res.calculated_values
    assert cv.earnings_yield_pct == pytest.approx(expected)
    assert cv.pe_ratio == pytest.approx(pe)
    assert cv.earnings_yield_is_proxy is True


def test_forward_pe_basis_is_passed_through():
    res, _ = run(params=FedModelParams(pe_basis="forward"))
    assert res.calculated_values.pe_basis == "forward"
    assert res.parameters.pe_basis == "forward"


# ---------------------------------------------------------------------------
# 6. Treasury-yield spread calculation
# ---------------------------------------------------------------------------


def test_spread_is_earnings_yield_minus_treasury_yield():
    res, _ = run(earnings=make_earnings(pe=20.0), treasury=make_treasury([4.0, 4.1, 4.25]))
    cv = res.calculated_values
    assert cv.treasury_yield_pct == pytest.approx(4.25)  # newest observation
    assert cv.fed_model_spread_pct == pytest.approx(5.0 - 4.25)
    assert cv.earnings_yield_as_of == TODAY.isoformat()
    assert cv.treasury_yield_as_of == TODAY.isoformat()
    assert cv.date_alignment_days == 0
    assert res.calculation_status == "ok"


@pytest.mark.parametrize(
    "spread,band,expected",
    [
        (1.0, 0.25, "equities_relatively_more_attractive"),
        (0.2501, 0.25, "equities_relatively_more_attractive"),
        (0.25, 0.25, "approximately_equal"),
        (0.0, 0.25, "approximately_equal"),
        (-0.25, 0.25, "approximately_equal"),
        (-0.2501, 0.25, "bonds_relatively_more_attractive"),
        (-2.0, 0.25, "bonds_relatively_more_attractive"),
        (0.4, 0.5, "approximately_equal"),
    ],
)
def test_relative_valuation_classification(spread, band, expected):
    assert fed.classify_spread(spread, band) == expected


def test_interpretation_matches_sign_of_spread():
    pos, _ = run(earnings=make_earnings(pe=15.0), treasury=make_treasury([4.0]))  # 6.67 - 4
    neg, _ = run(earnings=make_earnings(pe=40.0), treasury=make_treasury([4.5]))  # 2.5 - 4.5
    flat, _ = run(earnings=make_earnings(pe=25.0), treasury=make_treasury([4.0]))  # 4.0 - 4.0
    assert pos.interpretation.classification == "equities_relatively_more_attractive"
    assert "equities appear relatively more attractive" in pos.interpretation.summary
    assert neg.interpretation.classification == "bonds_relatively_more_attractive"
    assert "bonds appear relatively more attractive" in neg.interpretation.summary
    assert flat.interpretation.classification == "approximately_equal"
    assert "approximately equal" in flat.interpretation.summary
    assert "not a proven" in pos.interpretation.points[-1]


def test_treasury_stats_and_percentile():
    res, _ = run(treasury=make_treasury(np.linspace(1.0, 5.0, 100)))
    st = res.calculated_values.treasury_stats
    assert st.min_pct == 1.0 and st.max_pct == 5.0 and st.observations == 100
    assert st.current_percentile == 100.0
    assert st.lookback_years == 5


def test_stale_treasury_data_produces_alignment_warning():
    res, _ = run(treasury=make_treasury(end=TODAY - datetime.timedelta(days=12)))
    assert res.calculated_values.date_alignment_days >= 12
    assert any("days apart" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# 7. Missing earnings-yield handling (and missing Treasury)
# ---------------------------------------------------------------------------


def test_missing_earnings_yield_is_unavailable_with_real_treasury_history_only():
    res, _ = run(ey_exc=Set3DataUnavailable("No valid positive trailing P/E for VOO."))
    assert res.calculation_status == "partial"
    assert res.calculated_values is None  # no spread is invented
    assert "No valid positive trailing P/E" in res.data_availability.message
    assert res.data_availability.components["earnings_yield_current"].startswith("unavailable")
    assert res.historical_series.earnings_yield == []
    assert res.historical_series.fed_model_spread == []
    assert len(res.historical_series.treasury_yield) > 0  # real data still shown


def test_missing_treasury_is_unavailable():
    res, _ = run(ty_exc=Set3DataUnavailable("Treasury yield could not be retrieved."))
    assert res.calculation_status == "unavailable"
    assert res.calculated_values is None and res.historical_series is None
    assert "Treasury" in res.data_availability.message


def test_both_inputs_missing():
    res, _ = run(ey_exc=Set3DataUnavailable("no PE"), ty_exc=Set3DataUnavailable("no yield"))
    assert res.calculation_status == "unavailable"
    assert "no PE" in res.data_availability.message and "no yield" in res.data_availability.message


def test_unexpected_provider_exception_becomes_unavailable_without_leaking_details():
    res, _ = run(ey_exc=RuntimeError("socket exploded at 10.0.0.3"))
    assert res.calculation_status == "partial"
    assert "10.0.0.3" not in res.model_dump_json()
    assert "data-provider error" in res.data_availability.message


def test_historical_series_contains_only_real_observations():
    res, _ = run()
    h = res.historical_series
    assert len(h.earnings_yield) == 1 and len(h.fed_model_spread) == 1  # today's snapshot only
    assert h.earnings_yield_history_available is False
    assert "back-filled" in h.earnings_yield_history_note
    assert h.earnings_yield[0].date == TODAY.isoformat()
    assert h.fed_model_spread[0].value == pytest.approx(
        res.calculated_values.fed_model_spread_pct
    )
    assert res.data_availability.status == "partial"
    assert res.data_availability.components["earnings_yield_history"].startswith("unavailable")


def test_chart_data_merges_series_and_latest_comparison():
    res, _ = run(params=FedModelParams(chart_max_points=40))
    chart = res.chart_data
    assert len(chart.series) <= 41
    dates = [p.date for p in chart.series]
    assert dates == sorted(dates)
    last = chart.series[-1]
    assert last.date == TODAY.isoformat()
    assert last.earnings_yield_pct == pytest.approx(4.0)
    assert last.fed_model_spread_pct is not None
    assert [x.label for x in chart.latest_comparison][0].startswith("Earnings yield")
    assert sum(1 for p in chart.series if p.earnings_yield_pct is not None) == 1


def test_granger_is_reported_unavailable_and_distinguishes_concepts():
    res, _ = run()
    g = res.calculated_values.granger_causality
    assert g.status == "unavailable"
    assert "CPI" in g.message and "correlation" in g.message and "predictive" in g.message
    off, _ = run(params=FedModelParams(include_granger=False))
    assert off.calculated_values.granger_causality is None


# ---------------------------------------------------------------------------
# 8. Unsupported ETF handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["SHY", "GLD"])
def test_non_equity_etfs_are_not_applicable_and_make_no_data_calls(symbol):
    res, calls = run(symbol)
    assert res.calculation_status == "not_applicable"
    assert res.data_availability.status == "not_applicable"
    assert "not an equity ETF" in res.data_availability.message
    assert res.calculated_values is None and res.historical_series is None
    assert calls == {"ey": 0, "ty": 0}


@pytest.mark.parametrize("symbol", ["VOO", "VTI", "SPY", "QQQ"])
def test_equity_etfs_are_applicable(symbol):
    ok, reason = fed.check_equity_applicability(profile(symbol))
    assert ok and reason == ""


def test_unidentifiable_etf_type_is_not_applicable():
    ok, reason = fed.check_equity_applicability(fed.EtfProfile("XYZ", "X", (), "Alternative"))
    assert not ok and "could not be identified" in reason


def test_bond_tagged_etf_with_equity_word_is_still_not_applicable():
    ok, _ = fed.check_equity_applicability(
        fed.EtfProfile("CVT", "Convertible", ("Treasury",), "Equity-linked bonds")
    )
    assert not ok


# ---------------------------------------------------------------------------
# 9. Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [{"lookback_years": 0}, {"lookback_years": 41}, {"pe_basis": "peg"},
     {"neutral_band_pct": -0.1}, {"neutral_band_pct": 6}, {"chart_max_points": 5}],
)
def test_invalid_fed_parameters_rejected(kw):
    with pytest.raises(ValidationError):
        FedModelParams(**kw)


def test_fed_defaults():
    p = FedModelParams()
    assert (p.lookback_years, p.pe_basis, p.neutral_band_pct, p.include_granger) == (5, "trailing", 0.25, True)
    assert FedModelParams(lookbackYears=10).lookback_years == 10


# ---------------------------------------------------------------------------
# Data helpers: Treasury parsing & provider extraction (fake yfinance)
# ---------------------------------------------------------------------------


def test_parse_treasury_close_drops_invalid_and_counts():
    idx = pd.bdate_range("2026-01-01", periods=22, tz="America/New_York")
    vals = [4.0 + 0.01 * i for i in range(22)]
    vals[3], vals[7] = -1.0, 0.0  # invalid yields (10% of the series)
    vals[10] = np.nan  # missing, not counted as "dropped"
    out, dropped = data_service.parse_treasury_close(pd.Series(vals, index=idx))
    assert dropped == 2 and len(out) == 19
    assert (out > 0).all() and out.index.tz is None  # tz-naive daily index


def test_parse_treasury_close_rejects_mis_scaled_series_rather_than_rescaling():
    idx = pd.bdate_range("2026-01-01", periods=10)
    mis = pd.Series([42.5] * 8 + [4.2, 4.3], index=idx)  # looks like yield x 10
    with pytest.raises(Set3DataUnavailable, match="mis-scaled"):
        data_service.parse_treasury_close(mis)
    with pytest.raises(Set3DataUnavailable):
        data_service.parse_treasury_close(pd.Series([np.nan, np.nan], index=idx[:2]))


@pytest.mark.parametrize("v,expected", [(25, 25.0), ("19.5", 19.5), (0, None), (-3, None),
                                         (None, None), ("N/A", None), (float("inf"), None),
                                         (float("nan"), None), ("Infinity", None)])
def test_positive_finite(v, expected):
    assert data_service.positive_finite(v) == expected


def test_fetch_etf_earnings_yield_uses_requested_field(monkeypatch):
    async def fake_info(symbol):
        return {"trailingPE": 24.0, "forwardPE": 21.0}

    monkeypatch.setattr(data_service.yfinance_client, "fetch_info", fake_info)
    t = asyncio.run(data_service.fetch_etf_earnings_yield("VOO", "trailing"))
    f = asyncio.run(data_service.fetch_etf_earnings_yield("VOO", "forward"))
    assert (t.pe_ratio, t.field_name) == (24.0, "trailingPE")
    assert (f.pe_ratio, f.field_name) == (21.0, "forwardPE")
    assert t.is_proxy and "proxy" in t.source


@pytest.mark.parametrize("info", [None, {}, {"trailingPE": None}, {"trailingPE": -5},
                                  {"trailingPE": "Infinity"}, {"forwardPE": 20}])
def test_fetch_etf_earnings_yield_unavailable_for_missing_or_invalid_pe(monkeypatch, info):
    async def fake_info(symbol):
        return info

    monkeypatch.setattr(data_service.yfinance_client, "fetch_info", fake_info)
    with pytest.raises(Set3DataUnavailable):
        asyncio.run(data_service.fetch_etf_earnings_yield("VOO", "trailing"))


def test_fetch_treasury_history_unavailable_when_provider_returns_nothing(monkeypatch):
    async def none(*a, **k):
        return None

    monkeypatch.setattr(data_service.yfinance_client, "fetch_price_history_range", none)
    with pytest.raises(Set3DataUnavailable):
        asyncio.run(data_service.fetch_treasury_yield_history(5))


def test_fetch_treasury_history_requests_the_lookback_window(monkeypatch):
    seen = {}

    async def fake_hist(symbol, start, end):
        seen.update(symbol=symbol, start=start, end=end)
        idx = pd.bdate_range(end=pd.Timestamp(end), periods=50)
        return pd.DataFrame({"Close": np.linspace(4.0, 4.4, 50)}, index=idx)

    monkeypatch.setattr(data_service.yfinance_client, "fetch_price_history_range", fake_hist)
    out = asyncio.run(data_service.fetch_treasury_yield_history(3))
    assert seen["symbol"] == "^TNX"
    assert (seen["end"] - seen["start"]).days >= 3 * 365
    assert out.series.iloc[-1] == pytest.approx(4.4)


# ---------------------------------------------------------------------------
# 10. Independent error isolation
# ---------------------------------------------------------------------------


def test_fed_failure_does_not_affect_volatility_strategy():
    """The two strategies share no state: a Fed failure leaves AVP intact."""
    from app.strategies.set_3_strategies.quantitative_models import (
        asymmetric_volatility_persistence as avp,
    )
    from app.strategies.set_3_strategies.schemas.quantitative import AsymmetricVolatilityParams

    fed_res, _ = run(ey_exc=RuntimeError("provider down"), ty_exc=RuntimeError("provider down"))
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    frame = pd.DataFrame({"price": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))}, index=idx)
    avp_res = avp.compute_asymmetric_volatility_persistence(
        "VOO", frame, AsymmetricVolatilityParams(garch_enabled=False)
    )
    assert fed_res.calculation_status == "unavailable"
    assert avp_res.calculation_status == "ok"
