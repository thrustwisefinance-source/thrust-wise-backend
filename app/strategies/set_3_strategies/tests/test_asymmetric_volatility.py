"""
Tests for the Asymmetric Volatility Persistence strategy
(app.strategies.set_3_strategies.quantitative_models.asymmetric_volatility_persistence).

Pure computation on simulated price frames - no database, Redis or network.
"""

import asyncio
import math
import os

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/test")
os.environ.setdefault("EODHD_API_TOKEN", "test")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.strategies.set_3_strategies.base import StrategyDescriptor, run_isolated  # noqa: E402
from app.strategies.set_3_strategies.quantitative_models import (  # noqa: E402
    asymmetric_volatility_persistence as avp,
)
from app.strategies.set_3_strategies.schemas.quantitative import (  # noqa: E402
    AsymmetricVolatilityParams,
)
from app.strategies.set_3_strategies.services import calculation_service as calc  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def simulate_returns(n=1500, seed=11, omega=0.02, alpha=0.03, beta=0.90, gamma=0.12):
    """Daily % returns from a GJR-GARCH(1,1) process (gamma=0, alpha=beta=0 -> iid)."""
    rng = np.random.default_rng(seed)
    pers = alpha + beta + gamma / 2
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0] = omega / (1 - pers) if pers < 1 else 1.0
    for t in range(n):
        if t > 0:
            s2[t] = omega + (alpha + gamma * (r[t - 1] < 0)) * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = math.sqrt(s2[t]) * rng.standard_normal()
    return r  # percent


def frame_from_returns_pct(r_pct, start="2016-01-04"):
    dates = pd.bdate_range(start, periods=len(r_pct) + 1)
    price = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(np.asarray(r_pct) / 100.0)]))
    return pd.DataFrame({"price": price}, index=dates)


def params(**kw):
    return AsymmetricVolatilityParams(**kw)


@pytest.fixture(scope="module")
def gjr_frame():
    return frame_from_returns_pct(simulate_returns(n=1500, seed=11, gamma=0.14))


@pytest.fixture(scope="module")
def iid_frame():
    rng = np.random.default_rng(5)
    return frame_from_returns_pct(rng.normal(0, 1.0, 1500))


@pytest.fixture(scope="module")
def gjr_result(gjr_frame):
    return avp.compute_asymmetric_volatility_persistence("VOO", gjr_frame, params())


# ---------------------------------------------------------------------------
# 1. Asymmetric volatility calculation
# ---------------------------------------------------------------------------


def test_returns_and_historical_volatility_match_manual_calculation(gjr_frame, gjr_result):
    r = np.log(gjr_frame["price"]).diff().dropna().to_numpy()
    cv = gjr_result.calculated_values
    assert gjr_result.calculation_status == "ok"
    assert cv.returns.observations == len(r)
    assert cv.returns.historical_volatility_annualized_pct == pytest.approx(
        r.std(ddof=1) * math.sqrt(252) * 100, abs=1e-3
    )
    assert cv.returns.daily_volatility_pct == pytest.approx(r.std(ddof=1) * 100, abs=1e-4)
    assert cv.returns.mean_squared_return_pct2 == pytest.approx(np.mean(r**2) * 1e4, abs=1e-4)
    assert cv.rolling_volatility.window == 21
    manual_roll = pd.Series(r).rolling(21).std(ddof=1).iloc[-1] * math.sqrt(252) * 100
    assert cv.rolling_volatility.current_pct == pytest.approx(manual_roll, abs=1e-3)


def test_simple_returns_option_and_annualization_factor(gjr_frame):
    res = avp.compute_asymmetric_volatility_persistence(
        "VOO", gjr_frame, params(return_type="simple", annualization_factor=365)
    )
    r = gjr_frame["price"].pct_change().dropna().to_numpy()
    assert res.calculated_values.returns.return_type == "simple"
    assert res.calculated_values.returns.historical_volatility_annualized_pct == pytest.approx(
        r.std(ddof=1) * math.sqrt(365) * 100, abs=1e-3
    )


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def test_no_trading_signals_in_output(gjr_result):
    """The article gives a framework, not a trading system: no entry / exit /
    buy / sell / hold / signal / recommendation fields may exist."""
    forbidden = {"signal", "signals", "entry", "exit", "buy", "sell", "hold", "action",
                 "recommendation", "position", "trade"}
    for key in _all_keys(gjr_result.model_dump(mode="json")):
        words = set(str(key).lower().split("_"))
        assert not (words & forbidden), key


# ---------------------------------------------------------------------------
# 2. Squared-return autocorrelation
# ---------------------------------------------------------------------------


def test_squared_return_acf_matches_manual(gjr_frame, gjr_result):
    r = np.log(gjr_frame["price"]).diff().dropna().to_numpy()
    r2 = r**2
    xm = r2 - r2.mean()
    manual = [np.sum(xm[:-k] * xm[k:]) / np.sum(xm * xm) for k in (1, 2, 5, 20)]
    acf = {p.lag: p.acf for p in gjr_result.calculated_values.autocorrelation.values}
    for k, m in zip((1, 2, 5, 20), manual, strict=True):
        assert acf[k] == pytest.approx(m, abs=1e-5)
    assert gjr_result.calculated_values.autocorrelation.max_lag == 20
    assert gjr_result.calculated_values.autocorrelation.confidence_band == pytest.approx(
        1.96 / math.sqrt(len(r)), abs=1e-5
    )


def test_clustering_detected_for_garch_but_not_iid(gjr_result, iid_frame):
    assert gjr_result.calculated_values.persistence.clustering == "significant_clustering"
    assert gjr_result.calculated_values.persistence.ljung_box.reject_no_autocorrelation_5pct
    iid = avp.compute_asymmetric_volatility_persistence("VOO", iid_frame, params())
    assert iid.calculated_values.persistence.clustering == "no_significant_clustering"
    assert iid.calculated_values.persistence.ljung_box.p_value > 0.05


@pytest.mark.parametrize(
    "x,df,expected",
    [(3.841458820694124, 1, 0.05), (5.991464547107979, 2, 0.05),
     (18.307038053275146, 10, 0.05), (0.5, 3, 0.91889141)],
)
def test_chi2_survival_function_reference_values(x, df, expected):
    assert calc.chi2_sf(x, df) == pytest.approx(expected, abs=1e-6)


def test_autocorrelation_input_validation():
    with pytest.raises(ValueError):
        calc.autocorrelation(np.ones(50), 5)  # zero variance
    with pytest.raises(ValueError):
        calc.autocorrelation(np.arange(5.0), 10)  # too short
    with pytest.raises(ValueError):
        calc.autocorrelation(np.arange(50.0), 0)


# ---------------------------------------------------------------------------
# 3. Positive versus negative shock analysis
# ---------------------------------------------------------------------------


def test_negative_shocks_followed_by_higher_volatility_when_asymmetric():
    """Strong leverage effect (stationary GJR, persistence 0.995): the GARCH
    leverage LR test drives the conclusion."""
    frame = frame_from_returns_pct(
        simulate_returns(n=2500, seed=3, alpha=0.02, beta=0.85, gamma=0.25)
    )
    res = avp.compute_asymmetric_volatility_persistence("VOO", frame, params())
    a = res.calculated_values.asymmetry
    assert a.negative_shocks.count > 1000 and a.positive_shocks.count > 1000
    assert a.forward_volatility_ratio_negative_over_positive > 1.0
    assert a.conclusion == "negative_shocks_followed_by_higher_volatility"
    assert a.conclusion_method == "garch_lr_test"
    assert "likelihood-ratio" in a.conclusion_basis
    assert res.calculated_values.garch.leverage_test.p_value < 0.01
    reg = a.persistence_regression
    assert reg.persistence_after_negative_shock > reg.persistence_after_non_negative_day
    # classic leverage effect: corr(r_t, r2_t+1) is negative
    assert a.leverage_correlation[0].correlation < 0


def test_regression_fallback_used_when_garch_unavailable_and_is_labelled():
    frame = frame_from_returns_pct(
        simulate_returns(n=2500, seed=3, alpha=0.02, beta=0.85, gamma=0.25)
    )
    for p in (params(garch_enabled=False), params(garch_model="garch")):
        a = avp.compute_asymmetric_volatility_persistence("VOO", frame, p).calculated_values.asymmetry
        assert a.conclusion_method == "persistence_regression"
        assert "low-power" in a.conclusion_basis
        assert a.conclusion in {
            "negative_shocks_followed_by_higher_volatility",
            "no_clear_asymmetry",
        }


def test_symmetric_data_not_flagged_more_often_than_test_size():
    """Over many iid samples the persistence regression must reject at about its
    nominal 5% level (a single seed would be a coin-flip at that rate)."""
    rejections = 0
    n_sims = 100
    for seed in range(n_sims):
        rng = np.random.default_rng(1000 + seed)
        frame = frame_from_returns_pct(rng.normal(0, 1.0, 1000))
        res = avp.compute_asymmetric_volatility_persistence(
            "VOO", frame, params(garch_enabled=False)
        )
        if res.calculated_values.asymmetry.persistence_regression.significant_at_5pct:
            rejections += 1
    assert rejections / n_sims <= 0.15  # nominal 5%; allow Monte-Carlo noise


def test_no_clear_asymmetry_wording_does_not_claim_symmetry():
    rng = np.random.default_rng(1004)  # seed with a non-significant statistic
    frame = frame_from_returns_pct(rng.normal(0, 1.0, 1000))
    a = avp.compute_asymmetric_volatility_persistence(
        "VOO", frame, params(garch_enabled=False)
    ).calculated_values.asymmetry
    if a.conclusion == "no_clear_asymmetry":  # (true for the vast majority of seeds)
        assert "does not prove" in a.conclusion_basis


def test_shock_classification_and_threshold():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 600)
    sa0 = calc.shock_asymmetry(r, horizon=5, threshold_sigma=0.0, annualization=252, max_leverage_lag=3)
    assert sa0.negative.count == int((r < 0).sum())
    assert sa0.positive.count == int((r > 0).sum())
    sa1 = calc.shock_asymmetry(r, horizon=5, threshold_sigma=1.0, annualization=252, max_leverage_lag=3)
    sigma = r.std(ddof=1)
    assert sa1.negative.count == int((r < -sigma).sum())
    assert sa1.positive.count == int((r > sigma).sum())
    assert len(sa1.leverage_correlation) == 3


def test_forward_volatility_uses_only_future_returns():
    # a single huge negative shock at t=100 followed by calm vs. calm baseline
    r = np.full(400, 0.001) * np.where(np.arange(400) % 2 == 0, 1, -1)
    r[100] = -0.10
    sa = calc.shock_asymmetry(r, horizon=3, threshold_sigma=2.0, annualization=252, max_leverage_lag=1)
    # forward window of the shock (t+1..t+3) contains only the calm +/-0.001 returns
    assert sa.negative.count == 1
    assert sa.negative.forward_volatility_annualized_pct == pytest.approx(
        math.sqrt(0.001**2 * 252) * 100, rel=1e-6
    )
    assert sa.negative.sufficient is False  # fewer than 10 shocks flagged


def test_insufficient_shocks_gives_insufficient_data_conclusion():
    rng = np.random.default_rng(2)
    frame = frame_from_returns_pct(rng.normal(0, 1.0, 200))
    res = avp.compute_asymmetric_volatility_persistence(
        "VOO", frame, params(shock_threshold_sigma=3.0, garch_enabled=False)
    )
    assert res.calculated_values.asymmetry.conclusion == "insufficient_data"


# ---------------------------------------------------------------------------
# 4. GARCH availability and failure handling
# ---------------------------------------------------------------------------


def test_garch_estimates_close_to_truth_when_available(gjr_result):
    g = gjr_result.calculated_values.garch
    assert g.status == "available" and g.model == "GJR-GARCH(1,1)"
    assert g.converged is True
    assert 0.0 < g.parameters.alpha < 0.3 and 0.5 < g.parameters.beta < 1.0
    assert g.parameters.gamma > 0.03  # true gamma 0.14
    assert g.persistence == pytest.approx(
        g.parameters.alpha + g.parameters.beta + g.parameters.gamma / 2, abs=1e-5
    )
    assert g.half_life_days and g.half_life_days > 1
    assert g.leverage_test is not None and g.leverage_test.significant_at_5pct
    series = gjr_result.historical_series.garch_conditional_volatility_pct
    assert series[-1] == pytest.approx(g.current_conditional_volatility_annualized_pct, abs=1e-3)


def test_plain_garch_model_has_no_gamma(gjr_frame):
    res = avp.compute_asymmetric_volatility_persistence(
        "VOO", gjr_frame, params(garch_model="garch")
    )
    g = res.calculated_values.garch
    assert g.status == "available" and g.model == "GARCH(1,1)"
    assert g.parameters.gamma is None and g.leverage_test is None


def test_garch_unavailable_with_too_few_observations_reports_message_not_numbers():
    frame = frame_from_returns_pct(simulate_returns(n=200, seed=4))
    res = avp.compute_asymmetric_volatility_persistence("VOO", frame, params())
    g = res.calculated_values.garch
    assert g.status == "unavailable"
    assert "at least 250" in g.message and g.parameters is None and g.persistence is None
    assert res.calculation_status == "partial"  # the rest still computed
    assert res.data_availability.components["garch"] == "unavailable"
    assert all(v is None for v in res.historical_series.garch_conditional_volatility_pct)
    assert any("GARCH" in w for w in res.warnings)


def test_garch_estimation_failure_is_reported_not_fabricated(monkeypatch, gjr_frame):
    def boom(*a, **k):
        raise calc.GarchEstimationError("optimiser did not converge")

    monkeypatch.setattr(calc, "fit_garch", boom)
    res = avp.compute_asymmetric_volatility_persistence("VOO", gjr_frame, params())
    g = res.calculated_values.garch
    assert g.status == "failed" and "did not converge" in g.message
    assert g.parameters is None and g.log_likelihood is None
    assert res.calculated_values.persistence.clustering  # other analytics unaffected
    assert res.calculation_status == "partial"


def test_garch_can_be_disabled(gjr_frame):
    res = avp.compute_asymmetric_volatility_persistence("VOO", gjr_frame, params(garch_enabled=False))
    assert res.calculated_values.garch.status == "disabled"
    assert res.calculation_status == "ok"


def test_fit_garch_rejects_degenerate_input():
    with pytest.raises(calc.GarchEstimationError):
        calc.fit_garch(np.zeros(300))
    with pytest.raises(calc.GarchEstimationError):
        calc.fit_garch(np.arange(10.0))
    with pytest.raises(calc.GarchEstimationError):
        calc.fit_garch(np.append(np.random.default_rng(0).normal(size=300), np.nan))


# ---------------------------------------------------------------------------
# Series, charts, availability, insufficient data
# ---------------------------------------------------------------------------


def test_series_alignment_and_limits(gjr_frame):
    res = avp.compute_asymmetric_volatility_persistence(
        "VOO", gjr_frame, params(series_points=120, chart_max_points=50)
    )
    h = res.historical_series
    assert len(h.dates) == 120
    for arr in (h.returns_pct, h.squared_returns_pct2, h.rolling_volatility_pct,
                h.garch_conditional_volatility_pct):
        assert len(arr) == 120
    assert h.dates == sorted(h.dates)
    assert len(res.chart_data.volatility) <= 50
    assert res.chart_data.volatility[0].date < res.chart_data.volatility[-1].date
    assert len(res.chart_data.autocorrelation) == 20
    assert len(res.chart_data.shock_asymmetry) == 2
    assert res.data_availability.status == "available"
    assert res.data_availability.start_date < res.data_availability.end_date


def test_lookback_days_limits_the_window(gjr_frame):
    res = avp.compute_asymmetric_volatility_persistence(
        "VOO", gjr_frame, params(lookback_days=400, garch_enabled=False)
    )
    assert res.calculated_values.returns.observations < 400
    assert res.calculated_values.returns.observations > 200


def test_insufficient_history_returns_unavailable_without_values():
    frame = frame_from_returns_pct(simulate_returns(n=30, seed=1))
    res = avp.compute_asymmetric_volatility_persistence("VOO", frame, params())
    assert res.calculation_status == "unavailable"
    assert res.calculated_values is None and res.historical_series is None
    assert "Insufficient history" in res.data_availability.message
    assert res.data_availability.observations == 30


def test_empty_and_flat_prices_are_unavailable():
    empty = avp.compute_asymmetric_volatility_persistence("VOO", pd.DataFrame(columns=["price"]), params())
    assert empty.calculation_status == "unavailable" and "No price history" in empty.data_availability.message
    flat = pd.DataFrame({"price": np.full(300, 50.0)}, index=pd.bdate_range("2020-01-01", periods=300))
    res = avp.compute_asymmetric_volatility_persistence("VOO", flat, params())
    assert res.calculation_status == "unavailable" and "zero variance" in res.data_availability.message


def test_nan_and_nonpositive_prices_are_dropped():
    r = simulate_returns(n=400, seed=8)
    frame = frame_from_returns_pct(r)
    frame.iloc[50, 0] = np.nan
    frame.iloc[60, 0] = -1.0
    res = avp.compute_asymmetric_volatility_persistence("VOO", frame, params(garch_enabled=False))
    assert res.calculation_status == "ok"
    assert res.calculated_values.returns.observations <= 400 - 2


# ---------------------------------------------------------------------------
# 9. Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        {"vol_window": 1}, {"vol_window": 500}, {"acf_max_lag": 0}, {"acf_max_lag": 101},
        {"shock_horizon": 0}, {"shock_threshold_sigma": -0.1}, {"shock_threshold_sigma": 5},
        {"annualization_factor": 0}, {"garch_max_observations": 100},
        {"lookback_days": 10}, {"series_points": 5}, {"chart_max_points": 5000},
        {"return_type": "arithmetic"}, {"garch_model": "egarch"},
    ],
)
def test_invalid_parameters_rejected(kw):
    with pytest.raises(ValidationError):
        AsymmetricVolatilityParams(**kw)


def test_default_parameters_and_camel_case_aliases():
    p = AsymmetricVolatilityParams()
    assert (p.vol_window, p.acf_max_lag, p.annualization_factor, p.garch_model) == (21, 20, 252, "gjr")
    assert AsymmetricVolatilityParams(volWindow=10).vol_window == 10


def test_minimum_required_returns_depends_on_parameters():
    assert avp.minimum_returns_required(params()) == 60
    assert avp.minimum_returns_required(params(acf_max_lag=50)) == 150


# ---------------------------------------------------------------------------
# 10. Independent strategy error isolation
# ---------------------------------------------------------------------------

DESC = StrategyDescriptor("x", "X", "quantitative", "sub", "d", "v")


def test_run_isolated_converts_exception_into_error_value():
    async def boom():
        raise RuntimeError("secret internal detail")

    out = asyncio.run(run_isolated(DESC, boom, lambda msg: ("error", msg)))
    assert out[0] == "error"
    assert "secret" not in out[1]  # internals never leak to clients


def test_run_isolated_passes_through_success():
    async def ok():
        return 42

    assert asyncio.run(run_isolated(DESC, ok, lambda m: -1)) == 42


def test_unavailable_result_envelope_shape():
    res = avp.unavailable_result("VOO", params(), "boom", status="error", error="boom")
    assert res.calculation_status == "error" and res.error == "boom"
    assert res.strategy_id == "asymmetric_volatility_persistence"
    assert res.category == "quantitative" and res.subcategory == "Risk Management"
    assert res.calculated_values is None
