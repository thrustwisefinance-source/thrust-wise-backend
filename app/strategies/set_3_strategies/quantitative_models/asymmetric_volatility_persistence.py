"""
Set 3 / Quantitative / Risk Management - Asymmetric Volatility Persistence.

Source: Graham Giller's analysis of volatility persistence and asymmetric
volatility. The article supplies an ANALYTICAL FRAMEWORK, not a trading system:
this module therefore reports statistics only. It deliberately contains no
entry / exit signal, no position rule and no buy / sell / hold output.

Pipeline (all configurable through ``AsymmetricVolatilityParams``)
1. returns (log or simple) from the stored adjusted-close history;
2. squared returns;
3. autocorrelation of squared returns + Ljung-Box test (volatility clustering);
4. persistence indicators: lag-1 / mean ACF, AR(1) coefficient of squared
   returns and its half-life;
5. negative- vs positive-shock analysis: forward realised volatility by shock
   sign, an HC1-robust persistence regression with a negative-shock
   interaction, and the leverage correlation corr(r[t], r2[t+k]);
6. rolling volatility and chart series;
7. GARCH(1,1) / GJR-GARCH(1,1) quasi-ML estimates when enough observations
   exist (numpy-only implementation, no extra dependency); otherwise a clear
   ``unavailable`` / ``failed`` block with the reason - never invented numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.strategies.set_3_strategies.base import (
    CATEGORY_QUANTITATIVE,
    StrategyDescriptor,
)
from app.strategies.set_3_strategies.common import downsample_indices
from app.strategies.set_3_strategies.schemas.quantitative import (
    AVP_STRATEGY_ID,
    AcfChartPoint,
    AcfPoint,
    AsymmetricVolatilityCalculatedValues,
    AsymmetricVolatilityChartData,
    AsymmetricVolatilityHistoricalSeries,
    AsymmetricVolatilityParams,
    AsymmetricVolatilityResult,
    AsymmetryBlock,
    AsymmetryChartBar,
    AutocorrelationBlock,
    DataAvailability,
    GarchBlock,
    GarchParameters,
    Interpretation,
    LeverageLikelihoodRatio,
    LeveragePoint,
    LjungBoxResult,
    PersistenceBlock,
    PersistenceRegression,
    ReturnSummary,
    RollingVolatilitySummary,
    ShockClassStatsModel,
    VolatilityChartPoint,
)
from app.strategies.set_3_strategies.services import calculation_service as calc

CALCULATION_VERSION = "avp-1.0.0"
DESCRIPTOR = StrategyDescriptor(
    strategy_id=AVP_STRATEGY_ID,
    name="Asymmetric Volatility Persistence",
    category=CATEGORY_QUANTITATIVE,
    subcategory="Risk Management",
    description=(
        "Measures volatility clustering (autocorrelation of squared returns), how "
        "persistent it is, and whether negative shocks are followed by stronger "
        "volatility than positive shocks, with optional GARCH / GJR-GARCH "
        "estimates. An analytical framework only - it produces no trading signals."
    ),
    calculation_version=CALCULATION_VERSION,
)

GARCH_MIN_OBSERVATIONS = 250
SIGNIFICANCE_T = 1.96
DISCLAIMER_NOTE = (
    "Statistics describe the stored price history; they do not predict returns and "
    "are not entry or exit signals."
)


def _r(x: float | None, digits: int = 4) -> float | None:
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


def _rf(x: float, digits: int = 4) -> float:
    v = _r(x, digits)
    return 0.0 if v is None else v


NO_PROOF = (
    " 'Not significant' does not prove the response is symmetric: these tests have "
    "limited power on a single price history."
)


def _asymmetry_conclusion(
    sa: calc.ShockAsymmetry, reg_sig: bool | None, garch: GarchBlock
) -> tuple[str, str, str]:
    """(conclusion, method, basis).

    Primary evidence: the GJR-GARCH leverage likelihood-ratio test, which is far
    more powerful than a regression on squared returns, whose heavy-tailed
    regressors give it low power. When GARCH is not available (too little data,
    disabled, symmetric model requested, estimation failure) the HC1 persistence
    regression is used instead, and the basis says so.
    """
    lr = garch.leverage_test
    if garch.status == "available" and lr is not None and garch.parameters is not None \
            and garch.parameters.gamma is not None:
        gamma = garch.parameters.gamma
        text = (
            f"GJR-GARCH leverage likelihood-ratio test: chi2(1) = {lr.statistic:.2f}, "
            f"p = {lr.p_value:.4f}, gamma = {gamma:.4f}."
        )
        if lr.significant_at_5pct and gamma > 0:
            return "negative_shocks_followed_by_higher_volatility", "garch_lr_test", (
                text + " Negative shocks raise next-period variance significantly more than "
                "positive shocks of the same size."
            )
        if lr.significant_at_5pct and gamma < 0:
            return "positive_shocks_followed_by_higher_volatility", "garch_lr_test", (
                text + " Positive shocks raise next-period variance significantly more than "
                "negative shocks of the same size."
            )
        return "no_clear_asymmetry", "garch_lr_test", (
            text + " The asymmetry term is not significant at the 5% level." + NO_PROOF
        )

    prefix = (
        "GARCH leverage test unavailable; using the (low-power) persistence regression. "
    )
    if reg_sig is None or not (sa.negative.sufficient and sa.positive.sufficient):
        return "insufficient_data", "persistence_regression", (
            prefix + "Fewer than 10 shocks of one sign, or the regression could not be estimated."
        )
    t = sa.persistence_difference_t
    diff = sa.persistence_difference or 0.0
    if reg_sig and diff > 0:
        return "negative_shocks_followed_by_higher_volatility", "persistence_regression", (
            prefix + "Squared returns are significantly more persistent after negative "
            f"shocks (HC1 t = {t:.2f}, |t| >= {SIGNIFICANCE_T})."
        )
    if reg_sig and diff < 0:
        return "positive_shocks_followed_by_higher_volatility", "persistence_regression", (
            prefix + "Squared returns are significantly more persistent after non-negative "
            f"days (HC1 t = {t:.2f}, |t| >= {SIGNIFICANCE_T})."
        )
    return "no_clear_asymmetry", "persistence_regression", (
        prefix + "The persistence difference is not significant at the 5% level "
        f"(HC1 t = {t:.2f})." + NO_PROOF
    )


def minimum_returns_required(params: AsymmetricVolatilityParams) -> int:
    return max(60, 3 * params.acf_max_lag, params.vol_window + 1, params.shock_horizon + 30)


def unavailable_result(
    symbol: str,
    params: AsymmetricVolatilityParams,
    message: str,
    *,
    status: str = "unavailable",
    observations: int | None = None,
    error: str | None = None,
) -> AsymmetricVolatilityResult:
    """Result envelope carrying no numbers (insufficient data / failure)."""
    return AsymmetricVolatilityResult(
        strategy_id=DESCRIPTOR.strategy_id,
        strategy_name=DESCRIPTOR.name,
        subcategory=DESCRIPTOR.subcategory,
        symbol=symbol,
        description=DESCRIPTOR.description,
        calculation_version=CALCULATION_VERSION,
        parameters=params,
        data_availability=DataAvailability(
            status="unavailable", message=message, observations=observations
        ),
        calculation_status=status,  # type: ignore[arg-type]
        error=error if error is not None else (message if status == "error" else None),
    )


# ---------------------------------------------------------------------------
# GARCH block
# ---------------------------------------------------------------------------


def _garch_block(
    returns: pd.Series, params: AsymmetricVolatilityParams
) -> tuple[GarchBlock, np.ndarray | None]:
    """Returns (block, conditional-variance path aligned to the LAST n obs)."""
    if not params.garch_enabled:
        return GarchBlock(status="disabled", message="GARCH analysis disabled by request."), None
    window = returns.iloc[-params.garch_max_observations:]
    n = len(window)
    if n < GARCH_MIN_OBSERVATIONS:
        return (
            GarchBlock(
                status="unavailable",
                message=(
                    f"GARCH estimation needs at least {GARCH_MIN_OBSERVATIONS} return "
                    f"observations; only {n} are available."
                ),
                observations_used=n,
            ),
            None,
        )
    x_pct = window.to_numpy(dtype=float) * 100.0
    try:
        fit = calc.fit_garch(x_pct, params.garch_model)
    except calc.GarchEstimationError as exc:
        return (
            GarchBlock(
                status="failed",
                message=f"GARCH estimation failed: {exc}",
                model=params.garch_model,
                observations_used=n,
            ),
            None,
        )
    af = params.annualization_factor
    cur_vol = math.sqrt(float(fit.conditional_variance[-1]) * af)
    unc_vol = math.sqrt(fit.unconditional_variance * af)
    lr = None
    if params.garch_model == "gjr":
        try:
            base_fit = calc.fit_garch(x_pct, "garch")
            stat = max(0.0, 2.0 * (fit.log_likelihood - base_fit.log_likelihood))
            p = calc.chi2_sf(stat, 1)
            lr = LeverageLikelihoodRatio(
                statistic=_rf(stat), p_value=_rf(p, 6), significant_at_5pct=bool(p < 0.05)
            )
        except calc.GarchEstimationError:
            lr = None  # leverage test unavailable; the GJR estimates stay valid
    label = "GJR-GARCH(1,1)" if params.garch_model == "gjr" else "GARCH(1,1)"
    block = GarchBlock(
        status="available",
        model=label,
        estimation_method=(
            "Gaussian quasi-maximum likelihood, constant mean = sample mean, "
            "variance targeting (omega implied by the sample variance); "
            "Nelder-Mead optimiser; no standard errors reported"
        ),
        observations_used=n,
        parameters=GarchParameters(
            omega=_rf(fit.omega, 6),
            alpha=_rf(fit.alpha, 6),
            beta=_rf(fit.beta, 6),
            gamma=_r(fit.gamma, 6) if params.garch_model == "gjr" else None,
        ),
        persistence=_r(fit.persistence, 6),
        half_life_days=_r(fit.half_life, 2),
        unconditional_volatility_annualized_pct=_r(unc_vol, 4),
        current_conditional_volatility_annualized_pct=_r(cur_vol, 4),
        log_likelihood=_r(fit.log_likelihood, 3),
        aic=_r(fit.aic, 3),
        bic=_r(fit.bic, 3),
        converged=fit.converged,
        leverage_test=lr,
    )
    return block, fit.conditional_variance


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_asymmetric_volatility_persistence(
    symbol: str, prices: pd.DataFrame, params: AsymmetricVolatilityParams
) -> AsymmetricVolatilityResult:
    """Pure, synchronous computation from a price frame (column ``price``)."""
    if prices is None or prices.empty or "price" not in prices.columns:
        return unavailable_result(
            symbol, params, f"No price history is available for {symbol}.", observations=0
        )

    price = prices["price"].astype(float)
    if params.lookback_days is not None:
        cutoff = price.index.max() - pd.Timedelta(days=params.lookback_days)
        price = price[price.index >= cutoff]

    returns = calc.compute_returns(price, params.return_type)
    n = len(returns)
    need = minimum_returns_required(params)
    if n < need:
        return unavailable_result(
            symbol,
            params,
            f"Insufficient history: {n} return observations available, at least "
            f"{need} required for the selected parameters.",
            observations=n,
        )

    r = returns.to_numpy(dtype=float)
    af = params.annualization_factor
    warnings: list[str] = []
    sigma_daily = float(r.std(ddof=1))
    if sigma_daily <= 0:
        return unavailable_result(
            symbol, params, "Return series has zero variance; analysis not possible.",
            observations=n,
        )
    r2 = r**2

    # ---- historical + rolling volatility
    rolling = returns.rolling(params.vol_window).std(ddof=1) * math.sqrt(af) * 100.0
    roll_valid = rolling.dropna()
    cur_roll = float(roll_valid.iloc[-1]) if len(roll_valid) else None
    pct_rank = (
        float((roll_valid <= cur_roll).mean() * 100.0) if cur_roll is not None else None
    )
    returns_summary = ReturnSummary(
        return_type=params.return_type,
        observations=n,
        start_date=returns.index[0].date().isoformat(),
        end_date=returns.index[-1].date().isoformat(),
        mean_daily_return_pct=_rf(float(r.mean()) * 100.0, 5),
        daily_volatility_pct=_rf(sigma_daily * 100.0, 5),
        historical_volatility_annualized_pct=_rf(sigma_daily * math.sqrt(af) * 100.0),
        annualization_factor=af,
        mean_squared_return_pct2=_rf(float(r2.mean()) * 1e4, 6),
        max_squared_return_pct2=_rf(float(r2.max()) * 1e4, 6),
    )
    rolling_summary = RollingVolatilitySummary(
        window=params.vol_window,
        current_pct=_r(cur_roll),
        mean_pct=_r(float(roll_valid.mean())) if len(roll_valid) else None,
        min_pct=_r(float(roll_valid.min())) if len(roll_valid) else None,
        max_pct=_r(float(roll_valid.max())) if len(roll_valid) else None,
        current_percentile=_r(pct_rank, 2),
    )

    # ---- autocorrelation of squared returns
    acf = calc.autocorrelation(r2, params.acf_max_lag)
    band = 1.96 / math.sqrt(n)
    acf_block = AutocorrelationBlock(
        max_lag=params.acf_max_lag,
        confidence_band=_rf(band, 6),
        values=[
            AcfPoint(lag=i + 1, acf=_rf(v, 6), significant=bool(abs(v) > band))
            for i, v in enumerate(acf)
        ],
    )

    # ---- persistence
    lb = calc.ljung_box(acf, n)
    ar1 = ar1_half = None
    try:
        res = calc.ols_hc1(
            (r2 * 1e4)[1:], np.column_stack([np.ones(n - 1), (r2 * 1e4)[:-1]])
        )
        ar1 = float(res.coef[1])
        ar1_half = (math.log(0.5) / math.log(ar1)) if 0 < ar1 < 1 else None
    except ValueError:
        warnings.append("AR(1) persistence regression could not be estimated.")
    persistence = PersistenceBlock(
        lag1_acf=_rf(acf[0], 6),
        mean_acf=_rf(float(acf.mean()), 6),
        significant_lags=int(np.sum(np.abs(acf) > band)),
        ljung_box=LjungBoxResult(
            lags=lb.lags,
            statistic=_rf(lb.statistic, 4),
            p_value=_rf(lb.p_value, 6),
            reject_no_autocorrelation_5pct=bool(lb.p_value < 0.05),
        ),
        ar1_coefficient=_r(ar1, 6),
        ar1_half_life_days=_r(ar1_half, 2),
        clustering=(
            "significant_clustering" if lb.p_value < 0.05 else "no_significant_clustering"
        ),
    )

    # ---- GARCH
    garch_block, cond_var = _garch_block(returns, params)
    if garch_block.status in ("failed", "unavailable"):
        warnings.append(garch_block.message or "GARCH analysis unavailable.")

    # ---- shock asymmetry
    sa = calc.shock_asymmetry(
        r,
        horizon=params.shock_horizon,
        threshold_sigma=params.shock_threshold_sigma,
        annualization=af,
        max_leverage_lag=params.acf_max_lag,
    )

    def cls(s: calc.ShockClassStats) -> ShockClassStatsModel:
        return ShockClassStatsModel(
            count=s.count,
            mean_return_pct=_r(s.mean_return_pct, 4),
            next_day_volatility_annualized_pct=_r(s.next_day_volatility_annualized_pct),
            forward_volatility_annualized_pct=_r(s.forward_volatility_annualized_pct),
            sufficient_observations=s.sufficient,
        )

    sig = (
        None
        if sa.persistence_difference_t is None
        else bool(abs(sa.persistence_difference_t) >= SIGNIFICANCE_T)
    )
    conclusion, method, basis = _asymmetry_conclusion(sa, sig, garch_block)
    asym = AsymmetryBlock(
        shock_threshold_sigma=params.shock_threshold_sigma,
        horizon_days=params.shock_horizon,
        negative_shocks=cls(sa.negative),
        positive_shocks=cls(sa.positive),
        forward_volatility_ratio_negative_over_positive=_r(sa.forward_vol_ratio),
        next_day_volatility_ratio_negative_over_positive=_r(sa.next_day_vol_ratio),
        persistence_regression=PersistenceRegression(
            persistence_after_non_negative_day=_r(sa.persistence_non_negative, 6),
            persistence_after_negative_shock=_r(sa.persistence_negative, 6),
            difference=_r(sa.persistence_difference, 6),
            difference_t_statistic=_r(sa.persistence_difference_t, 4),
            significant_at_5pct=sig,
            observations=sa.regression_n,
        ),
        leverage_correlation=[LeveragePoint(lag=k, correlation=_r(c, 6)) for k, c in sa.leverage_correlation],
        conclusion=conclusion,  # type: ignore[arg-type]
        conclusion_method=method,  # type: ignore[arg-type]
        conclusion_basis=basis,
    )

    # ---- series
    garch_vol = pd.Series(np.nan, index=returns.index)
    if cond_var is not None:
        garch_vol.iloc[-len(cond_var):] = np.sqrt(cond_var * af)

    k = min(params.series_points, n)
    tail = slice(n - k, n)

    def vals(arr: np.ndarray | pd.Series, scale: float = 1.0, digits: int = 4) -> list[float | None]:
        a = np.asarray(arr, dtype=float)[tail] * scale
        return [None if not math.isfinite(v) else round(float(v), digits) for v in a]

    hist = AsymmetricVolatilityHistoricalSeries(
        dates=[d.date().isoformat() for d in returns.index[tail]],
        returns_pct=vals(r, 100.0, 5),
        squared_returns_pct2=vals(r2, 1e4, 6),
        rolling_volatility_pct=vals(rolling, 1.0, 4),
        garch_conditional_volatility_pct=vals(garch_vol, 1.0, 4),
    )

    idx = downsample_indices(n, params.chart_max_points)
    vol_chart = [
        VolatilityChartPoint(
            date=returns.index[i].date().isoformat(),
            rolling_volatility_pct=_r(rolling.iloc[i]) if math.isfinite(rolling.iloc[i]) else None,
            garch_volatility_pct=_r(garch_vol.iloc[i]) if math.isfinite(garch_vol.iloc[i]) else None,
        )
        for i in idx
    ]
    chart = AsymmetricVolatilityChartData(
        volatility=vol_chart,
        autocorrelation=[
            AcfChartPoint(lag=p.lag, acf=p.acf, upper_band=_rf(band, 6), lower_band=_rf(-band, 6))
            for p in acf_block.values
        ],
        shock_asymmetry=[
            AsymmetryChartBar(
                label="After negative shocks",
                count=asym.negative_shocks.count,
                forward_volatility_annualized_pct=asym.negative_shocks.forward_volatility_annualized_pct,
            ),
            AsymmetryChartBar(
                label="After positive shocks",
                count=asym.positive_shocks.count,
                forward_volatility_annualized_pct=asym.positive_shocks.forward_volatility_annualized_pct,
            ),
        ],
        leverage_correlation=asym.leverage_correlation,
    )

    calculated = AsymmetricVolatilityCalculatedValues(
        returns=returns_summary,
        rolling_volatility=rolling_summary,
        autocorrelation=acf_block,
        persistence=persistence,
        asymmetry=asym,
        garch=garch_block,
    )

    # ---- interpretation (descriptive only)
    points = [
        (
            f"Squared returns {'show' if lb.p_value < 0.05 else 'do not show'} statistically "
            f"significant autocorrelation (Ljung-Box Q({lb.lags}) = {lb.statistic:.1f}, "
            f"p = {lb.p_value:.4f}); lag-1 ACF = {acf[0]:.3f}."
        ),
        asym.conclusion_basis,
    ]
    if ar1 is not None:
        hl = f", half-life about {ar1_half:.1f} days" if ar1_half else ""
        points.append(f"AR(1) coefficient of squared returns = {ar1:.3f}{hl}.")
    if garch_block.status == "available" and garch_block.persistence is not None:
        g_txt = (
            f" Leverage (gamma) estimate = {garch_block.parameters.gamma:.3f}"
            if garch_block.parameters and garch_block.parameters.gamma is not None
            else ""
        )
        points.append(
            f"{garch_block.model} persistence (alpha+beta{'+gamma/2' if g_txt else ''}) = "
            f"{garch_block.persistence:.3f}.{g_txt}"
        )
    elif garch_block.message:
        points.append(garch_block.message)
    interpretation = Interpretation(
        summary=(
            "Volatility clustering is "
            + ("statistically significant" if lb.p_value < 0.05 else "not statistically significant")
            + f"; asymmetry assessment: {conclusion.replace('_', ' ')}."
        ),
        points=points,
        classification=conclusion,
    )

    availability = DataAvailability(
        status="available" if garch_block.status in ("available", "disabled") else "partial",
        observations=n,
        start_date=returns_summary.start_date,
        end_date=returns_summary.end_date,
        message=None if garch_block.status in ("available", "disabled") else garch_block.message,
        components={
            "returns": "available",
            "autocorrelation": "available",
            "shock_analysis": "available" if conclusion != "insufficient_data" else "partial",
            "garch": garch_block.status,
        },
    )
    return AsymmetricVolatilityResult(
        strategy_id=DESCRIPTOR.strategy_id,
        strategy_name=DESCRIPTOR.name,
        subcategory=DESCRIPTOR.subcategory,
        symbol=symbol,
        description=DESCRIPTOR.description,
        calculation_version=CALCULATION_VERSION,
        parameters=params,
        calculated_values=calculated,
        historical_series=hist,
        chart_data=chart,
        interpretation=interpretation,
        data_availability=availability,
        calculation_status="ok" if availability.status == "available" else "partial",
        warnings=warnings,
        notes=[
            DISCLAIMER_NOTE,
            "Returns use the stored adjusted close; volatility is annualised with "
            f"sqrt({af}). Missing trading days make the adjacent return span the gap.",
            "ACF significance band = +/-1.96/sqrt(n) (white-noise approximation).",
        ],
    )
