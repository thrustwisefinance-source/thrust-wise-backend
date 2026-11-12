"""
Numerical building blocks for the Set 3 quantitative strategies.

Pure functions only: numpy / pandas / math - no I/O, no database, no network
and no dependency beyond what the project already installs (scipy and
statsmodels / arch are deliberately NOT required).

Contents
--------
* ``compute_returns``      simple / log returns from a price series
* ``autocorrelation``      sample autocorrelation function (lags 1..K)
* ``chi2_sf``              chi-square survival function (no scipy)
* ``ljung_box``            Ljung-Box portmanteau test
* ``ols_hc1``              OLS with heteroskedasticity-robust (HC1) errors
* ``shock_asymmetry``      negative- vs positive-shock volatility analysis
* ``fit_garch``            GARCH(1,1) / GJR-GARCH(1,1) quasi-ML estimation
                           (variance targeting, Nelder-Mead, numpy only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def compute_returns(
    prices: pd.Series, kind: Literal["log", "simple"] = "log"
) -> pd.Series:
    """Per-period returns as DECIMAL fractions (0.01 = 1 %), NaN rows dropped.

    Non-positive prices are removed first (a log of them is undefined). A
    missing trading day simply makes the neighbouring return span the gap;
    no value is interpolated.
    """
    p = pd.to_numeric(prices, errors="coerce").astype(float)
    p = p[np.isfinite(p) & (p > 0)]
    if kind == "log":
        r = np.log(p).diff()
    else:
        r = p.pct_change()
    r = r.replace([np.inf, -np.inf], np.nan).dropna()
    return r


# ---------------------------------------------------------------------------
# Autocorrelation / Ljung-Box
# ---------------------------------------------------------------------------


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (length ``max_lag``)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if n <= max_lag + 1:
        raise ValueError("series too short for the requested number of lags")
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom <= 0 or not math.isfinite(denom):
        raise ValueError("series has zero variance")
    return np.array(
        [float(np.dot(xm[:-k], xm[k:])) / denom for k in range(1, max_lag + 1)]
    )


def _gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (Numerical-Recipes style)."""
    if x <= 0:
        return 1.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:  # series for P(a, x)
        ap, term = a, 1.0 / a
        total = term
        for _ in range(2000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return max(0.0, min(1.0, 1.0 - total * math.exp(log_prefactor)))
    # continued fraction for Q(a, x) (modified Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return max(0.0, min(1.0, h * math.exp(log_prefactor)))


def chi2_sf(x: float, df: int) -> float:
    """P(Chi2_df > x)."""
    if df < 1:
        raise ValueError("df must be >= 1")
    if x <= 0:
        return 1.0
    return _gammaincc(df / 2.0, x / 2.0)


@dataclass(frozen=True)
class LjungBox:
    lags: int
    statistic: float
    p_value: float


def ljung_box(acf: np.ndarray, n: int) -> LjungBox:
    """Ljung-Box Q over every lag in ``acf`` (lags 1..len(acf))."""
    m = len(acf)
    ks = np.arange(1, m + 1)
    q = float(n * (n + 2) * np.sum(acf**2 / (n - ks)))
    return LjungBox(lags=m, statistic=q, p_value=chi2_sf(q, m))


# ---------------------------------------------------------------------------
# OLS with HC1 robust errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OlsResult:
    coef: np.ndarray
    std_err: np.ndarray
    t_stat: np.ndarray
    r_squared: float
    n: int


def ols_hc1(y: np.ndarray, X: np.ndarray) -> OlsResult:
    """OLS with White/HC1 heteroskedasticity-robust standard errors.

    Robust errors matter here: the regressors/targets are squared returns,
    whose error variance is far from constant.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n <= k:
        raise ValueError("not enough observations for the regression")
    xtx = X.T @ X
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("regression design matrix is singular") from exc
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    meat = (X * (resid**2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv * (n / (n - k))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")
    return OlsResult(beta, se, t, r2, n)


# ---------------------------------------------------------------------------
# Negative vs positive shock analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShockClassStats:
    count: int
    mean_return_pct: float | None
    next_day_volatility_annualized_pct: float | None
    forward_volatility_annualized_pct: float | None
    sufficient: bool


@dataclass(frozen=True)
class ShockAsymmetry:
    threshold_sigma: float
    horizon: int
    negative: ShockClassStats
    positive: ShockClassStats
    forward_vol_ratio: float | None
    next_day_vol_ratio: float | None
    persistence_non_negative: float | None
    persistence_negative: float | None
    persistence_difference: float | None
    persistence_difference_t: float | None
    regression_n: int
    leverage_correlation: list[tuple[int, float | None]] = field(default_factory=list)


MIN_SHOCKS_PER_CLASS = 10


def _class_stats(
    mask: np.ndarray,
    r: np.ndarray,
    r2: np.ndarray,
    fwd_var: np.ndarray,
    horizon: int,
    annualization: float,
) -> ShockClassStats:
    n = len(r)
    count = int(mask.sum())
    if count == 0:
        return ShockClassStats(0, None, None, None, False)
    mean_ret = float(r[mask].mean()) * 100.0
    nxt_mask = mask[: n - 1]
    nxt = r2[1:][nxt_mask]
    fwd_mask = mask[: n - horizon]
    fwd = fwd_var[fwd_mask]
    enough = nxt.size >= MIN_SHOCKS_PER_CLASS and fwd.size >= MIN_SHOCKS_PER_CLASS
    nxt_vol = math.sqrt(float(nxt.mean()) * annualization) * 100.0 if nxt.size else None
    fwd_vol = math.sqrt(float(fwd.mean()) * annualization) * 100.0 if fwd.size else None
    return ShockClassStats(count, mean_ret, nxt_vol, fwd_vol, bool(enough))


def shock_asymmetry(
    returns: np.ndarray,
    *,
    horizon: int,
    threshold_sigma: float,
    annualization: float,
    max_leverage_lag: int,
) -> ShockAsymmetry:
    """
    Compare volatility following NEGATIVE shocks with volatility following
    POSITIVE shocks.

    A shock is a return below ``-threshold_sigma * sigma`` (negative) or above
    ``+threshold_sigma * sigma`` (positive); with ``threshold_sigma == 0`` it is
    simply the sign of the return (zero returns belong to neither class).

    * forward volatility = sqrt(mean(r^2 over the next ``horizon`` days)),
      annualised, averaged over all shocks of the class;
    * persistence regression (squared returns in %^2, HC1 errors):
          r2[t+1] = a + b * r2[t] + c * r2[t] * 1[negative shock at t] + e
      ``b`` is the persistence after a non-negative day, ``b + c`` after a
      negative shock; ``c`` > 0 means volatility is MORE persistent after bad
      news;
    * leverage correlation corr(r[t], r2[t+k]) for k = 1..max_leverage_lag
      (a negative value means falls tend to precede higher squared returns).
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if horizon < 1 or n <= horizon + 2:
        raise ValueError("series too short for the requested shock horizon")
    sigma = float(r.std(ddof=1))
    thr = threshold_sigma * sigma
    if thr > 0:
        neg, pos = r < -thr, r > thr
    else:
        neg, pos = r < 0, r > 0
    r2 = r**2
    cs = np.concatenate([[0.0], np.cumsum(r2)])
    # forward realised variance over days t+1..t+horizon, defined for t <= n-1-horizon
    t_idx = np.arange(0, n - horizon)
    fwd_var = (cs[t_idx + 1 + horizon] - cs[t_idx + 1]) / horizon

    neg_stats = _class_stats(neg, r, r2, fwd_var, horizon, annualization)
    pos_stats = _class_stats(pos, r, r2, fwd_var, horizon, annualization)

    def ratio(a: float | None, b: float | None) -> float | None:
        return a / b if a is not None and b not in (None, 0) else None

    fwd_ratio = ratio(neg_stats.forward_volatility_annualized_pct,
                      pos_stats.forward_volatility_annualized_pct)
    nxt_ratio = ratio(neg_stats.next_day_volatility_annualized_pct,
                      pos_stats.next_day_volatility_annualized_pct)

    b_non_neg = b_neg = diff = diff_t = None
    reg_n = 0
    r_pct2 = (r * 100.0) ** 2
    y = r_pct2[1:]
    x = r_pct2[:-1]
    dummy = neg[:-1].astype(float)
    try:
        res = ols_hc1(y, np.column_stack([np.ones_like(x), x, x * dummy]))
        b_non_neg = float(res.coef[1])
        b_neg = float(res.coef[1] + res.coef[2])
        diff = float(res.coef[2])
        diff_t = float(res.t_stat[2]) if np.isfinite(res.t_stat[2]) else None
        reg_n = res.n
    except ValueError:
        pass

    lev: list[tuple[int, float | None]] = []
    for k in range(1, max_leverage_lag + 1):
        if n - k < 10:
            lev.append((k, None))
            continue
        a, b = r[:-k], r2[k:]
        if a.std() == 0 or b.std() == 0:
            lev.append((k, None))
        else:
            lev.append((k, float(np.corrcoef(a, b)[0, 1])))

    return ShockAsymmetry(
        threshold_sigma=threshold_sigma,
        horizon=horizon,
        negative=neg_stats,
        positive=pos_stats,
        forward_vol_ratio=fwd_ratio,
        next_day_vol_ratio=nxt_ratio,
        persistence_non_negative=b_non_neg,
        persistence_negative=b_neg,
        persistence_difference=diff,
        persistence_difference_t=diff_t,
        regression_n=reg_n,
        leverage_correlation=lev,
    )


# ---------------------------------------------------------------------------
# GARCH(1,1) / GJR-GARCH(1,1) by variance-targeting quasi-maximum-likelihood
# ---------------------------------------------------------------------------

GarchKind = Literal["garch", "gjr"]


class GarchEstimationError(Exception):
    """The model could not be estimated reliably (message is user-facing)."""


@dataclass(frozen=True)
class GarchFit:
    model: GarchKind
    omega: float  # in (percent return)^2 units
    alpha: float
    beta: float
    gamma: float  # 0.0 for plain GARCH
    persistence: float  # alpha + beta + gamma / 2
    log_likelihood: float
    aic: float
    bic: float
    n_obs: int
    converged: bool
    iterations: int
    mean_pct: float
    unconditional_variance: float  # (percent return)^2, daily
    conditional_variance: np.ndarray  # (percent return)^2, daily, aligned to input

    @property
    def n_params(self) -> int:
        return 3 if self.model == "gjr" else 2

    @property
    def half_life(self) -> float | None:
        if 0 < self.persistence < 1:
            return math.log(0.5) / math.log(self.persistence)
        return None


def _garch_variance_path(
    eps: np.ndarray, neg: np.ndarray, omega: float, alpha: float, beta: float,
    gamma: float, sigma2_0: float,
) -> np.ndarray:
    n = len(eps)
    sig2 = np.empty(n)
    sig2[0] = sigma2_0
    e2 = (eps * eps).tolist()
    ng = neg.tolist()
    prev = sigma2_0
    out = [sigma2_0]
    for t in range(1, n):
        prev = omega + (alpha + gamma * ng[t - 1]) * e2[t - 1] + beta * prev
        out.append(prev)
    sig2[:] = out
    return sig2


def _neg_loglik(
    theta: np.ndarray, eps: np.ndarray, neg: np.ndarray, var: float, kind: GarchKind
) -> float:
    alpha, beta = float(theta[0]), float(theta[1])
    gamma = float(theta[2]) if kind == "gjr" else 0.0
    pers = alpha + beta + gamma / 2.0
    # feasibility: positivity of the conditional variance and stationarity
    if alpha < 0 or beta < 0 or alpha + gamma < 0 or pers >= 0.9999 or alpha > 1 or beta > 1:
        return 1e12
    omega = var * (1.0 - pers)
    if omega <= 0:
        return 1e12
    sig2 = _garch_variance_path(eps, neg, omega, alpha, beta, gamma, var)
    if not np.all(np.isfinite(sig2)) or np.any(sig2 <= 0):
        return 1e12
    ll = -0.5 * (math.log(2 * math.pi) + np.log(sig2) + eps * eps / sig2)
    val = -float(ll.sum())
    return val if math.isfinite(val) else 1e12


def _nelder_mead(
    f, x0: np.ndarray, step: np.ndarray, max_iter: int = 400, tol: float = 1e-8
) -> tuple[np.ndarray, float, int, bool]:
    """Minimal Nelder-Mead simplex minimiser (adaptive-free, standard coefficients)."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        p = np.array(x0, dtype=float)
        p[i] += step[i]
        simplex.append(p)
    vals = [f(p) for p in simplex]
    it = 0
    converged = False
    while it < max_iter:
        it += 1
        order = np.argsort(vals)
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol * (1.0 + abs(vals[0])) and \
                max(np.max(np.abs(p - simplex[0])) for p in simplex[1:]) < 1e-6:
            converged = True
            break
        centroid = np.mean(simplex[:-1], axis=0)
        worst = simplex[-1]
        xr = centroid + (centroid - worst)
        fr = f(xr)
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - worst)
            fe = f(xe)
            if fe < fr:
                simplex[-1], vals[-1] = xe, fe
            else:
                simplex[-1], vals[-1] = xr, fr
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = xr, fr
        else:
            if fr < vals[-1]:
                xc = centroid + 0.5 * (xr - centroid)
            else:
                xc = centroid + 0.5 * (worst - centroid)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                simplex[-1], vals[-1] = xc, fc
            else:  # shrink
                best = simplex[0]
                simplex = [best] + [best + 0.5 * (p - best) for p in simplex[1:]]
                vals = [vals[0]] + [f(p) for p in simplex[1:]]
    order = int(np.argmin(vals))
    return simplex[order], float(vals[order]), it, converged


def fit_garch(returns_pct: np.ndarray, model: GarchKind = "gjr") -> GarchFit:
    """
    Gaussian quasi-ML estimate of GARCH(1,1) or GJR-GARCH(1,1) on PERCENT
    returns with a constant mean (the sample mean) and variance targeting
    (omega = sample_variance * (1 - persistence)), which leaves 2 (GARCH) or 3
    (GJR) free parameters.

        sigma2[t] = omega + (alpha + gamma * 1[eps[t-1] < 0]) * eps[t-1]^2
                    + beta * sigma2[t-1]

    Raises ``GarchEstimationError`` when the estimate is not trustworthy
    (non-finite likelihood, optimiser did not converge, degenerate series).
    """
    x = np.asarray(returns_pct, dtype=float)
    if x.ndim != 1 or len(x) < 50:
        raise GarchEstimationError("GARCH needs at least 50 observations")
    if not np.all(np.isfinite(x)):
        raise GarchEstimationError("returns contain non-finite values")
    mean = float(x.mean())
    eps = x - mean
    var = float(eps.var())
    if var <= 0:
        raise GarchEstimationError("return series has zero variance")
    neg = (eps < 0).astype(float)

    def obj(theta: np.ndarray) -> float:
        return _neg_loglik(theta, eps, neg, var, model)

    if model == "gjr":
        starts = [np.array([0.03, 0.90, 0.08]), np.array([0.08, 0.80, 0.10]),
                  np.array([0.02, 0.95, 0.02])]
        steps = np.array([0.02, 0.03, 0.03])
    else:
        starts = [np.array([0.08, 0.90]), np.array([0.15, 0.75]), np.array([0.04, 0.95])]
        steps = np.array([0.03, 0.03])

    best = None
    for s in starts:
        if obj(s) >= 1e11:
            continue
        theta, val, iters, conv = _nelder_mead(obj, s, steps)
        if best is None or val < best[1]:
            best = (theta, val, iters, conv)
    if best is None or best[1] >= 1e11:
        raise GarchEstimationError("no feasible GARCH parameters found")
    theta, nll, iters, conv = best
    alpha, beta = float(theta[0]), float(theta[1])
    gamma = float(theta[2]) if model == "gjr" else 0.0
    pers = alpha + beta + gamma / 2.0
    omega = var * (1.0 - pers)
    sig2 = _garch_variance_path(eps, neg, omega, alpha, beta, gamma, var)
    ll = -nll
    k = 3 if model == "gjr" else 2
    n = len(x)
    if not conv:
        raise GarchEstimationError(
            "GARCH optimisation did not converge; no estimates are reported"
        )
    return GarchFit(
        model=model, omega=omega, alpha=alpha, beta=beta, gamma=gamma,
        persistence=pers, log_likelihood=ll, aic=2 * k - 2 * ll,
        bic=k * math.log(n) - 2 * ll, n_obs=n, converged=conv, iterations=iters,
        mean_pct=mean, unconditional_variance=var, conditional_variance=sig2,
    )
