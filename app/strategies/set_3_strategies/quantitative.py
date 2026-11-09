"""
Set 3 - QUANTITATIVE strategies.

Statistical, mathematical, stochastic and quantitative investment models.
Register every new quantitative strategy in ``QUANTITATIVE_STRATEGIES`` at the
bottom of this file (and add its field to ``QuantitativeSet3Strategies`` in
schemas/strategy_set3.py).

Strategy 1: Randomized Investment Simulation (Antifragility Experiment)
-----------------------------------------------------------------------
An EXPERIMENTAL STOCHASTIC SIMULATION - not a technical indicator, not a
trading signal and not a recommendation. Each iteration is an independent
portfolio (own cash and share balance) that makes a random buy / sell / hold
decision with a random size on every trading day. Results are compared with
Buy & Hold of the same asset (and of a benchmark) over the same window,
capital and cost model.

The experiment is inspired by Taleb's "Antifragile" discussion of
randomness. Nothing here tests, proves or disproves that theory: a
simulation on one price history cannot do so. It only reports how a
distribution of randomly-traded portfolios compared with Buy & Hold on this
particular history.

Modelling assumptions (all echoed in the response ``notes``)
* Price: adjusted close for both the simulations and Buy & Hold.
* Execution: all orders fill at that day's close price. The random draws
  never look at prices (or any future data), so there is no look-ahead bias;
  prices only cap the order size (affordability), at the fill price itself.
* Whole shares only. Buy size <= ``max_purchase_allocation_percent`` of cash;
  sell size <= ``max_sell_percent`` of held shares; the actual size is a
  uniform random integer up to that bound. No leverage, no shorting.
* Commission = ``commission_percent`` of trade value + ``commission_per_trade``
  flat, paid on every executed trade by the simulations AND by Buy & Hold's
  single entry trade.
* Idle cash earns no interest (Buy & Hold is invested from day one, so part
  of any gap is simply time in the market).
* No taxes, slippage, dividends beyond what adjusted prices embed, or
  end-of-period liquidation (everyone is marked to market on the last day).
* Randomness: one ``numpy.random.SeedSequence`` is spawned into one child
  stream per iteration, so iterations are independent and iteration k is the
  same no matter how many iterations are requested.
"""

from __future__ import annotations

import secrets

import numpy as np

from app.schemas.strategy_set3 import RandomizedSimulationParams
from app.strategies.set_3_strategies.common import (
    Set3Context,
    Set3Strategy,
    Set3StrategyUnavailable,
    annualized_return_pct,
    downsample_indices,
    max_drawdown_pct,
    rnd,
    slice_dates,
    to_jsonable,
    years_between,
)

STRATEGY_NAME = "Randomized Investment Simulation (Antifragility Experiment)"
MIN_TRADING_DAYS = 20
PRICE_BASIS = "adjusted_close"
EXECUTION_ASSUMPTION = "same-day close; random decisions use no price information"


# ---------------------------------------------------------------------------
# Core simulation (pure numpy; no I/O)
# ---------------------------------------------------------------------------


def _max_affordable_shares(
    budget: np.ndarray, price: float, rate: float, flat: float
) -> np.ndarray:
    """Largest whole share count whose price + commission fits in ``budget``."""
    unit = price * (1.0 + rate)
    return np.floor(np.maximum(budget - flat, 0.0) / unit).astype(np.int64)


def simulate_random_portfolios(
    prices: np.ndarray,
    *,
    initial_capital: float,
    buy_probability: float,
    sell_probability: float,
    max_purchase_fraction: float,
    max_sell_fraction: float,
    iterations: int,
    seed: int,
    commission_rate: float,
    commission_flat: float,
) -> dict[str, np.ndarray]:
    """
    Run ``iterations`` independent random portfolios over ``prices``.

    The loop is over days and vectorised over iterations; every iteration owns
    its cash / share balance and its own random stream.

    Returns arrays: ``equity`` (iterations x days), ``cash`` / ``shares`` (final
    balances), ``min_cash`` / ``min_shares`` (lowest balance ever held - used
    to prove the no-negative invariants), ``buys`` / ``sells`` /
    ``commission`` per iteration.
    """
    n_days = len(prices)
    children = np.random.SeedSequence(seed).spawn(iterations)
    # draws[i, t] = (action roll, size roll) for iteration i on day t
    draws = np.stack(
        [np.random.Generator(np.random.PCG64(c)).random((n_days, 2)) for c in children]
    )

    cash = np.full(iterations, float(initial_capital))
    shares = np.zeros(iterations, dtype=np.int64)
    buys = np.zeros(iterations, dtype=np.int64)
    sells = np.zeros(iterations, dtype=np.int64)
    commission = np.zeros(iterations)
    equity = np.empty((iterations, n_days))
    min_cash = cash.copy()
    min_shares = shares.copy()
    sell_threshold = buy_probability + sell_probability

    for t in range(n_days):
        price = float(prices[t])
        action_roll = draws[:, t, 0]
        size_roll = draws[:, t, 1]
        want_buy = action_roll < buy_probability
        want_sell = (~want_buy) & (action_roll < sell_threshold)

        # ---- buys: spend at most max_purchase_fraction of available cash
        max_buy = _max_affordable_shares(
            cash * max_purchase_fraction, price, commission_rate, commission_flat
        )
        buy_qty = np.where(
            want_buy & (max_buy >= 1), 1 + np.floor(size_roll * max_buy), 0
        ).astype(np.int64)
        buy_value = buy_qty * price
        buy_fee = np.where(buy_qty > 0, buy_value * commission_rate + commission_flat, 0.0)

        # ---- sells: sell at most max_sell_fraction of held shares
        max_sell = np.floor(shares * max_sell_fraction).astype(np.int64)
        sell_qty = np.where(
            want_sell & (max_sell >= 1), 1 + np.floor(size_roll * max_sell), 0
        ).astype(np.int64)
        sell_value = sell_qty * price
        sell_fee = np.where(sell_qty > 0, sell_value * commission_rate + commission_flat, 0.0)
        # A flat fee larger than the proceeds must never push cash below zero.
        sell_qty = np.where(cash + sell_value - sell_fee >= 0, sell_qty, 0)
        sell_value = sell_qty * price
        sell_fee = np.where(sell_qty > 0, sell_fee, 0.0)

        # buy and sell are mutually exclusive per day (want_sell excludes want_buy)
        cash = cash - buy_value - buy_fee + sell_value - sell_fee
        # float round-off guard only (affordability already guarantees >= 0)
        cash = np.maximum(cash, 0.0)
        shares = shares + buy_qty - sell_qty
        buys += buy_qty > 0
        sells += sell_qty > 0
        commission += buy_fee + sell_fee

        equity[:, t] = cash + shares * price
        min_cash = np.minimum(min_cash, cash)
        min_shares = np.minimum(min_shares, shares)

    return {
        "equity": equity,
        "cash": cash,
        "shares": shares,
        "buys": buys,
        "sells": sells,
        "commission": commission,
        "min_cash": min_cash,
        "min_shares": min_shares,
    }


def buy_and_hold(
    prices: np.ndarray, *, initial_capital: float, commission_rate: float, commission_flat: float
) -> dict:
    """Buy as many whole shares as possible on day 1 (same costs), then hold."""
    first = float(prices[0])
    unit = first * (1.0 + commission_rate)
    shares = int(max(initial_capital - commission_flat, 0.0) // unit)
    fee = shares * first * commission_rate + (commission_flat if shares > 0 else 0.0)
    cash = max(initial_capital - shares * first - fee, 0.0)
    equity = cash + shares * np.asarray(prices, dtype=float)
    return {"equity": equity, "shares": shares, "commission": fee}


def _bh_result(symbol: str, bh: dict, capital: float, years: float) -> dict:
    final = float(bh["equity"][-1])
    total = final / capital - 1.0
    return {
        "symbol": symbol,
        "initial_capital": rnd(capital),
        "shares": int(bh["shares"]),
        "final_value": rnd(final),
        "total_return_percent": rnd(total * 100),
        "annualized_return_percent": rnd(annualized_return_pct(total, years)),
        "max_drawdown_percent": rnd(max_drawdown_pct(bh["equity"])),
        "total_commission": rnd(bh["commission"]),
    }


# ---------------------------------------------------------------------------
# Strategy entry point
# ---------------------------------------------------------------------------


def compute_randomized_investment_simulation(ctx: Set3Context) -> dict:
    params: RandomizedSimulationParams = ctx.params or RandomizedSimulationParams()
    warnings: list[str] = []

    frame = slice_dates(ctx.prices, params.start_date, params.end_date)
    n_days = len(frame)
    if n_days < MIN_TRADING_DAYS:
        raise Set3StrategyUnavailable(
            f"Only {n_days} trading day(s) of {ctx.symbol} history in the selected "
            f"window; at least {MIN_TRADING_DAYS} are required."
        )

    if params.start_date and frame.index[0].date() > params.start_date:
        warnings.append(
            f"Requested start_date {params.start_date} is before the first available "
            f"price; simulation starts on {frame.index[0].date()}."
        )
    if params.end_date and frame.index[-1].date() < params.end_date:
        warnings.append(
            f"Requested end_date {params.end_date} is after the last available "
            f"price; simulation ends on {frame.index[-1].date()}."
        )

    dates = frame.index
    prices = frame["price"].to_numpy(dtype=float)
    start_d, end_d = dates[0].date(), dates[-1].date()
    years = years_between(start_d, end_d)
    if years < 1.0:
        warnings.append(
            "Window is shorter than one year: annualised figures extrapolate a short "
            "period and are not meaningful."
        )

    capital = float(params.initial_capital)
    rate = params.commission_percent / 100.0
    flat = float(params.commission_per_trade)
    seed_supplied = params.seed is not None
    seed_used = params.seed if seed_supplied else secrets.randbits(31)

    sim = simulate_random_portfolios(
        prices,
        initial_capital=capital,
        buy_probability=params.buy_probability,
        sell_probability=params.sell_probability,
        max_purchase_fraction=params.max_purchase_allocation_percent / 100.0,
        max_sell_fraction=params.max_sell_percent / 100.0,
        iterations=params.iterations,
        seed=seed_used,
        commission_rate=rate,
        commission_flat=flat,
    )
    equity = sim["equity"]

    # ---- Buy & Hold of the same asset, same window / capital / costs
    bh = buy_and_hold(prices, initial_capital=capital, commission_rate=rate, commission_flat=flat)
    bh_result = _bh_result(ctx.symbol, bh, capital, years)
    bh_return = bh["equity"][-1] / capital - 1.0

    # ---- Benchmark Buy & Hold, aligned to the simulation dates (as-of / ffill,
    #      so a benchmark value never comes from a later date)
    bench_equity = None
    bench_result = None
    bench_return = None
    bench_symbol = params.benchmark_symbol
    bench_frame = ctx.extra_prices.get(bench_symbol)
    if bench_symbol == ctx.symbol:
        bench_equity, bench_result, bench_return = bh["equity"], bh_result, bh_return
    elif bench_frame is None or bench_frame.empty:
        warnings.append(f"Benchmark {bench_symbol} has no stored price history; benchmark omitted.")
    else:
        aligned = bench_frame["price"].reindex(dates.union(bench_frame.index)).ffill().reindex(dates)
        if aligned.isna().any():
            warnings.append(
                f"Benchmark {bench_symbol} history starts after {start_d}; benchmark omitted."
            )
        else:
            b = buy_and_hold(
                aligned.to_numpy(dtype=float),
                initial_capital=capital,
                commission_rate=rate,
                commission_flat=flat,
            )
            bench_equity = b["equity"]
            bench_result = _bh_result(bench_symbol, b, capital, years)
            bench_return = bench_equity[-1] / capital - 1.0

    # ---- per-iteration results
    final = equity[:, -1]
    total_ret = final / capital - 1.0
    ann = annualized_return_pct(total_ret, years)
    mdd = max_drawdown_pct(equity)
    diff_bh = (total_ret - bh_return) * 100.0
    beat_bh = total_ret > bh_return
    beat_bench = None if bench_return is None else total_ret > bench_return

    iteration_results = []
    for i in range(params.iterations):
        iteration_results.append(
            {
                "iteration": i + 1,
                "final_value": rnd(final[i]),
                "final_cash": rnd(sim["cash"][i]),
                "final_shares": int(sim["shares"][i]),
                "total_return_percent": rnd(total_ret[i] * 100),
                "annualized_return_percent": None if ann is None else rnd(ann[i]),
                "max_drawdown_percent": rnd(mdd[i]),
                "buy_trades": int(sim["buys"][i]),
                "sell_trades": int(sim["sells"][i]),
                "total_commission": rnd(sim["commission"][i]),
                "difference_vs_buy_and_hold_points": rnd(diff_bh[i]),
                "beat_buy_and_hold": bool(beat_bh[i]),
                "beat_benchmark": None if beat_bench is None else bool(beat_bench[i]),
            }
        )

    # ---- aggregate statistics across iterations
    n = params.iterations
    aggregate = {
        "iterations": n,
        "initial_capital": rnd(capital),
        "average_final_value": rnd(final.mean()),
        "median_final_value": rnd(np.median(final)),
        "average_return_percent": rnd(total_ret.mean() * 100),
        "median_return_percent": rnd(np.median(total_ret) * 100),
        "best_return_percent": rnd(total_ret.max() * 100),
        "worst_return_percent": rnd(total_ret.min() * 100),
        "std_dev_return_percent": rnd(total_ret.std(ddof=1) * 100 if n > 1 else 0.0),
        "average_annualized_return_percent": None if ann is None else rnd(ann.mean()),
        "median_annualized_return_percent": None if ann is None else rnd(np.median(ann)),
        "average_max_drawdown_percent": rnd(mdd.mean()),
        "median_max_drawdown_percent": rnd(np.median(mdd)),
        "worst_max_drawdown_percent": rnd(mdd.min()),
        "average_trades": rnd((sim["buys"] + sim["sells"]).mean()),
        "percent_outperforming_buy_and_hold": rnd(beat_bh.mean() * 100),
        "percent_outperforming_benchmark": None if beat_bench is None else rnd(beat_bench.mean() * 100),
        "average_difference_vs_buy_and_hold_points": rnd(diff_bh.mean()),
        "median_difference_vs_buy_and_hold_points": rnd(np.median(diff_bh)),
        "average_difference_vs_benchmark_points": (
            None if bench_return is None else rnd((total_ret.mean() - bench_return) * 100)
        ),
    }

    # ---- chart data (portfolio value over time), downsampled for the frontend
    best_i = int(np.argmax(final))
    worst_i = int(np.argmin(final))
    median_i = int(np.argsort(final)[(n - 1) // 2])
    mean_path = equity.mean(axis=0)
    median_path = np.median(equity, axis=0)
    p5 = np.percentile(equity, 5, axis=0)
    p95 = np.percentile(equity, 95, axis=0)
    chart = []
    for t in downsample_indices(n_days, params.chart_max_points):
        chart.append(
            {
                "date": dates[t].date().isoformat(),
                "buy_and_hold": rnd(bh["equity"][t]),
                "benchmark": None if bench_equity is None else rnd(bench_equity[t]),
                "mean": rnd(mean_path[t]),
                "median": rnd(median_path[t]),
                "p5": rnd(p5[t]),
                "p95": rnd(p95[t]),
                "best_iteration": rnd(equity[best_i, t]),
                "worst_iteration": rnd(equity[worst_i, t]),
                "median_iteration": rnd(equity[median_i, t]),
            }
        )

    notes = [
        "Experimental stochastic simulation for research / education. Random trading "
        "is not a recommended or guaranteed investment strategy, and past simulated "
        "results do not predict future returns.",
        "Inspired by the Antifragility discussion of randomness; this experiment does "
        "not prove or disprove that theory.",
        f"Prices: {PRICE_BASIS}. Execution: {EXECUTION_ASSUMPTION}.",
        "Whole shares only; no leverage or shorting; cash and holdings can never go negative.",
        "Commission (percent + flat) applies to every executed trade, including Buy & Hold's entry.",
        "Idle cash earns no interest; everything is marked to market on the last day (no liquidation).",
        "Each iteration uses an independent random stream derived from one seed.",
        "'aggregate' summarises all iterations; 'iterationResults' are the individual runs.",
    ]
    if not seed_supplied:
        warnings.append(
            f"No seed supplied: results are not reproducible by default. "
            f"Re-send with seed={seed_used} to reproduce this exact run."
        )

    return to_jsonable(
        {
            "name": STRATEGY_NAME,
            "category": "quantitative",
            "symbol": ctx.symbol,
            "parameters": {
                "initial_capital": rnd(capital),
                "buy_probability": params.buy_probability,
                "sell_probability": params.sell_probability,
                "hold_probability": rnd(max(0.0, 1.0 - params.buy_probability - params.sell_probability), 6),
                "max_purchase_allocation_percent": params.max_purchase_allocation_percent,
                "max_sell_percent": params.max_sell_percent,
                "iterations": n,
                "seed_used": int(seed_used),
                "seed_supplied": seed_supplied,
                "reproducible": seed_supplied,
                "benchmark_symbol": bench_symbol,
                "commission_percent": params.commission_percent,
                "commission_per_trade": flat,
                "price_basis": PRICE_BASIS,
                "execution_assumption": EXECUTION_ASSUMPTION,
            },
            "period": {
                "start_date": start_d,
                "end_date": end_d,
                "trading_days": n_days,
                "years": rnd(years, 3),
            },
            "buy_and_hold": bh_result,
            "benchmark": bench_result,
            "aggregate": aggregate,
            "iteration_results": iteration_results,
            "sample_paths": {
                "best_iteration": best_i + 1,
                "worst_iteration": worst_i + 1,
                "median_iteration": median_i + 1,
            },
            "chart_data": chart,
            "warnings": warnings,
            "notes": notes,
        }
    )


def _required_symbols(params: RandomizedSimulationParams | None) -> tuple[str, ...]:
    return ((params or RandomizedSimulationParams()).benchmark_symbol,)


def _cacheable(params: RandomizedSimulationParams | None) -> bool:
    # Without a seed every call must produce a fresh random draw.
    return params is not None and params.seed is not None


# ---------------------------------------------------------------------------
# Registry - add new quantitative strategies here
# ---------------------------------------------------------------------------

QUANTITATIVE_STRATEGIES: dict[str, Set3Strategy] = {
    "randomized_investment_simulation": Set3Strategy(
        key="randomized_investment_simulation",
        title=STRATEGY_NAME,
        category="quantitative",
        compute=compute_randomized_investment_simulation,
        required_symbols=_required_symbols,
        cacheable=_cacheable,
    ),
}
