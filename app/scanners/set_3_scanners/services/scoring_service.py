"""
Pure scoring logic for the Technical-Fundamental Composite Scoring scanner.

No I/O. Conditions (article scoring rules, thresholds configurable):

    price > 50-day moving average   -> 1 point
    RSI < 70                        -> 1 point
    forward P/E < 20                -> 1 point
    debt-to-equity < 1              -> 1 point

Each condition is evaluated INDEPENDENTLY and reports one of
``passed`` / ``failed`` / ``unavailable`` / ``not_meaningful``:

* ``unavailable``    the input is missing -> NO point and NO pretence of failure
                     or success (score = null); the stock's score is flagged
                     ``score_complete = false``.
* ``not_meaningful`` the input exists but cannot be compared with "below X"
                     (forward P/E <= 0 means negative / zero expected earnings;
                     negative debt-to-equity means negative equity). Awarding the
                     point for such a value (as a naive ``value < threshold`` does)
                     would be wrong, so no point is given and the reason is stated.

RSI < 30 is exposed as a SEPARATE flag and never adds to the 0-4 score.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.scanners.set_3_scanners.schemas.quantitative import (
    ConditionResult,
    EffectiveParameters,
    IndividualScores,
    OpportunityFlag,
    ScannerDataAvailability,
    StockScoreResult,
)

MAX_SCORE = 4


def required_price_observations(ma_period: int, rsi_period: int) -> int:
    """Minimum closes for BOTH indicators (MA needs `ma_period`, RSI needs period+1)."""
    return max(ma_period, rsi_period + 1)


def fetch_target_observations(ma_period: int, rsi_period: int) -> int:
    """How many trading observations to ask for: the minimum plus a warm-up
    buffer so Wilder smoothing is not dominated by its seed values."""
    return required_price_observations(ma_period, rsi_period) + 100


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def simple_moving_average(close: pd.Series, period: int) -> float | None:
    """Latest simple moving average over exactly ``period`` closes, else None."""
    c = close.dropna()
    if len(c) < period:
        return None
    v = float(c.iloc[-period:].mean())
    return v if math.isfinite(v) else None


def wilder_rsi(close: pd.Series, period: int) -> float | None:
    """Latest RSI with Wilder's smoothing (first average = simple mean of the
    first ``period`` changes, then avg = (prev*(period-1) + x) / period).

    Needs at least period + 1 closes. Returns None when RSI is undefined (flat
    prices: no gains and no losses at all).
    """
    c = close.dropna().to_numpy(dtype=float)
    if len(c) < period + 1:
        return None
    delta = np.diff(c)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = float(gains[:period].mean())
    avg_loss = float(losses[:period].mean())
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else None
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def _cond(name: str, rule: str, threshold: float, value: float | None, status: str,
          message: str | None = None) -> ConditionResult:
    passed = True if status == "passed" else False if status in ("failed", "not_meaningful") else None
    score = 1 if status == "passed" else 0 if status in ("failed", "not_meaningful") else None
    return ConditionResult(
        name=name, rule=rule, threshold=threshold,
        value=None if value is None else round(float(value), 4),
        status=status,  # type: ignore[arg-type]
        passed=passed, score=score, message=message,
    )


def eval_price_above_ma(price: float | None, ma: float | None, period: int) -> ConditionResult:
    rule = f"Current price above the {period}-day moving average"
    if price is None or ma is None:
        return _cond("price_above_ma", rule, period, None, "unavailable",
                     f"Not enough price history to compute the {period}-day moving average.")
    return _cond("price_above_ma", rule, period, price - ma,
                 "passed" if price > ma else "failed")


def eval_rsi_below(rsi: float | None, threshold: float, period: int) -> ConditionResult:
    rule = f"RSI({period}) below {threshold:g}"
    if rsi is None:
        return _cond("rsi_below_upper", rule, threshold, None, "unavailable",
                     f"RSI({period}) could not be calculated from the available prices.")
    return _cond("rsi_below_upper", rule, threshold, rsi, "passed" if rsi < threshold else "failed")


def eval_forward_pe(pe: float | None, threshold: float) -> ConditionResult:
    rule = f"Forward P/E below {threshold:g}"
    if pe is None:
        return _cond("forward_pe_below_threshold", rule, threshold, None, "unavailable",
                     "Forward P/E is not available from the data provider.")
    if pe <= 0:
        return _cond("forward_pe_below_threshold", rule, threshold, pe, "not_meaningful",
                     "Forward P/E is zero or negative (no positive expected earnings); "
                     "it cannot be compared with the threshold, so no point is awarded.")
    return _cond("forward_pe_below_threshold", rule, threshold, pe, "passed" if pe < threshold else "failed")


def eval_debt_to_equity(de: float | None, threshold: float) -> ConditionResult:
    rule = f"Debt-to-equity ratio below {threshold:g}"
    if de is None:
        return _cond("debt_to_equity_below_threshold", rule, threshold, None, "unavailable",
                     "Debt-to-equity is not available from the data provider.")
    if de < 0:
        return _cond("debt_to_equity_below_threshold", rule, threshold, de, "not_meaningful",
                     "Debt-to-equity is negative (negative shareholder equity); it cannot be "
                     "compared with the threshold, so no point is awarded.")
    return _cond("debt_to_equity_below_threshold", rule, threshold, de, "passed" if de < threshold else "failed")


def eval_rsi_opportunity(rsi: float | None, threshold: float) -> OpportunityFlag:
    if rsi is None:
        return OpportunityFlag(threshold=threshold, rsi=None, flagged=None,
                               message="RSI unavailable; opportunity flag not evaluated.")
    return OpportunityFlag(threshold=threshold, rsi=round(rsi, 4), flagged=bool(rsi < threshold))


# ---------------------------------------------------------------------------
# Whole-stock scoring
# ---------------------------------------------------------------------------


def score_stock(
    symbol: str,
    close: pd.Series | None,
    forward_pe: float | None,
    debt_to_equity: float | None,
    params: EffectiveParameters,
    *,
    debt_to_equity_provider_value: float | None = None,
    extra_messages: list[str] | None = None,
) -> StockScoreResult:
    """Combine the four independent conditions into a result (never raises on
    missing data)."""
    messages = list(extra_messages or [])
    required = required_price_observations(params.ma_period, params.rsi_period)
    close = close.dropna() if close is not None else pd.Series(dtype=float)
    n = int(len(close))

    price = float(close.iloc[-1]) if n else None
    as_of = close.index[-1].date().isoformat() if n else None
    ma = simple_moving_average(close, params.ma_period)
    rsi = wilder_rsi(close, params.rsi_period)

    if n and n < required:
        messages.append(
            f"Only {n} price observations available; {required} are needed for the "
            "selected moving-average and RSI periods."
        )
    if n == 0:
        messages.append("No price history was returned by the data provider.")

    c1 = eval_price_above_ma(price, ma, params.ma_period)
    c2 = eval_rsi_below(rsi, params.rsi_upper_threshold, params.rsi_period)
    c3 = eval_forward_pe(forward_pe, params.forward_pe_threshold)
    c4 = eval_debt_to_equity(debt_to_equity, params.debt_to_equity_threshold)
    conds = [c1, c2, c3, c4]
    evaluated = sum(1 for c in conds if c.score is not None)
    total = sum(c.score or 0 for c in conds) if evaluated else None
    for c in conds:
        if c.status in ("unavailable", "not_meaningful") and c.message:
            messages.append(f"{c.name}: {c.message}")

    if evaluated == MAX_SCORE:
        status = "complete"
    elif evaluated > 0:
        status = "partial"
    else:
        status = "unavailable"

    return StockScoreResult(
        symbol=symbol,
        as_of_date=as_of,
        current_price=None if price is None else round(price, 4),
        moving_average=None if ma is None else round(ma, 4),
        rsi=None if rsi is None else round(rsi, 4),
        forward_pe=None if forward_pe is None else round(forward_pe, 4),
        debt_to_equity=None if debt_to_equity is None else round(debt_to_equity, 4),
        debt_to_equity_provider_value=debt_to_equity_provider_value,
        price_above_ma=c1,
        rsi_below_upper=c2,
        forward_pe_below_threshold=c3,
        debt_to_equity_below_threshold=c4,
        individual_scores=IndividualScores(
            price_above_ma=c1.score,
            rsi_below_upper=c2.score,
            forward_pe_below_threshold=c3.score,
            debt_to_equity_below_threshold=c4.score,
        ),
        total_score=total,
        evaluated_conditions=evaluated,
        score_complete=evaluated == MAX_SCORE,
        rsi_opportunity=eval_rsi_opportunity(rsi, params.rsi_opportunity_threshold),
        data_availability=ScannerDataAvailability(
            price_history_observations=n,
            required_price_observations=required,
            price_history_available=ma is not None and rsi is not None,
            forward_pe_available=forward_pe is not None,
            debt_to_equity_available=debt_to_equity is not None,
            messages=messages,
        ),
        calculation_status=status,  # type: ignore[arg-type]
    )
