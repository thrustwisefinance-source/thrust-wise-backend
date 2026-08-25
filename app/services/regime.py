"""
Risk-On / Risk-Off Market Regime engine.

Unlike every strategy in services/strategies.py (single-symbol technical
indicators), this module is explicitly CROSS-ASSET: it looks at the
behavior of every ETF in the Thrustwise universe together, on weekly
data, and classifies the overall market "regime" plus each ETF's role
within it. This is a quantitative, descriptive signal only — it is NOT a
crash predictor, NOT a buy/sell recommendation, and carries no
guarantee about future returns.

Pipeline (mirrors the source methodology):

    daily prices (per ETF)
            -> resampled to weekly OHLCV
            -> technical features per ETF
            -> per-ETF z-score standardization
            -> pairwise Dynamic Time Warping (DTW) distance
            -> 2-cluster average-linkage clustering
            -> cluster labeled Risk-On / Risk-Off
            -> per-symbol result + market-wide regime score

IMPORTANT — UNIVERSE LIMITATION: the source methodology this is modeled
on used a broader universe (~21 ETFs spanning many asset classes). The
Thrustwise backend currently only tracks the six ETFs in
app.constants.ETF_REGISTRY (VOO, VTI, SPY, QQQ, GLD, SHY). A 6-ETF
universe is NOT equivalent to a 21-ETF universe: clustering six assets
into two groups is a much coarser, noisier read on "market regime" than
clustering across many asset classes and geographies. This is reported
back in the response (`etf_universe`) and should be read as a narrow,
directional signal, not a comprehensive market-regime model.

Same shape as the rest of the services layer: pure functions over
`dict[str, list[DailyPrice]]` already loaded from the database — no DB
or HTTP access here, no external calls, no ML libraries (DTW and
clustering are implemented directly with pandas/numpy below since the
project has no scipy/sklearn dependency).
"""

import math

import numpy as np
import pandas as pd

from app.models import DailyPrice

# Reuse the existing EMA/RSI implementations rather than duplicating them
# (per project convention — see services/strategies.py module docstring).
from app.services.strategies import _ema, _rsi

STRATEGY_KEY = "risk_on_risk_off"

WEEKLY_RESAMPLE_RULE = "W-FRI"
BENCHMARK_SYMBOL = "SPY"

RSI_WEEKS = 14
MACD_FAST_WEEKS = 12
MACD_SLOW_WEEKS = 26
MACD_SIGNAL_WEEKS = 9
BB_WEEKS = 20
BB_STD_MULTIPLIER = 2
ADX_WEEKS = 14
CMF_WEEKS = 20
VOL_WINDOW_WEEKS = 12
TRADING_WEEKS_PER_YEAR = 52

# Minimum weekly bars an ETF needs before it's considered "eligible" —
# covers the longest feature warm-up (ADX/CMF ~20 weeks) plus enough
# subsequent history to standardize and compare meaningfully.
MIN_WEEKS_REQUIRED = 90

# Trailing window (weeks) of standardized features compared via DTW.
DTW_WINDOW_WEEKS = 52
# Minimum number of complete (non-NaN) weeks within that window required
# for a symbol to be included in the DTW/clustering step.
DTW_MIN_WEEKS = 30

# Regime-score chart: last N weeks shown, matching the spirit of
# strategies.CHART_LOOKBACK_DAYS but at weekly resolution.
CHART_LOOKBACK_WEEKS = 104

# Spreads the composite (roughly N(0, ~0.4)) score out across 0-100 via a
# logistic curve. Chosen empirically so a "typical" week reads near 50
# and a strongly one-sided week approaches the 0/100 bounds; documented
# heuristic, not a fitted parameter.
REGIME_SCORE_SCALE = 3.0

FEATURE_COLUMNS = [
    "rsi",
    "macd_hist",
    "ppo",
    "cmf",
    "bb_percent_b",
    "rolling_vol",
    "downside_dev",
    "adx",
    "scaled_return",
]

# Sign convention for the composite "risk appetite" score: +1 means
# "higher reading = more risk-on", -1 means the opposite. ADX (trend
# strength, direction-less) is handled separately since it needs to be
# signed by the direction of momentum first.
FEATURE_SIGN = {
    "rsi": 1,
    "macd_hist": 1,
    "ppo": 1,
    "cmf": 1,
    "bb_percent_b": 1,
    "rolling_vol": -1,
    "downside_dev": -1,
    "scaled_return": 1,
}


def _round(value, digits: int = 3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(value, digits)


# ---------------------------------------------------------------------------
# Weekly resampling + per-ETF technical features
# ---------------------------------------------------------------------------


def _to_weekly_frame(prices: list[DailyPrice]) -> pd.DataFrame:
    """High/Low/Close/Volume resampled to weekly bars (Friday close)."""
    if not prices:
        return pd.DataFrame(columns=["high", "low", "close", "volume"])
    daily = pd.DataFrame(
        {
            "high": [p.high for p in prices],
            "low": [p.low for p in prices],
            "close": [p.close for p in prices],
            "volume": [p.volume or 0 for p in prices],
        },
        index=pd.DatetimeIndex([p.date for p in prices]),
    ).sort_index()
    weekly = daily.resample(WEEKLY_RESAMPLE_RULE).agg(
        {"high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return weekly.dropna(subset=["close"])


def _adx(weekly: pd.DataFrame, period: int = ADX_WEEKS) -> pd.Series:
    """Wilder's Average Directional Index (trend strength, 0-100,
    direction-less)."""
    high, low, close = weekly["high"], weekly["low"], weekly["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=weekly.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=weekly.index,
    )

    atr = true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0, np.nan)
    )
    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _cmf(weekly: pd.DataFrame, period: int = CMF_WEEKS) -> pd.Series:
    """Chaikin Money Flow: volume-weighted accumulation/distribution."""
    high, low, close, volume = (
        weekly["high"],
        weekly["low"],
        weekly["close"],
        weekly["volume"],
    )
    price_range = (high - low).replace(0, np.nan)
    money_flow_multiplier = (((close - low) - (high - close)) / price_range).fillna(
        0.0
    )
    money_flow_volume = money_flow_multiplier * volume
    return money_flow_volume.rolling(period).sum() / volume.rolling(
        period
    ).sum().replace(0, np.nan)


def _bollinger_percent_b(
    close: pd.Series, period: int = BB_WEEKS, mult: float = BB_STD_MULTIPLIER
) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + mult * std
    lower = sma - mult * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


def _rolling_volatility(returns: pd.Series, window: int = VOL_WINDOW_WEEKS) -> pd.Series:
    """Annualized rolling volatility of weekly returns."""
    return returns.rolling(window).std() * math.sqrt(TRADING_WEEKS_PER_YEAR)


def _downside_deviation(returns: pd.Series, window: int = VOL_WINDOW_WEEKS) -> pd.Series:
    """Annualized rolling downside deviation (only negative weeks count)."""
    downside = returns.clip(upper=0.0)
    return (downside.pow(2).rolling(window).mean()).pow(0.5) * math.sqrt(
        TRADING_WEEKS_PER_YEAR
    )


def _build_features(
    weekly: pd.DataFrame, spy_scaled_return: pd.Series | None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix for one ETF. Returns (features, weekly_returns,
    rolling_volatility) — the latter two are reused by the caller to
    build SPY's `spy_scaled_return` for the other ETFs.
    """
    close = weekly["close"]
    returns = close.pct_change()

    ema_fast = _ema(close, MACD_FAST_WEEKS)
    ema_slow = _ema(close, MACD_SLOW_WEEKS)
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(
        span=MACD_SIGNAL_WEEKS, adjust=False, min_periods=MACD_SIGNAL_WEEKS
    ).mean()
    macd_hist = macd_line - signal
    ppo = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan) * 100

    rsi = _rsi(close, RSI_WEEKS)
    vol = _rolling_volatility(returns)
    downside_dev = _downside_deviation(returns)
    bb_percent_b = _bollinger_percent_b(close)
    cmf = _cmf(weekly)
    adx = _adx(weekly)

    own_scaled_return = returns / vol.replace(0, np.nan)
    if spy_scaled_return is not None:
        scaled_return = own_scaled_return - spy_scaled_return.reindex(
            own_scaled_return.index
        )
    else:
        # This IS the SPY series itself (or SPY is unavailable) — relative
        # performance vs SPY is undefined, so this feature is left neutral.
        scaled_return = pd.Series(0.0, index=own_scaled_return.index)

    features = pd.DataFrame(
        {
            "rsi": rsi,
            "macd_hist": macd_hist,
            "ppo": ppo,
            "cmf": cmf,
            "bb_percent_b": bb_percent_b,
            "rolling_vol": vol,
            "downside_dev": downside_dev,
            "adx": adx,
            "scaled_return": scaled_return,
        }
    )
    return features, returns, own_scaled_return


def _standardize(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature z-score using this ETF's own history (mean/std),
    matching the source methodology's "standardize the indicators" step
    — puts every ETF's features on a comparable scale before cross-asset
    comparison (e.g. GLD's volatility scale vs QQQ's).

    A feature with zero variance (e.g. `scaled_return` for SPY itself,
    which is always exactly 0 — see _build_features) would otherwise
    divide by zero and turn every row NaN, silently dropping that whole
    symbol from later steps. Those columns are treated as "no signal"
    and filled with 0 rather than propagated as NaN; genuine warm-up
    NaNs (rolling windows not yet full) in other columns are untouched.
    """
    mean = features.mean(skipna=True)
    std = features.std(skipna=True)
    standardized = (features - mean) / std.replace(0, np.nan)

    zero_variance_cols = std[std == 0].index
    if len(zero_variance_cols) > 0:
        standardized[zero_variance_cols] = standardized[zero_variance_cols].fillna(
            0.0
        )
    return standardized


def _composite_score(standardized_row: pd.Series) -> float | None:
    """Single 'risk appetite' number from one week's standardized
    features: positive = more risk-on, negative = more risk-off. ADX
    (trend strength, direction-less) is signed by the direction of MACD
    momentum to turn it into a directional contribution.
    """
    terms = []
    for col in FEATURE_COLUMNS:
        if col == "adx":
            continue
        val = standardized_row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        terms.append(FEATURE_SIGN[col] * val)

    adx_val = standardized_row.get("adx")
    macd_val = standardized_row.get("macd_hist")
    if (
        adx_val is not None
        and macd_val is not None
        and not (isinstance(adx_val, float) and math.isnan(adx_val))
        and not (isinstance(macd_val, float) and math.isnan(macd_val))
    ):
        direction = 1.0 if macd_val >= 0 else -1.0
        terms.append(adx_val * direction)

    if not terms:
        return None
    return float(np.mean(terms))


def _regime_score_from_composite(composite: float) -> float:
    return 100.0 / (1.0 + math.exp(-REGIME_SCORE_SCALE * composite))


# ---------------------------------------------------------------------------
# Dynamic Time Warping + clustering
# ---------------------------------------------------------------------------


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW distance between two (T, F) sequences using Euclidean
    local cost. Unconstrained warping path (no Sakoe-Chiba band) — both
    inputs are capped at DTW_WINDOW_WEEKS rows so the O(n*m) DP table
    stays small (<= ~2,700 cells) and cheap even across every ETF pair.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")

    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local_cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = local_cost + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    return float(cost[n, m])


def _cluster_into_two(symbols: list[str], distance: dict[tuple[str, str], float]) -> dict[str, int]:
    """Average-linkage agglomerative clustering into exactly 2 clusters.
    Small-N (<=6 ETFs today) so a plain O(k^3) merge loop is more than
    fast enough — no external clustering dependency required.
    """
    if len(symbols) <= 2:
        return {sym: i for i, sym in enumerate(symbols)}

    def pair_distance(sym_a: str, sym_b: str) -> float:
        key = (sym_a, sym_b) if (sym_a, sym_b) in distance else (sym_b, sym_a)
        return distance[key]

    clusters: list[set[str]] = [{s} for s in symbols]

    while len(clusters) > 2:
        best_pair = None
        best_avg = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                pair_dists = [
                    pair_distance(a, b) for a in clusters[i] for b in clusters[j]
                ]
                avg = sum(pair_dists) / len(pair_dists)
                if avg < best_avg:
                    best_avg = avg
                    best_pair = (i, j)
        i, j = best_pair
        merged = clusters[i] | clusters[j]
        clusters = [c for idx, c in enumerate(clusters) if idx not in (i, j)]
        clusters.append(merged)

    assignment: dict[str, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        for sym in cluster:
            assignment[sym] = cluster_id
    return assignment


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def compute_market_regime(prices_by_symbol: dict[str, list[DailyPrice]]) -> dict | None:
    """Full cross-ETF Risk-On/Risk-Off computation.

    Returns None when fewer than 2 ETFs have enough weekly history to
    compare (clustering needs at least two points) — callers should
    treat that as "unavailable", the same "None" convention every other
    strategy in this codebase uses for insufficient data. A single bad
    or missing symbol never crashes the computation for the rest of the
    universe; it's simply excluded and reported in `skipped_symbols`.

    This function is deliberately the ONLY place DTW + clustering runs.
    It is meant to be called once (via a shared cache key — see
    routers/strategy.py) rather than per individual ETF request, since
    the result does not depend on which symbol the frontend asked about.
    """
    weekly_frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for symbol, prices in prices_by_symbol.items():
        weekly = _to_weekly_frame(prices)
        if len(weekly) < MIN_WEEKS_REQUIRED:
            skipped.append(symbol)
            continue
        weekly_frames[symbol] = weekly

    if len(weekly_frames) < 2:
        return None

    # Pass 1: SPY's own scaled-return series (if SPY qualifies), used as
    # the relative-performance benchmark for every other ETF's feature.
    spy_scaled_return = None
    if BENCHMARK_SYMBOL in weekly_frames:
        spy_close = weekly_frames[BENCHMARK_SYMBOL]["close"]
        spy_returns = spy_close.pct_change()
        spy_vol = _rolling_volatility(spy_returns)
        spy_scaled_return = spy_returns / spy_vol.replace(0, np.nan)

    features_by_symbol: dict[str, pd.DataFrame] = {}
    returns_by_symbol: dict[str, pd.Series] = {}
    for symbol, weekly in weekly_frames.items():
        pass_spy = None if symbol == BENCHMARK_SYMBOL else spy_scaled_return
        features, returns, _ = _build_features(weekly, pass_spy)
        features_by_symbol[symbol] = features
        returns_by_symbol[symbol] = returns

    standardized_by_symbol = {
        symbol: _standardize(features)
        for symbol, features in features_by_symbol.items()
    }

    # Trim to the shared DTW comparison window and drop incomplete rows.
    dtw_ready: dict[str, np.ndarray] = {}
    for symbol, standardized in standardized_by_symbol.items():
        window = standardized.tail(DTW_WINDOW_WEEKS).dropna()
        if len(window) >= DTW_MIN_WEEKS:
            dtw_ready[symbol] = window[FEATURE_COLUMNS].to_numpy(dtype=float)
        else:
            skipped.append(symbol)

    dtw_symbols = list(dtw_ready.keys())
    if len(dtw_symbols) < 2:
        return None

    distance: dict[tuple[str, str], float] = {}
    for i in range(len(dtw_symbols)):
        for j in range(i + 1, len(dtw_symbols)):
            sym_a, sym_b = dtw_symbols[i], dtw_symbols[j]
            distance[(sym_a, sym_b)] = _dtw_distance(dtw_ready[sym_a], dtw_ready[sym_b])

    cluster_assignment = _cluster_into_two(dtw_symbols, distance)

    # Latest composite score per symbol (for cluster labeling + market score)
    composite_by_symbol: dict[str, float] = {}
    for symbol in dtw_symbols:
        standardized = standardized_by_symbol[symbol].dropna(how="all")
        if standardized.empty:
            continue
        score = _composite_score(standardized.iloc[-1])
        if score is not None:
            composite_by_symbol[symbol] = score

    cluster_avg_composite: dict[int, list[float]] = {}
    for symbol, cluster_id in cluster_assignment.items():
        if symbol in composite_by_symbol:
            cluster_avg_composite.setdefault(cluster_id, []).append(
                composite_by_symbol[symbol]
            )
    cluster_label: dict[int, str] = {}
    if cluster_avg_composite:
        ranked = sorted(
            cluster_avg_composite.items(), key=lambda kv: sum(kv[1]) / len(kv[1])
        )
        risk_off_cluster_id = ranked[0][0]
        risk_on_cluster_id = ranked[-1][0]
        cluster_label[risk_off_cluster_id] = "Risk-Off"
        cluster_label[risk_on_cluster_id] = "Risk-On"
    # Any cluster id not covered above (e.g. all-NaN composites) defaults
    # to "Risk-Off" — the conservative reading when data is inconclusive.

    regime_score = (
        _regime_score_from_composite(
            sum(composite_by_symbol.values()) / len(composite_by_symbol)
        )
        if composite_by_symbol
        else 50.0
    )
    market_regime = "Risk-On" if regime_score >= 50.0 else "Risk-Off"

    # DTW cohesion metric per symbol: mean distance to every other symbol
    # in the comparison set (lower = more "in step" with the rest of the
    # universe this period, higher = more of an outlier).
    dtw_distance_to_market: dict[str, float] = {}
    for symbol in dtw_symbols:
        others = [
            distance[(symbol, other)] if (symbol, other) in distance else distance[(other, symbol)]
            for other in dtw_symbols
            if other != symbol
        ]
        if others:
            dtw_distance_to_market[symbol] = sum(others) / len(others)

    per_symbol: dict[str, dict] = {}
    for symbol in dtw_symbols:
        features = features_by_symbol[symbol]
        if features.empty:
            continue
        latest = features.iloc[-1]
        per_symbol[symbol] = {
            "etf_regime": cluster_label.get(cluster_assignment[symbol], "Risk-Off"),
            "cluster": cluster_assignment[symbol],
            "adx": _round(latest.get("adx")),
            "rolling_volatility": _round(latest.get("rolling_vol")),
            "downside_deviation": _round(latest.get("downside_dev")),
            "scaled_return_vs_spy": _round(latest.get("scaled_return")),
            "dtw_distance_to_market": _round(dtw_distance_to_market.get(symbol)),
        }

    chart_data = _build_chart(standardized_by_symbol, returns_by_symbol)

    return {
        "market_regime": market_regime,
        "regime_score": _round(regime_score, 2),
        "etf_universe": sorted(dtw_symbols),
        "skipped_symbols": sorted(set(skipped)),
        "per_symbol": per_symbol,
        "chart_data": chart_data,
    }


def _build_chart(
    standardized_by_symbol: dict[str, pd.DataFrame],
    returns_by_symbol: dict[str, pd.Series],
) -> list[dict]:
    """Market-wide regime-score history.

    NOTE ON SCOPE: re-running DTW + clustering at every historical week
    would be computationally prohibitive for an API request (the whole
    point of caching the single current-snapshot clustering result — see
    routers/strategy.py). This chart instead tracks the same underlying
    standardized features' cross-sectional composite score over time
    (without DTW/clustering at each point), which is a documented
    approximation of regime evolution, not a re-run of the full pipeline
    at every date.
    """
    all_composites: dict[pd.Timestamp, list[float]] = {}
    all_vols: dict[pd.Timestamp, list[float]] = {}

    for symbol, standardized in standardized_by_symbol.items():
        tail = standardized.tail(CHART_LOOKBACK_WEEKS)
        for ts, row in tail.iterrows():
            score = _composite_score(row)
            if score is not None:
                all_composites.setdefault(ts, []).append(score)
        vol_series = returns_by_symbol.get(symbol)
        if vol_series is not None:
            vol_tail = _rolling_volatility(vol_series).tail(CHART_LOOKBACK_WEEKS)
            for ts, val in vol_tail.items():
                if not (isinstance(val, float) and math.isnan(val)):
                    all_vols.setdefault(ts, []).append(float(val))

    rows = []
    for ts in sorted(all_composites.keys()):
        composites = all_composites[ts]
        if not composites:
            continue
        avg_composite = sum(composites) / len(composites)
        score = _regime_score_from_composite(avg_composite)
        vols = all_vols.get(ts)
        avg_vol = sum(vols) / len(vols) if vols else None
        rows.append(
            {
                "date": ts.date().isoformat(),
                "regime_score": _round(score, 2),
                "regime_state": "Risk-On" if score >= 50.0 else "Risk-Off",
                "avg_rolling_volatility": _round(avg_vol),
            }
        )
    return rows[-CHART_LOOKBACK_WEEKS:]


def extract_symbol_view(market_regime: dict | None, symbol: str) -> dict | None:
    """Per-symbol response slice from the shared market-wide computation.
    Returns None if the market-wide computation is unavailable, or if
    `symbol` specifically didn't have enough history to be included.
    """
    if market_regime is None:
        return None
    per_symbol = market_regime["per_symbol"].get(symbol)
    if per_symbol is None:
        return None

    return {
        "symbol": symbol,
        "market_regime": market_regime["market_regime"],
        "regime_score": market_regime["regime_score"],
        "etf_universe": market_regime["etf_universe"],
        "etf_regime": per_symbol["etf_regime"],
        "cluster": per_symbol["cluster"],
        "adx": per_symbol["adx"],
        "rolling_volatility": per_symbol["rolling_volatility"],
        "downside_deviation": per_symbol["downside_deviation"],
        "scaled_return_vs_spy": per_symbol["scaled_return_vs_spy"],
        "dtw_distance_to_market": per_symbol["dtw_distance_to_market"],
        "chart_data": market_regime["chart_data"],
    }