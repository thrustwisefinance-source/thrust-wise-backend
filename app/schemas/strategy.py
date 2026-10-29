"""
Strategy Analytics response shapes.

Analytical / visualization data only — no signal, action, or
recommendation fields anywhere in this module.
"""

import datetime
from typing import Any, Literal

from pydantic import Field

from app.schemas.common import CamelModel


class Ema50RsiChartPoint(CamelModel):
    date: str
    close: float | None = None
    ema50: float | None = None
    rsi: float | None = None


class Ema8Ema21ChartPoint(CamelModel):
    date: str
    close: float | None = None
    ema8: float | None = None
    ema21: float | None = None


class Ema50RsiStrategy(CamelModel):
    price: float
    ema50: float
    rsi: float
    trend: str  # "Bullish" | "Bearish"
    price_above_ema50: bool
    distance_from_ema50_percent: float
    rsi_turning_up: bool
    pullback_active: bool
    chart_data: list[Ema50RsiChartPoint]


class Ema8Ema21Strategy(CamelModel):
    price: float
    ema8: float
    ema21: float
    trend: str  # "Bullish" | "Bearish"
    bullish_cross: bool
    bearish_cross: bool
    price_above_both_emas: bool
    distance_to_ema8_percent: float
    pullback_active: bool
    chart_data: list[Ema8Ema21ChartPoint]


class MacdChartPoint(CamelModel):
    date: str
    close: float | None = None
    macd: float | None = None
    signal: float | None = None
    histogram: float | None = None


class BollingerBandsChartPoint(CamelModel):
    date: str
    close: float | None = None
    sma20: float | None = None
    upper_band: float | None = None
    lower_band: float | None = None


class BetterBreakoutChartPoint(CamelModel):
    date: str
    close: float | None = None
    breakout_price: float | None = None
    sma40: float | None = None


class SmaTrendChartPoint(CamelModel):
    date: str
    close: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    sma100: float | None = None
    sma200: float | None = None


class MacdStrategy(CamelModel):
    price: float
    ema12: float
    ema26: float
    macd_line: float
    signal_line: float
    histogram: float
    trend: str  # "Bullish" | "Bearish"
    bullish_crossover: bool
    bearish_crossover: bool
    chart_data: list[MacdChartPoint]


class BollingerBandsStrategy(CamelModel):
    price: float
    sma20: float
    upper_band: float
    lower_band: float
    price_near_upper_band: bool
    price_near_lower_band: bool
    breakout_status: str  # "Above Upper Band" | "Below Lower Band" | "Within Bands"
    band_width_percent: float
    chart_data: list[BollingerBandsChartPoint]


class BetterBreakoutStrategy(CamelModel):
    price: float
    high_offset: float
    low_offset: float
    max_offset: float
    average_offset: float
    breakout_price: float
    sma40: float
    momentum: float
    breakout_active: bool
    chart_data: list[BetterBreakoutChartPoint]


class SmaTrendStrategy(CamelModel):
    price: float
    sma20: float
    sma50: float
    sma100: float
    sma200: float
    trend: str  # "Bullish" | "Bearish" | "Neutral"
    golden_cross: bool
    death_cross: bool
    price_above_sma200: bool
    chart_data: list[SmaTrendChartPoint]


class VtVsVtiVxusPerformancePoint(CamelModel):
    date: str
    vt_value: float
    vti_vxus_value: float


class VtVsVtiVxusStrategy(CamelModel):
    """VT vs VTI + VXUS Portfolio Comparison.

    A multi-ETF portfolio-analytics comparison (grows a hypothetical
    investment in 100% VT against 60% VTI + 40% VXUS) — not a technical
    indicator, so it intentionally carries none of the trend/crossover/
    signal-style fields the six strategies above expose. Surfaced under
    the VTI strategies response ("VTI tile") per product placement, but
    computed from VT and VXUS price history as well, not derived from
    VTI data alone.
    """

    strategy: str
    start_date: str
    end_date: str
    initial_investment: float
    vt_allocation: float
    vti_allocation: float
    vxus_allocation: float
    performance: list[VtVsVtiVxusPerformancePoint]


class EvarRiskChartPoint(CamelModel):
    date: str
    close: float | None = None
    evar_percent: float | None = None
    tail_risk_percentile: float | None = None


class EvarRiskStrategy(CamelModel):
    """EVaR Risk / Position Sizing.

    Entropic Value at Risk (EVaR), informed by Tsallis (q-exponential)
    entropy, estimates 1-day downside tail risk with more sensitivity to
    fat tails than a plain Gaussian VaR. This is a quantitative risk
    signal only — it does not predict crashes and does not guarantee any
    outcome. See services.strategies module docstring (Strategy 7) for
    the full formula, assumptions, and parameter choices.
    """

    price: float
    evar_percent: float
    tsallis_q: float
    confidence_level: float
    lookback_days: int
    tail_risk_percentile: float
    tail_risk_level: str  # "Low" | "Moderate" | "Elevated" | "High"
    risk_regime: str  # "Calm" | "Normal" | "Elevated" | "Stressed"
    suggested_exposure_percent: float
    chart_data: list[EvarRiskChartPoint]


class RiskOnRiskOffChartPoint(CamelModel):
    date: str
    regime_score: float | None = None
    regime_state: str | None = None  # "Risk-On" | "Risk-Off"
    avg_rolling_volatility: float | None = None


class RiskOnRiskOffStrategy(CamelModel):
    """Risk-On / Risk-Off Market Regime.

    Cross-asset strategy: analyzes the behavior of every ETF in the
    Thrustwise universe together (weekly technical features, standardized,
    compared via Dynamic Time Warping, grouped via clustering) to
    classify the overall market regime and this ETF's role within it.
    NOT a single-ETF technical indicator, and NOT a predictive or
    guaranteed signal. See services.regime module docstring for the full
    methodology and universe-size limitation.
    """

    symbol: str
    market_regime: str  # "Risk-On" | "Risk-Off"
    regime_score: float  # 0-100
    etf_universe: list[str]
    etf_regime: str  # "Risk-On" | "Risk-Off"
    cluster: int
    adx: float | None = None
    rolling_volatility: float | None = None
    downside_deviation: float | None = None
    scaled_return_vs_spy: float | None = None
    dtw_distance_to_market: float | None = None
    chart_data: list[RiskOnRiskOffChartPoint]


class TripleMaPullbackChartPoint(CamelModel):
    date: str
    close: float | None = None
    ma1: float | None = None
    ma2: float | None = None
    ma3: float | None = None
    entry_level: float | None = None
    exit_level: float | None = None


class TripleMaPullbackStrategy(CamelModel):
    """Triple-MA Pullback.

    Trend-confirmed mean-reversion strategy: an uptrend is confirmed via
    three nested SMAs (MA1 = SMA(close), MA2 = SMA(MA1), MA3 = SMA(MA2),
    with MA1 > MA2 > MA3 and close > MA3), then ATR(14)-based zones around
    the 5-day mean define a pullback entry level and an exit level. See
    services.strategies module docstring (Strategy 8) for the full
    formula and the documented period assumption.
    """

    price: float
    ma1: float
    ma2: float
    ma3: float
    mean5: float
    atr14: float
    entry_level: float
    exit_level: float
    trend: str  # "Uptrend" | "No Confirmed Uptrend"
    uptrend_confirmed: bool
    pullback_active: bool
    exit_zone_active: bool
    current_signal: str
    chart_data: list[TripleMaPullbackChartPoint]


class TltMonthlyCycleChartPoint(CamelModel):
    date: str
    close: float | None = None


class TltMonthlyCycleStrategy(CamelModel):
    """TLT Monthly Cycle.

    Calendar-based strategy that uses recurring monthly timing patterns
    (short near the start of the month, long shortly before month-end)
    rather than technical indicators. Computed from TLT's own price
    history — independent of whichever symbol's page is being viewed,
    same as Risk-On/Risk-Off is cross-ETF and shown on every symbol's
    response. Trading-day-of-month windows are approximated using
    business days (Mon-Fri); no exchange-holiday calendar is available,
    so dates near holidays may be off by a day. See services.strategies
    module docstring (Strategy 9) for the documented window assumptions.
    """

    price: float
    month: str  # "YYYY-MM"
    current_phase: str
    current_position: str  # "Short" | "Long" | "Flat"
    short_entry_date: str
    short_exit_date: str
    long_entry_date: str
    long_exit_date: str
    next_expected_action: str
    next_expected_date: str
    chart_data: list[TltMonthlyCycleChartPoint]


class TqqqTmfIefChartPoint(CamelModel):
    date: str
    tqqq: float | None = None
    tmf: float | None = None
    ief: float | None = None
    portfolio_value: float | None = None
    state: str | None = None  # "Normal" | "Defensive"
    is_rebalance_event: bool = False
    is_crash_event: bool = False
    is_recovery_event: bool = False


class TqqqTmfIefRebalancingStrategy(CamelModel):
    """TQQQ / TMF / IEF Rebalancing (Portfolio / Allocation strategy).

    Multi-ETF portfolio-allocation and crash-defense strategy — not a
    single-symbol technical indicator, so (like VtVsVtiVxusStrategy) it
    intentionally carries none of the trend/crossover/signal-style fields
    the technical strategies expose. Normal state holds 50% TQQQ / 50%
    TMF, rebalanced every two months; a single-day TQQQ decline of 20% or
    more moves the full portfolio into IEF until TQQQ's close exceeds its
    pre-crash reference price. See services.tqqq_tmf_ief_strategy module
    docstring for the full state-machine and simulation methodology.

    `backtest_*` fields are ThrustWise's own calculation from ThrustWise's
    own ingested price history (see `backtest_label`) — never the source
    article's reported performance figures.
    """

    strategy: str
    state: str  # "Normal" | "Defensive / Crash"
    tqqq_allocation_percent: float
    tmf_allocation_percent: float
    ief_allocation_percent: float
    tqqq_price: float
    tmf_price: float
    ief_price: float
    tqqq_daily_return_percent: float
    last_rebalance_date: str
    next_rebalance_date: str
    rebalance_frequency_months: int
    crash_filter_status: str  # "Inactive" | "Triggered"
    crash_filter_threshold_percent: float
    crash_trigger_date: str | None = None
    pre_crash_tqqq_price: float | None = None
    recovery_status: str | None = None
    next_expected_action: str
    backtest_start_date: str
    backtest_end_date: str
    backtest_initial_investment: float
    backtest_final_value: float
    backtest_total_return_percent: float
    backtest_cagr_percent: float
    backtest_max_drawdown_percent: float
    backtest_label: str
    disclaimer: str
    chart_data: list[TqqqTmfIefChartPoint]


class MswingChartPoint(CamelModel):
    date: str
    close: float | None = None
    mswing: float | None = None
    mswing_ema9: float | None = None
    sma50: float | None = None


class MswingStrategy(CamelModel):
    """Mswing Momentum.

    Single-symbol momentum strategy: Mswing = 20-day + 50-day close
    percent-change, confirmed by its own 9-period EMA and by price
    relative to SMA50. Descriptive momentum state and default
    entry/exit conditions only — no buy/sell signal, matching every
    other strategy in this module. See services.strategies module
    docstring (Strategy 12) for the full formula, the four-state
    classification, and the documented Mswing == 0 boundary assumption.

    `relative_strength` is Mswing(this symbol) - Mswing(QQQ), reusing
    the existing ETF price infrastructure as the index (see
    services.strategies MSWING_INDEX_SYMBOL) — null when the index
    reading itself is unavailable rather than fabricated.
    """

    price: float
    mswing: float
    mswing_ema9: float
    sma50: float
    mswing_state: str  # "Strong bullish momentum" | "Weakening bullish momentum" |
    # "Recovering bearish momentum" | "Bearish momentum"
    mswing_above_zero: bool
    mswing_above_ema: bool
    price_above_sma50: bool
    zero_line_bullish_cross: bool
    zero_line_bearish_cross: bool
    ema_bullish_cross: bool
    ema_bearish_cross: bool
    bullish_condition_active: bool
    bearish_exit_condition_active: bool
    relative_strength: float | None = None
    relative_strength_index_symbol: str
    chart_data: list[MswingChartPoint]


class MomentumReversalChartPoint(CamelModel):
    date: str
    close: float | None = None
    daily_return_percent: float | None = None
    position: str | None = None  # "Long" | "Short" | "Flat"


class MomentumReversalStrategy(CamelModel):
    """Momentum Reversal.

    Contrarian single-symbol strategy: close.pct_change() below -0.1%
    flags a Bullish Reversal, above +0.1% flags a Bearish Reversal,
    otherwise the previous position is retained. The source article
    demonstrates this on 1-minute TSLA bars; this codebase only ingests
    daily bars (see app.models.DailyPrice), so the identical rule and
    thresholds are applied to daily close-to-close returns instead of
    introducing a new intraday data pipeline. See services.strategies
    module docstring (Strategy 13) for the full data-resolution caveat.
    Descriptive/analytical only — not a buy/sell recommendation.
    """

    price: float
    daily_return_percent: float
    buy_threshold_percent: float
    sell_threshold_percent: float
    reversal_signal: str  # "Bullish Reversal" | "Bearish Reversal" | "No Signal"
    current_position: str  # "Long" | "Short" | "Flat"
    position_changed_today: bool
    chart_data: list[MomentumReversalChartPoint]


class FactorRanking(CamelModel):
    factor: str
    one_month_return_percent: float
    rank: int


class FactorMomentumChartPoint(CamelModel):
    date: str
    close: float | None = None
    one_month_return_percent: float | None = None


class FactorMomentumStrategy(CamelModel):
    """One-Month Factor Momentum.

    Cross-ETF strategy: ranks "factors" by their previous ONE-MONTH
    (not the traditional 12-month) average return, then reads whether
    this symbol is aligned with the leading or lagging factor.
    `factor_data_basis` documents exactly what stands in for factor data
    here — this codebase has no standalone style-factor (Value/Quality/
    Low-Vol) index data, so the existing ETF_REGISTRY category tags
    (Equity, Index, Technology, Gold, Treasury — the same tags used by
    the ETF Explorer's filter chips) are reused as the factor grouping.
    See services.strategies module docstring (Strategy 14) for the full
    methodology and this limitation. Descriptive/analytical only — not a
    buy/sell recommendation.
    """

    symbol: str
    symbol_tags: list[str]
    one_month_return_percent: float
    factor_momentum_score: float
    factor_rankings: list[FactorRanking]
    leading_factor: str
    lagging_factor: str
    is_aligned_with_leading_factor: bool
    exposure_guidance: str  # "Maintain / Increase Exposure" | "Neutral" | "Reduce Exposure"
    suggested_exposure_percent: float
    etf_universe: list[str]
    lookback_trading_days: int
    factor_data_basis: str
    chart_data: list[FactorMomentumChartPoint]


class HmmRegimeSwitchingChartPoint(CamelModel):
    date: str
    close: float | None = None
    log_return_percent: float | None = None
    trending_probability_percent: float | None = None
    regime: str | None = None  # "Trending" | "High-Volatility"


class HmmRegimeSwitchingStrategy(CamelModel):
    """HMM Regime-Switching.

    Single-symbol strategy: a 2-state Gaussian Hidden Markov Model is
    fit on daily log returns (Baum-Welch / forward-backward, implemented
    directly with numpy — no HMM library is a dependency of this
    project). The lower-variance fitted state is read as "Trending /
    Momentum-Favorable", the higher-variance state as "High-Volatility /
    Mean-Reversion-Favorable"; `recommended_approach` reflects that
    regime-to-approach switch described in the source article. See
    services.strategies module docstring (Strategy 15) for the full
    model and assumptions. Descriptive/analytical only — not a buy/sell
    recommendation.
    """

    price: float
    current_regime: str  # "Trending / Momentum-Favorable" | "High-Volatility / Mean-Reversion-Favorable"
    recommended_approach: str  # "Momentum" | "Mean Reversion"
    trending_probability_percent: float
    high_volatility_probability_percent: float
    regime_persistence_probability_percent: float
    regime_changed_today: bool
    trending_state_mean_return_percent: float
    trending_state_volatility_percent: float
    high_vol_state_mean_return_percent: float
    high_vol_state_volatility_percent: float
    lookback_days: int
    chart_data: list[HmmRegimeSwitchingChartPoint]


class SqueezeMomentumChartPoint(CamelModel):
    date: str
    close: float | None = None
    upper_bb: float | None = None
    lower_bb: float | None = None
    upper_kc: float | None = None
    lower_kc: float | None = None
    momentum: float | None = None
    squeeze_state: str | None = None  # "on" | "off" | "none"


class SqueezeMomentumStrategy(CamelModel):
    """Squeeze Momentum Indicator.

    Bollinger Bands vs. Keltner Channels volatility-compression
    ("squeeze") detector plus a rolling linear-regression momentum
    oscillator. `momentum` is the LAST FITTED VALUE of the regression
    line (not the slope) — see services.strategies module docstring
    (Strategy 16) for the full formula and the article's
    squeeze_on/squeeze_off definitions. Descriptive/analytical only —
    the article's entry-on-release / exit-on-squeeze logic is exposed
    as `squeeze_state` / `momentum_direction` to read, never as an
    executed or recommended trade.
    """

    price: float
    upper_bb: float
    lower_bb: float
    upper_kc: float
    lower_kc: float
    momentum: float
    momentum_direction: str  # "Rising" | "Falling" | "Flat"
    momentum_state: str
    squeeze_state: str  # "Squeeze On" | "Squeeze Released" | "No Squeeze"
    squeeze_on: bool
    squeeze_off: bool
    squeeze_released_today: bool
    bb_length: int
    bb_mult: float
    kc_mult: float
    chart_data: list[SqueezeMomentumChartPoint]


class SocAvalancheChartPoint(CamelModel):
    """One point of the log-log avalanche-size histogram (not a
    date-indexed time series — see services.strategies Strategy 17)."""

    size: float | None = None
    density: float | None = None


class SocAvalancheHistoryPoint(CamelModel):
    avalanche_index: int
    size_percent: float | None = None


class SocAvalancheStrategy(CamelModel):
    """Self-Organized Criticality — Avalanche Distribution.

    Fits a power-law exponent (alpha) to the distribution of drawdown
    "avalanche" sizes (see services.strategies module docstring,
    Strategy 17, for the exact avalanche definition and the log-log
    fitting methodology — documented explicitly since the source
    material does not specify one). Descriptive only: does not predict
    crashes and does not guarantee any future outcome.
    """

    price: float
    alpha: float
    criticality_regime: str  # "Gaussian" | "Transitional" | "Critical" | "Super-critical"
    r_squared: float | None = None
    avalanche_count: int
    mean_avalanche_size_percent: float
    max_avalanche_size_percent: float
    current_drawdown_percent: float
    log_bins_used: int
    chart_data: list[SocAvalancheChartPoint]
    avalanche_history: list[SocAvalancheHistoryPoint]
    methodology_note: str


class WaveletTrendChartPoint(CamelModel):
    date: str
    close: float | None = None
    trend_short: float | None = None
    trend_medium: float | None = None
    trend_long: float | None = None


class WaveletTrendStrategy(CamelModel):
    """Adaptive Causal Wavelet Trend Filter.

    Causal (no look-ahead) multi-resolution Mexican Hat/Ricker wavelet
    trend filter, volatility-adjusted. See services.strategies module
    docstring (Strategy 18) for the exact causal-truncation methodology
    and the three time scales. Descriptive only — not a buy/sell
    recommendation.
    """

    price: float
    current_trend_state: str
    wavelet_trend_value: float
    trend_short: float
    trend_medium: float
    trend_long: float
    volatility_adjustment_percent: float | None = None
    scales_days: dict[str, int]
    chart_data: list[WaveletTrendChartPoint]


class FirstPassageTimeChartPoint(CamelModel):
    date: str
    trading_day: int
    upside_hit_probability_percent: float | None = None
    downside_hit_probability_percent: float | None = None


class FirstPassageTimeStrategy(CamelModel):
    """First Passage Time Distribution Analysis.

    Closed-form first-passage-time (inverse-Gaussian) probability and
    expected-time-to-target estimates under a Brownian-motion-with-drift
    approximation of daily log returns. See services.strategies module
    docstring (Strategy 19) for the full formula and documented
    assumptions. Descriptive/probabilistic only — never a certainty.
    """

    price: float
    mean_daily_log_return_percent: float
    daily_volatility_percent: float
    horizon_trading_days: int
    upside_target_percent: float
    downside_target_percent: float
    upside_target_price: float
    downside_target_price: float
    upside_target_probability_percent: float | None = None
    downside_target_probability_percent: float | None = None
    expected_days_to_upside_target: float | None = None
    expected_days_to_downside_target: float | None = None
    lookback_days: int
    assumptions_note: str
    chart_data: list[FirstPassageTimeChartPoint]


class PortfolioAssetWeight(CamelModel):
    symbol: str
    weight_percent: float


class PortfolioSummary(CamelModel):
    weights: list[PortfolioAssetWeight]
    expected_return_percent: float
    volatility_percent: float
    sharpe_ratio: float


class EfficientFrontierPoint(CamelModel):
    target_return_percent: float
    expected_return_percent: float
    volatility_percent: float
    sharpe_ratio: float
    weights: list[PortfolioAssetWeight]


class KellySizing(CamelModel):
    symbol: str
    single_asset_kelly_fraction_percent: float
    portfolio_kelly_weight_percent: float
    half_kelly_weight_percent: float


class PortfolioOptimizationRequest(CamelModel):
    """POST body for /api/portfolio/optimization."""

    symbols: list[str]
    risk_free_rate_percent: float = 0.0


class PortfolioOptimizationResult(CamelModel):
    """Kelly Criterion + Mean-Variance Optimization across a basket of
    symbols — genuinely multi-asset, unlike every other strategy in
    this module (each of which is a function of one symbol's own price
    history). See services.strategies.compute_portfolio_optimization
    for the full methodology, assumptions, and the documented long-only
    approximation. Descriptive/analytical only — not a guaranteed-
    optimal allocation and not investment advice.
    """

    symbols: list[str]
    lookback_trading_days: int
    risk_free_rate_percent: float
    min_volatility_portfolio: PortfolioSummary
    max_sharpe_portfolio: PortfolioSummary
    efficient_frontier: list[EfficientFrontierPoint]
    kelly_sizing: list[KellySizing]
    long_only_note: str
    disclaimer: str


# ---------------------------------------------------------------------------
# Strategies 20-26: additional Technical Strategies. All are single-symbol,
# computed from the symbol's own daily OHLC history, and analytical only
# (state / condition flags - no action field). See the matching section in
# services.strategies for formulas, assumptions and exactness notes.
# ---------------------------------------------------------------------------


class TdSetupChartPoint(CamelModel):
    date: str
    close: float | None = None
    bearish_setup_count: int | None = None
    bullish_setup_count: int | None = None


class TdSetupStrategy(CamelModel):
    """TD Setup / TD 9 (Tom DeMark Sequential, Part I only).

    Bearish setup: Close[t] > Close[t-4] for consecutive candles; bullish
    setup: Close[t] < Close[t-4]. A completed count of 9 is a POTENTIAL
    exhaustion area - not a reversal guarantee and not a buy/sell
    instruction. The Countdown phase is not implemented. Any interruption
    resets the count to 0. See services.strategies Strategy 20.
    """

    price: float
    setup_type: str  # "Bearish" | "Bullish" | "None"
    setup_count: int  # 0-9
    setup_completed: bool
    exhaustion_state: str  # "Potential Upside Exhaustion" | "Potential Downside Exhaustion" |
    # "Setup In Progress" | "No Active Setup"
    consecutive_qualifying_candles: int
    bars_since_completion: int | None = None
    comparison_lag_candles: int
    setup_length: int
    last_completed_setup_type: str | None = None
    last_completed_setup_date: str | None = None
    chart_data: list[TdSetupChartPoint]


class SupertrendConfirmationChartPoint(CamelModel):
    date: str
    close: float | None = None
    supertrend: float | None = None
    confirmation_state: str | None = None  # "Buy Confirmed" | "Sell Confirmed" | "No Confirmation"


class SupertrendConfirmationStrategy(CamelModel):
    """Supertrend Confirmation.

    Buy condition: the two candles before the signal candle are green,
    close is above the Supertrend and the Supertrend is bullish. Sell
    condition is the mirror image. A bare Supertrend direction change is
    never reported as a condition. Descriptive condition flags only - not
    a recommendation. Supertrend parameters (ATR 10 x 3.0) are an
    assumption because the source states none. See services.strategies
    Strategy 21.
    """

    price: float
    supertrend: float
    supertrend_trend: str  # "Bullish" | "Bearish"
    price_above_supertrend: bool
    supertrend_flipped_today: bool
    previous_two_candles: list[str]  # oldest first; "Green" | "Red" | "Neutral"
    buy_condition_active: bool
    sell_condition_active: bool
    confirmation_state: str  # "Buy Confirmed" | "Sell Confirmed" | "No Confirmation"
    last_buy_condition_date: str | None = None
    last_sell_condition_date: str | None = None
    confirmation_candles: int
    atr_period: int
    multiplier: float
    parameters_note: str
    chart_data: list[SupertrendConfirmationChartPoint]


class UfxVervoortChartPoint(CamelModel):
    date: str
    close: float | None = None
    ufx_stand_in_band: float | None = None
    setup_state: str | None = None  # "Buy Setup" | "Sell Setup" | "No Setup"


class UfxVervoortHeikenAshiStrategy(CamelModel):
    """UFX + Vervoort Heiken Ashi.

    NOT an exact reproduction: UFX Pro and Vervoort Heiken Ashi (Long
    Term) are proprietary TradingView scripts. The article's rule
    structure is evaluated with labelled public-formula stand-ins
    (ATR(6) x 2 Supertrend-style band; standard Heiken Ashi colour) -
    see `ufx_pro_exact`, `vervoort_heiken_ashi_exact`,
    `unmapped_source_settings` and `implementation_note`. Risk/reward is
    1 : 1.5. Descriptive only. See services.strategies Strategy 22.
    """

    price: float
    setup_state: str  # "Buy Setup" | "Sell Setup" | "No Setup"
    buy_setup_active: bool
    sell_setup_active: bool
    ufx_stand_in_trend: str  # "Bullish" | "Bearish"
    ufx_stand_in_flipped_today: bool
    ufx_stand_in_band: float
    heiken_ashi_histogram_stand_in: str  # "Green" | "Red" | "Neutral"
    candle_confirmation: str  # "Bullish" | "Bearish" | "Neutral"
    last_setup_type: str | None = None  # "Buy" | "Sell"
    last_setup_date: str | None = None
    bars_since_last_setup: int | None = None
    last_setup_entry: float | None = None
    last_setup_stop: float | None = None
    last_setup_target: float | None = None
    risk_reward_ratio: float
    ufx_atr_period: int
    ufx_factor: float
    ufx_pro_exact: bool
    vervoort_heiken_ashi_exact: bool
    unmapped_source_settings: list[str]
    implementation_note: str
    chart_data: list[UfxVervoortChartPoint]


class TrendSignalsTrendTideChartPoint(CamelModel):
    date: str
    close: float | None = None
    candle_colour: str | None = None  # "Green" | "Red" | "Neutral"


class TrendSignalsTrendTideStrategy(CamelModel):
    """Trend Signals + Trend Tide Oscillator.

    The source gives no formula or parameters for either indicator, so
    neither is reproduced (`trend_signals_available` /
    `trend_tide_available` are False and the setup flags cannot become
    True). Only the price-confirmation candle and the 1 : 1.5 risk/reward
    framework are reported. See services.strategies Strategy 23.
    """

    price: float
    candle_confirmation: str  # "Bullish" | "Bearish" | "Neutral"
    trend_signals_available: bool
    trend_tide_available: bool
    trend_signals_state: str
    trend_tide_state: str
    buy_setup_active: bool
    sell_setup_active: bool
    setup_state: str
    risk_reward_ratio: float
    implementation_note: str
    chart_data: list[TrendSignalsTrendTideChartPoint]


class PivotLevelsChartPoint(CamelModel):
    date: str
    close: float | None = None
    pivot: float | None = None
    r1: float | None = None
    r2: float | None = None
    r3: float | None = None
    s1: float | None = None
    s2: float | None = None
    s3: float | None = None


class PivotLevelsStrategy(CamelModel):
    """Pivot Levels (classic floor-trader pivots from the previous daily
    candle) with the source's upper / middle / lower zone reading. Zone
    context labels are descriptive only - no buy/sell rule is derived.
    See services.strategies Strategy 24.
    """

    price: float
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    price_above_pivot: bool
    zone: str  # "Upper Zone" | "Middle Zone" | "Lower Zone"
    zone_context: str
    nearest_resistance_level: str | None = None
    nearest_resistance: float | None = None
    distance_to_resistance_percent: float | None = None
    nearest_support_level: str | None = None
    nearest_support: float | None = None
    distance_to_support_percent: float | None = None
    basis: str
    chart_data: list[PivotLevelsChartPoint]


class CprChartPoint(CamelModel):
    date: str
    close: float | None = None
    pivot: float | None = None
    top_cpr: float | None = None
    bottom_cpr: float | None = None


class CprStrategy(CamelModel):
    """CPR - Central Pivot Range from the previous daily candle:
    Pivot = (H+L+C)/3, BC = (H+L)/2, TC = 2*Pivot - BC. Narrow / wide is
    ranked against the symbol's own trailing CPR widths (the source gives
    no numeric threshold). `market_bias` describes price vs the CPR - not
    a buy/sell rule. See services.strategies Strategy 25.
    """

    price: float
    pivot: float
    top_cpr: float
    bottom_cpr: float
    cpr_width: float
    cpr_width_percent: float
    cpr_width_percentile_rank: float  # 0-100, mid-rank vs trailing widths
    cpr_width_class: str  # "Narrow" | "Normal" | "Wide"
    cpr_width_interpretation: str
    price_position: str  # "Above CPR" | "Below CPR" | "Inside CPR"
    market_bias: str  # "Bullish Day Bias" | "Bearish Day Bias" | "Neutral"
    cpr_support: float | None = None
    cpr_resistance: float | None = None
    width_lookback_days: int
    basis: str
    chart_data: list[CprChartPoint]


class TaCandlesTrendHealthChartPoint(CamelModel):
    date: str
    close: float | None = None
    ema21: float | None = None
    ema50: float | None = None
    roc: float | None = None
    atr_percent: float | None = None
    green_ratio: float | None = None


class TaCandlesTrendHealthStrategy(CamelModel):
    """TA Candles / Trend Health.

    NOT the proprietary TradingView TA Candles / MasterPiece indicator
    (`is_exact_tradingview_indicator` is False): ThrustWise's own
    transparent trend / momentum / volatility / candle-consistency read
    built from standard measures. See services.strategies Strategy 26.
    """

    price: float
    ema21: float
    ema50: float
    trend: str  # "Bullish" | "Bearish" | "Mixed"
    momentum_roc_percent: float
    momentum_state: str  # "Positive" | "Negative" | "Flat"
    atr_percent: float
    volatility_state: str  # "Expanding" | "Contracting" | "Normal"
    green_candle_ratio: float
    red_candle_ratio: float
    current_candle_streak: int
    current_candle_streak_colour: str
    strength_alignment: str  # "Aligned Bullish" | "Aligned Bearish" | "Mixed"
    trend_health_state: str
    strength_window_candles: int
    is_exact_tradingview_indicator: bool
    methodology_note: str
    chart_data: list[TaCandlesTrendHealthChartPoint]


# ---------------------------------------------------------------------------
# Strategies 27-28: fundamental-valuation QUANTITATIVE strategies. Unlike the
# strategies above these are not functions of the requested ETF's price
# history: Graham valuation is per-stock, the Magic Formula is a ranking over
# a stock universe. Both always return an object whose `status` is
# "available" or "unavailable" (never silently null). Analytical only - no
# buy/sell field. See services.strategies Strategies 27-28.
# ---------------------------------------------------------------------------


class GrahamSensitivityScenario(CamelModel):
    scenario: str
    growth_percent: float | None = None
    estimated_value: float | None = None
    upside_downside_percent: float | None = None


class GrahamValuationStrategy(CamelModel):
    """Graham growth-valuation estimate: EPS x (8.5 + 2g) x (4.4 / Y).

    `g` and `Y` are percent-unit numbers (8 = 8%). Growth is capped at
    `max_growth_percent` (raw value kept in `growth_percent_raw`).
    `growth_basis` says whether growth is a forward analyst estimate or a
    labelled historical-CAGR proxy. Not an intrinsic-value guarantee and not
    a price prediction. Per-stock: for an ETF symbol (or a stock without
    usable inputs) `status` is "unavailable" with `unavailable_reasons`.
    """

    status: str  # "available" | "unavailable"
    unavailable_reasons: list[str] = Field(default_factory=list)
    symbol: str | None = None
    estimated_value: float | None = None
    eps: float | None = None
    eps_period: str | None = None
    eps_basis: str | None = None
    growth_percent_raw: float | None = None
    growth_percent_applied: float | None = None
    growth_basis: str | None = None  # "forward_analyst_consensus" | "historical_eps_cagr_proxy"
    growth_source: str | None = None
    growth_horizon: str | None = None
    max_growth_percent: float | None = None
    aaa_yield_percent: float | None = None
    aaa_yield_source: str | None = None
    aaa_yield_as_of: str | None = None
    baseline_yield_percent: float | None = None
    formula: str | None = None
    market_price: float | None = None
    market_price_as_of: str | None = None
    difference_abs: float | None = None
    upside_downside_percent: float | None = None
    difference_definition: str | None = None
    margin_of_safety_percent: float | None = None
    margin_of_safety_price: float | None = None
    price_within_margin_of_safety: bool | None = None
    sensitivity: list[GrahamSensitivityScenario] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str | None = None


class MagicFormulaRow(CamelModel):
    """One stock's Magic Formula factors and ranks. Percent fields are
    percent units (12.5 = 12.5%). Ranks are null unless `ranked` is true;
    `exclusion_reason` says why a stock was not ranked."""

    symbol: str | None = None
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
    ebit: float | None = None
    ebit_period: str | None = None
    ebit_basis: str | None = None
    enterprise_value: float | None = None
    enterprise_value_source: str | None = None
    earnings_yield_percent: float | None = None
    earnings_yield_status: str | None = None
    earnings_yield_unavailable_reason: str | None = None
    net_working_capital: float | None = None
    net_fixed_assets: float | None = None
    invested_capital: float | None = None
    balance_sheet_period: str | None = None
    return_on_capital_percent: float | None = None
    return_on_capital_status: str | None = None
    return_on_capital_unavailable_reason: str | None = None
    earnings_yield_rank: int | None = None
    return_on_capital_rank: int | None = None
    combined_rank: int | None = None
    overall_position: int | None = None
    ranked: bool = False
    exclusion_reason: str | None = None
    data_quality_warnings: list[str] = Field(default_factory=list)


class GreenblattMagicFormulaStrategy(CamelModel):
    """Greenblatt Magic Formula ranking over a stock universe (cross-
    sectional, like Factor Momentum it is computed once and cached, and
    shown on every symbol's response). `top_ranked` holds the best
    `top_n` rows by combined rank (lower is better); `requested_symbol_result`
    is the requested stock's own row when the request was for a stock in the
    universe. A relative ranking only - not an absolute quality guarantee.
    """

    status: str  # "available" | "unavailable"
    unavailable_reasons: list[str] = Field(default_factory=list)
    universe: str | None = None
    universe_label: str | None = None
    universe_size: int | None = None
    eligible_universe_definition: str | None = None
    ranked_count: int | None = None
    excluded_count: int | None = None
    excluded_by_reason: dict[str, int] = Field(default_factory=dict)
    top_n: int | None = None
    top_ranked: list[MagicFormulaRow] = Field(default_factory=list)
    requested_symbol_result: MagicFormulaRow | None = None
    methodology: list[str] = Field(default_factory=list)
    data_updated_at: str | None = None
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str | None = None


# ---------------------------------------------------------------------------
# Strategies 29-32: article-based additions (see the matching section in
# services.strategies for methodology, assumptions and limitations):
#   29 Seasonal Investment Performance Heat Map   (calendar group)
#   30 Investment Diversification Heat Map        (portfolio; POST route)
#   31 EMA + Bollinger Bands + ATR backtest       (technical group)
#   32 Technical Crash-Warning Risk Indicator     (quantitative group)
# All additive. Analytical / historical-simulation data only - no order,
# signal or recommendation fields.
# ---------------------------------------------------------------------------


class HeatMapColorScale(CamelModel):
    metric: str
    neutral: float
    max_abs_return_percent: float | None = None
    note: str


class SeasonalMonthCell(CamelModel):
    year: int
    month: int
    label: str
    return_percent: float | None = None
    partial_return_percent: float | None = None
    sign: str | None = None
    # ok | partial_month | no_base | prior_month_missing | missing | out_of_range
    status: str
    base_date: str | None = None
    end_date: str | None = None


class SeasonalQuarterCell(CamelModel):
    year: int
    quarter: int
    label: str
    return_percent: float | None = None  # only when status == "complete"
    partial_return_percent: float | None = None  # linked leading months (quarter-to-date)
    sign: str | None = None
    status: str  # complete | partial | missing | out_of_range
    months_included: int = 0


class SeasonalCalendarYear(CamelModel):
    return_percent: float | None = None
    sign: str | None = None
    status: str  # complete | partial | missing


class SeasonalYearRow(CamelModel):
    year: int
    months: list[SeasonalMonthCell]
    quarters: list[SeasonalQuarterCell]
    calendar_year: SeasonalCalendarYear


class PeriodSummary(CamelModel):
    month: int | None = None
    quarter: int | None = None
    label: str
    count: int
    average_return_percent: float | None = None
    median_return_percent: float | None = None
    positive_count: int = 0
    positive_rate_percent: float | None = None
    best_return_percent: float | None = None
    worst_return_percent: float | None = None


class SeasonalHeatMapStrategy(CamelModel):
    symbol: str | None = None
    price_basis: str
    adjustment_policy: str
    requested_start: str | None = None
    requested_end: str | None = None
    data_first_date: str
    data_last_date: str
    as_of: str
    month_labels: list[str]
    quarter_labels: list[str]
    rows: list[SeasonalYearRow]
    monthly_summary: list[PeriodSummary]
    quarterly_summary: list[PeriodSummary]
    color_scale: HeatMapColorScale
    methodology: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class DiversificationCell(CamelModel):
    year: int
    return_percent: float | None = None
    partial_return_percent: float | None = None
    sign: str | None = None
    # complete | partial | no_base | prior_year_missing | not_listed | missing
    # (portfolio cells also: incomplete | no_weights)
    status: str
    rank: int | None = None
    rank_among: int | None = None
    window_start: str | None = None
    window_end: str | None = None
    reason: str | None = None


class YearReturn(CamelModel):
    year: int
    return_percent: float | None = None


class DiversificationAsset(CamelModel):
    symbol: str
    weight_percent: float | None = None
    first_date: str | None = None
    last_date: str | None = None
    cells: list[DiversificationCell]
    complete_years: int
    average_annual_return_percent: float | None = None
    best_year: YearReturn | None = None
    worst_year: YearReturn | None = None


class DiversificationPortfolio(CamelModel):
    weights_percent: dict[str, float | None]
    rebalancing_policy: str
    policy_description: str
    cells: list[DiversificationCell]


class DiversificationRankEntry(CamelModel):
    rank: int
    symbol: str
    is_portfolio: bool
    return_percent: float | None = None


class DiversificationYearRanking(CamelModel):
    year: int
    complete_asset_count: int
    ranking: list[DiversificationRankEntry]


class DiversificationHeatMapStrategy(CamelModel):
    symbols: list[str]
    price_basis: str
    adjustment_policy: str
    requested_start: str | None = None
    requested_end: str | None = None
    years: list[int]
    as_of: str
    assets: list[DiversificationAsset]
    portfolio: DiversificationPortfolio | None = None
    yearly_rankings: list[DiversificationYearRanking]
    color_scale: HeatMapColorScale
    methodology: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class DiversificationHeatMapRequest(CamelModel):
    """POST body for /api/portfolio/diversification_heat_map.

    `weights_percent` (optional) maps ticker -> percent weight: long-only,
    finite, summing to 100 (+/-0.01); omitted tickers get 0%. Without it
    the heat map is returned without a portfolio row. `start`/`end` have
    calendar-year granularity."""

    symbols: list[str]
    weights_percent: dict[str, float] | None = None
    start: datetime.date | None = None
    end: datetime.date | None = None
    price_basis: Literal["adjusted", "close"] = "adjusted"


class EbaSegmentMetrics(CamelModel):
    segment: str  # train | out_of_sample
    start: str
    end: str
    bars: int
    total_return_percent: float | None = None
    cagr_percent: float | None = None
    max_drawdown_percent: float | None = None
    trade_count: int
    long_trades: int
    short_trades: int
    win_rate_percent: float | None = None
    profit_factor: float | None = None
    profit_factor_note: str | None = None
    average_trade_percent: float | None = None
    exposure_percent: float | None = None
    ambiguous_bars: int = 0
    benchmark_buy_hold_return_percent: float | None = None
    excess_return_vs_benchmark_percent: float | None = None


class EbaTrade(CamelModel):
    side: str  # long | short  (historical simulated trade)
    segment: str
    signal_date: str
    entry_date: str
    entry_price: float | None = None
    exit_date: str
    exit_price: float | None = None
    exit_reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    atr_at_signal: float | None = None
    price_return_percent: float | None = None
    equity_return_percent: float | None = None
    bars_held: int


class EbaEquityPoint(CamelModel):
    date: str
    segment: str
    equity: float | None = None
    benchmark: float | None = None


class EbaChartPoint(CamelModel):
    date: str
    close: float | None = None
    ema: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    rsi: float | None = None
    atr: float | None = None


class EbaLatest(CamelModel):
    date: str
    close: float | None = None
    ema: float | None = None
    bb_upper: float | None = None
    bb_lower: float | None = None
    rsi: float | None = None
    atr: float | None = None
    trend: str  # Uptrend | Downtrend | Neutral
    long_setup_active: bool
    short_setup_active: bool
    reference_stop_distance: float | None = None
    reference_take_profit_distance: float | None = None
    note: str


class EbaParameterCandidate(CamelModel):
    stop_loss_atr_mult: float
    take_profit_ratio: float
    train_total_return_percent: float | None = None
    train_trade_count: int
    train_max_drawdown_percent: float | None = None


class EbaSelectedParameters(CamelModel):
    stop_loss_atr_mult: float
    take_profit_ratio: float


class EbaParameterSelection(CamelModel):
    enabled: bool
    objective: str
    candidates: list[EbaParameterCandidate] = Field(default_factory=list)
    selected: EbaSelectedParameters | None = None
    note: str


class EbaParameters(CamelModel):
    """Effective parameters of the run (article defaults unless overridden)."""

    ema_period: int
    bb_period: int
    bb_std: float
    rsi_period: int
    atr_period: int
    trend_lookback: int
    stop_loss_atr_mult: float
    take_profit_ratio: float
    rsi_long_exit: float
    rsi_short_exit: float
    commission_percent: float
    slippage_percent: float
    position_fraction: float
    train_fraction: float
    allow_short: bool


class EmaBollingerAtrStrategy(CamelModel):
    """EMA(30) trend + Bollinger(15, 1.5) entry conditions + ATR(7) risk
    levels, with a historical train / out-of-sample backtest. Analytics
    only: no orders, no broker connection, no profitability claim."""

    timeframe: str  # 1d | 1h
    data_source: str
    bars_used: int
    first_bar: str
    last_bar: str
    parameters: EbaParameters
    latest: EbaLatest
    train_fraction: float
    train: EbaSegmentMetrics
    out_of_sample: EbaSegmentMetrics
    parameter_selection: EbaParameterSelection
    trade_count_total: int
    trades: list[EbaTrade]
    equity_curve: list[EbaEquityPoint]
    chart_data: list[EbaChartPoint]
    assumptions: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class CrashCondition(CamelModel):
    id: str
    label: str
    rule: str
    scored: bool
    triggered: bool
    triggered_since: str | None = None
    observed_at: str
    note: str | None = None


class CrashIndicators(CamelModel):
    close: float | None = None
    sma_short: float | None = None
    sma_long: float | None = None
    rsi: float | None = None
    macd: float | None = None
    signal: float | None = None
    histogram: float | None = None
    drawdown_from_252d_high_percent: float | None = None


class CrashDivergence(CamelModel):
    first_swing_date: str
    first_swing_close: float | None = None
    first_swing_rsi: float | None = None
    second_swing_date: str
    second_swing_close: float | None = None
    second_swing_rsi: float | None = None
    confirmed_on: str


class CrashClassification(CamelModel):
    bars: int
    crash_labelled_bars: int
    base_rate_percent: float | None = None
    warning_bars: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision_percent: float | None = None
    recall_percent: float | None = None
    false_positive_rate_percent: float | None = None
    episodes: int
    episodes_detected: int
    episodes_missed: int
    median_lead_bars: float | None = None
    precision_lift_vs_base_rate: float | None = None


class CrashEvaluationBlock(CamelModel):
    start: str
    end: str
    indicator: CrashClassification
    baseline_sma200: CrashClassification
    always_warn_precision_percent: float | None = None


class CrashValidationDefinition(CamelModel):
    crash_event: str
    prediction_horizon_bars: int
    warning_rule: str
    baseline_rule: str
    split: str
    note: str


class CrashParameters(CamelModel):
    sma_short: int
    sma_long: int
    rsi_period: int
    macd: list[int]
    macd_recent_cross_bars: int
    histogram_decline_bars: int
    swing_half_window: int
    divergence_lookback_bars: int
    warning_min_triggered: int


class CrashValidation(CamelModel):
    status: str  # available | unavailable
    reason: str | None = None
    definition: CrashValidationDefinition
    in_sample: CrashEvaluationBlock | None = None
    out_of_sample: CrashEvaluationBlock | None = None
    warnings: list[str] = Field(default_factory=list)


class CrashChartPoint(CamelModel):
    date: str
    close: float | None = None
    sma_short: float | None = None
    sma_long: float | None = None
    rsi: float | None = None
    macd: float | None = None
    signal: float | None = None
    histogram: float | None = None
    triggered_count: int | None = None
    triggered: list[str] = Field(default_factory=list)


class CrashWarningStrategy(CamelModel):
    """EXPERIMENTAL transparent technical-risk monitor (daily). A count of
    explicit indicator conditions - NOT a crash forecast, no probability."""

    timeframe: str
    data_basis: str
    as_of: str
    bars_used: int
    indicators: CrashIndicators
    conditions: list[CrashCondition]
    informational: list[CrashCondition]
    divergence: CrashDivergence | None = None
    triggered_count: int
    scored_condition_count: int
    risk_level: str  # Low | Elevated | High (heuristic cut-offs)
    warning_active: bool
    assessment: str
    validation: CrashValidation
    parameters: CrashParameters
    chart_data: list[CrashChartPoint]
    methodology: list[str] = Field(default_factory=list)
    data_quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str


# ---------------------------------------------------------------------------
# Strategy categories (see section 10/11 of the reorg spec).
#
# These are an ADDITIVE, category-organized view of the exact same data
# already carried by StrategyAnalytics' flat fields below — they do NOT
# replace those flat fields. This keeps any existing consumer that reads
# e.g. `data.emaRsi` / `data.evarRisk` directly working unchanged, while
# giving a new consumer `data.technicalStrategies` / etc. as a
# convenience grouping.
# ---------------------------------------------------------------------------


class TechnicalStrategies(CamelModel):
    ema50_rsi: Ema50RsiStrategy | None = None
    ema821: Ema8Ema21Strategy | None = None
    macd: MacdStrategy | None = None
    bollinger_bands: BollingerBandsStrategy | None = None
    better_breakout: BetterBreakoutStrategy | None = None
    sma_trend: SmaTrendStrategy | None = None
    triple_ma_pullback: TripleMaPullbackStrategy | None = None
    mswing: MswingStrategy | None = None
    td_setup: TdSetupStrategy | None = None
    supertrend_confirmation: SupertrendConfirmationStrategy | None = None
    ufx_vervoort_heiken_ashi: UfxVervoortHeikenAshiStrategy | None = None
    trend_signals_trend_tide: TrendSignalsTrendTideStrategy | None = None
    pivot_levels: PivotLevelsStrategy | None = None
    cpr: CprStrategy | None = None
    ta_candles_trend_health: TaCandlesTrendHealthStrategy | None = None
    ema_bollinger_atr: EmaBollingerAtrStrategy | None = None


class QuantitativeStrategies(CamelModel):
    evar_risk: EvarRiskStrategy | None = None
    momentum_reversal: MomentumReversalStrategy | None = None
    squeeze_momentum: SqueezeMomentumStrategy | None = None
    soc_avalanche: SocAvalancheStrategy | None = None
    wavelet_trend_filter: WaveletTrendStrategy | None = None
    first_passage_time: FirstPassageTimeStrategy | None = None
    graham_valuation: GrahamValuationStrategy | None = None
    greenblatt_magic_formula: GreenblattMagicFormulaStrategy | None = None
    crash_warning: CrashWarningStrategy | None = None


class RegimeStrategies(CamelModel):
    risk_on_risk_off: RiskOnRiskOffStrategy | None = None
    hmm_regime_switching: HmmRegimeSwitchingStrategy | None = None
    factor_momentum: FactorMomentumStrategy | None = None


class CalendarStrategies(CamelModel):
    tlt_monthly_cycle: TltMonthlyCycleStrategy | None = None
    seasonal_heat_map: SeasonalHeatMapStrategy | None = None


class PortfolioStrategies(CamelModel):
    tqqq_tmf_ief_rebalancing: TqqqTmfIefRebalancingStrategy | None = None
    vt_vs_vti_vxus: VtVsVtiVxusStrategy | None = None


class StrategyAnalytics(CamelModel):
    """Top-level payload for GET /etfs/{symbol}/strategies.

    Each field is null when there isn't enough price history yet for a
    stable indicator reading (see services.strategies MIN_ROWS_*). New
    strategy fields are additive only — existing fields/shape are
    unchanged for backward compatibility with the frontend.
    """

    ema50_rsi: Ema50RsiStrategy | None = None
    ema821: Ema8Ema21Strategy | None = None
    macd: MacdStrategy | None = None
    bollinger_bands: BollingerBandsStrategy | None = None
    better_breakout: BetterBreakoutStrategy | None = None
    sma_trend: SmaTrendStrategy | None = None
    # Portfolio-analytics strategy (VT vs VTI+VXUS). Only populated on the
    # VTI symbol's response ("VTI tile" placement) — null for every other
    # symbol. See VtVsVtiVxusStrategy docstring.
    vt_vs_vti_vxus: VtVsVtiVxusStrategy | None = None
    # Strategy 7 (single-ETF, all symbols): EVaR Risk / Position Sizing.
    evar_risk: EvarRiskStrategy | None = None
    # Strategy 8 (cross-ETF, all symbols): Risk-On / Risk-Off Market Regime.
    risk_on_risk_off: RiskOnRiskOffStrategy | None = None
    # Strategy 9 (single-ETF, all symbols): Triple-MA Pullback.
    triple_ma_pullback: TripleMaPullbackStrategy | None = None
    # Strategy 10 (TLT's own data, all symbols — like Risk-On/Risk-Off):
    # TLT Monthly Cycle (calendar-based, not a technical indicator).
    tlt_monthly_cycle: TltMonthlyCycleStrategy | None = None
    # Strategy 11 (TQQQ/TMF/IEF's own data, all symbols — cross-cutting
    # like Risk-On/Risk-Off and TLT Monthly Cycle above): TQQQ / TMF / IEF
    # Rebalancing, a portfolio-allocation strategy, not a technical
    # indicator.
    tqqq_tmf_ief_rebalancing: TqqqTmfIefRebalancingStrategy | None = None
    # Strategy 12 (single-ETF, all symbols): Mswing Momentum.
    mswing: MswingStrategy | None = None
    # Strategy 13 (single-ETF, all symbols): Momentum Reversal.
    momentum_reversal: MomentumReversalStrategy | None = None
    # Strategy 14 (cross-ETF, all symbols — same cross-cutting treatment
    # as Risk-On/Risk-Off): One-Month Factor Momentum.
    factor_momentum: FactorMomentumStrategy | None = None
    # Strategy 15 (single-ETF, all symbols): HMM Regime-Switching.
    hmm_regime_switching: HmmRegimeSwitchingStrategy | None = None
    # Strategy 16 (single-ETF, all symbols): Squeeze Momentum Indicator.
    squeeze_momentum: SqueezeMomentumStrategy | None = None
    # Strategy 17 (single-ETF, all symbols): Self-Organized Criticality —
    # Avalanche Distribution.
    soc_avalanche: SocAvalancheStrategy | None = None
    # Strategy 18 (single-ETF, all symbols): Adaptive Causal Wavelet
    # Trend Filter.
    wavelet_trend_filter: WaveletTrendStrategy | None = None
    # Strategy 19 (single-ETF, all symbols): First Passage Time
    # Distribution Analysis.
    first_passage_time: FirstPassageTimeStrategy | None = None
    # Strategies 20-26 (single-ETF, all symbols): additional Technical
    # Strategies - TD Setup / TD 9, Supertrend Confirmation, UFX + Vervoort
    # Heiken Ashi, Trend Signals + Trend Tide, Pivot Levels, CPR, and
    # TA Candles / Trend Health.
    td_setup: TdSetupStrategy | None = None
    supertrend_confirmation: SupertrendConfirmationStrategy | None = None
    ufx_vervoort_heiken_ashi: UfxVervoortHeikenAshiStrategy | None = None
    trend_signals_trend_tide: TrendSignalsTrendTideStrategy | None = None
    pivot_levels: PivotLevelsStrategy | None = None
    cpr: CprStrategy | None = None
    ta_candles_trend_health: TaCandlesTrendHealthStrategy | None = None
    # Strategies 27-28 (Quantitative, fundamental valuation): Graham
    # growth-valuation (per-stock) and Greenblatt Magic Formula (stock-
    # universe ranking). Always an object with a `status`, never silently null.
    graham_valuation: GrahamValuationStrategy | None = None
    greenblatt_magic_formula: GreenblattMagicFormulaStrategy | None = None
    # Strategies 29, 31, 32 (single-ETF, all symbols): Seasonal Performance
    # Heat Map (calendar), EMA + Bollinger + ATR backtest (technical) and the
    # experimental Technical Crash-Warning indicator (quantitative / risk).
    # Strategy 30 (Diversification Heat Map) is multi-asset and is served by
    # POST /api/portfolio/diversification_heat_map instead.
    seasonal_heat_map: SeasonalHeatMapStrategy | None = None
    ema_bollinger_atr: EmaBollingerAtrStrategy | None = None
    crash_warning: CrashWarningStrategy | None = None

    # Category-organized view of the SAME strategies above (additive —
    # see the "Strategy categories" note preceding TechnicalStrategies).
    technical_strategies: TechnicalStrategies | None = None
    quantitative_strategies: QuantitativeStrategies | None = None
    regime_strategies: RegimeStrategies | None = None
    calendar_strategies: CalendarStrategies | None = None
    portfolio_strategies: PortfolioStrategies | None = None
