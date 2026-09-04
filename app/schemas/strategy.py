"""
Strategy Analytics response shapes.

Analytical / visualization data only — no signal, action, or
recommendation fields anywhere in this module.
"""

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