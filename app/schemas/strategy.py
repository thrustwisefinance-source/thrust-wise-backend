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