"""
Pydantic models for the two additive Set 3 QUANTITATIVE strategies:

* ``asymmetric_volatility_persistence``  (Quantitative -> Risk Management)
* ``fed_model_relative_valuation``       (Quantitative -> Asset Allocation /
                                          Relative Valuation)

Conventions follow the rest of the project (``CamelModel``: snake_case in
Python, camelCase JSON). Dates are ISO ``YYYY-MM-DD`` exchange dates (no time
zone); every ``*_pct`` value is in PERCENT units (5.0 means 5 %).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import CamelModel

CalculationStatus = Literal["ok", "partial", "unavailable", "not_applicable", "error"]
AvailabilityStatus = Literal["available", "partial", "unavailable", "not_applicable"]

# ---------------------------------------------------------------------------
# Shared envelope
# ---------------------------------------------------------------------------


class DataAvailability(CamelModel):
    status: AvailabilityStatus
    message: str | None = None
    observations: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    # component name -> "available" | "unavailable: <reason>" | ...
    components: dict[str, str] = {}


class Interpretation(CamelModel):
    summary: str
    points: list[str] = []
    classification: str | None = None
    disclaimer: str = (
        "Analytical description of historical data only - not a trading signal "
        "or investment recommendation."
    )


class DatePoint(CamelModel):
    date: str
    value: float


class LabeledValue(CamelModel):
    label: str
    value: float


class StrategyEnvelope(CamelModel):
    """Fields every Set 3 strategy result carries."""

    strategy_id: str
    strategy_name: str
    category: Literal["quantitative"] = "quantitative"
    subcategory: str
    symbol: str
    description: str
    calculation_version: str
    interpretation: Interpretation | None = None
    data_availability: DataAvailability
    calculation_status: CalculationStatus
    error: str | None = None
    warnings: list[str] = []
    notes: list[str] = []


# ===========================================================================
# Strategy 1 - Asymmetric Volatility Persistence
# ===========================================================================

AVP_STRATEGY_ID = "asymmetric_volatility_persistence"


class AsymmetricVolatilityParams(CamelModel):
    """Validated request parameters (query-string driven)."""

    return_type: Literal["log", "simple"] = "log"
    # Calendar-day window ending at the newest price; None = all stored history.
    lookback_days: int | None = Field(None, ge=90, le=36_500)
    vol_window: int = Field(21, ge=2, le=252)
    acf_max_lag: int = Field(20, ge=1, le=100)
    shock_horizon: int = Field(5, ge=1, le=60)
    shock_threshold_sigma: float = Field(0.0, ge=0.0, le=3.0)
    annualization_factor: int = Field(252, ge=1, le=366)
    garch_enabled: bool = True
    garch_model: Literal["garch", "gjr"] = "gjr"
    garch_max_observations: int = Field(2500, ge=250, le=6000)
    series_points: int = Field(500, ge=10, le=5000)
    chart_max_points: int = Field(400, ge=10, le=2000)

    @model_validator(mode="after")
    def _cross_checks(self) -> "AsymmetricVolatilityParams":
        if self.vol_window > 1 and self.acf_max_lag >= 10 * self.vol_window:
            raise ValueError("acf_max_lag is unreasonably large relative to vol_window")
        return self


class ReturnSummary(CamelModel):
    return_type: str
    observations: int
    start_date: str
    end_date: str
    mean_daily_return_pct: float
    daily_volatility_pct: float
    historical_volatility_annualized_pct: float
    annualization_factor: int
    mean_squared_return_pct2: float
    max_squared_return_pct2: float


class RollingVolatilitySummary(CamelModel):
    window: int
    current_pct: float | None = None
    mean_pct: float | None = None
    min_pct: float | None = None
    max_pct: float | None = None
    current_percentile: float | None = None


class AcfPoint(CamelModel):
    lag: int
    acf: float
    significant: bool


class LjungBoxResult(CamelModel):
    lags: int
    statistic: float
    p_value: float
    reject_no_autocorrelation_5pct: bool


class AutocorrelationBlock(CamelModel):
    series: str = "squared_returns"
    max_lag: int
    confidence_band: float
    values: list[AcfPoint]


class PersistenceBlock(CamelModel):
    lag1_acf: float
    mean_acf: float
    significant_lags: int
    ljung_box: LjungBoxResult
    ar1_coefficient: float | None = None
    ar1_half_life_days: float | None = None
    clustering: Literal["significant_clustering", "no_significant_clustering"]


class ShockClassStatsModel(CamelModel):
    count: int
    mean_return_pct: float | None = None
    next_day_volatility_annualized_pct: float | None = None
    forward_volatility_annualized_pct: float | None = None
    sufficient_observations: bool


class PersistenceRegression(CamelModel):
    persistence_after_non_negative_day: float | None = None
    persistence_after_negative_shock: float | None = None
    difference: float | None = None
    difference_t_statistic: float | None = None
    significant_at_5pct: bool | None = None
    observations: int = 0


class LeveragePoint(CamelModel):
    lag: int
    correlation: float | None = None


class AsymmetryBlock(CamelModel):
    shock_threshold_sigma: float
    horizon_days: int
    negative_shocks: ShockClassStatsModel
    positive_shocks: ShockClassStatsModel
    forward_volatility_ratio_negative_over_positive: float | None = None
    next_day_volatility_ratio_negative_over_positive: float | None = None
    persistence_regression: PersistenceRegression
    leverage_correlation: list[LeveragePoint]
    conclusion: Literal[
        "negative_shocks_followed_by_higher_volatility",
        "positive_shocks_followed_by_higher_volatility",
        "no_clear_asymmetry",
        "insufficient_data",
    ]
    conclusion_method: Literal["garch_lr_test", "persistence_regression"]
    conclusion_basis: str


class GarchParameters(CamelModel):
    omega: float
    alpha: float
    beta: float
    gamma: float | None = None


class LeverageLikelihoodRatio(CamelModel):
    statistic: float
    p_value: float
    significant_at_5pct: bool
    comparison_model: str = "garch"


class GarchBlock(CamelModel):
    status: Literal["available", "unavailable", "failed", "disabled"]
    message: str | None = None
    model: str | None = None
    estimation_method: str | None = None
    observations_used: int | None = None
    parameters: GarchParameters | None = None
    persistence: float | None = None
    half_life_days: float | None = None
    unconditional_volatility_annualized_pct: float | None = None
    current_conditional_volatility_annualized_pct: float | None = None
    log_likelihood: float | None = None
    aic: float | None = None
    bic: float | None = None
    converged: bool | None = None
    leverage_test: LeverageLikelihoodRatio | None = None


class AsymmetricVolatilityCalculatedValues(CamelModel):
    returns: ReturnSummary
    rolling_volatility: RollingVolatilitySummary
    autocorrelation: AutocorrelationBlock
    persistence: PersistenceBlock
    asymmetry: AsymmetryBlock
    garch: GarchBlock


class AsymmetricVolatilityHistoricalSeries(CamelModel):
    """Most recent ``series_points`` observations, one aligned array per field."""

    dates: list[str]
    returns_pct: list[float | None]
    squared_returns_pct2: list[float | None]
    rolling_volatility_pct: list[float | None]
    garch_conditional_volatility_pct: list[float | None]


class VolatilityChartPoint(CamelModel):
    date: str
    rolling_volatility_pct: float | None = None
    garch_volatility_pct: float | None = None


class AcfChartPoint(CamelModel):
    lag: int
    acf: float
    upper_band: float
    lower_band: float


class AsymmetryChartBar(CamelModel):
    label: str
    count: int
    forward_volatility_annualized_pct: float | None = None


class AsymmetricVolatilityChartData(CamelModel):
    """Full-history, down-sampled series for charts."""

    volatility: list[VolatilityChartPoint]
    autocorrelation: list[AcfChartPoint]
    shock_asymmetry: list[AsymmetryChartBar]
    leverage_correlation: list[LeveragePoint]


class AsymmetricVolatilityResult(StrategyEnvelope):
    parameters: AsymmetricVolatilityParams
    calculated_values: AsymmetricVolatilityCalculatedValues | None = None
    historical_series: AsymmetricVolatilityHistoricalSeries | None = None
    chart_data: AsymmetricVolatilityChartData | None = None


# ===========================================================================
# Strategy 2 - Fed Model Relative Valuation
# ===========================================================================

FED_STRATEGY_ID = "fed_model_relative_valuation"


class FedModelParams(CamelModel):
    lookback_years: int = Field(5, ge=1, le=40)
    pe_basis: Literal["trailing", "forward"] = "trailing"
    # |spread| below this many percentage points is reported as "approximately equal"
    neutral_band_pct: float = Field(0.25, ge=0.0, le=5.0)
    chart_max_points: int = Field(400, ge=10, le=2000)
    include_granger: bool = True


class TreasuryStats(CamelModel):
    lookback_years: int
    observations: int
    start_date: str
    end_date: str
    min_pct: float
    max_pct: float
    mean_pct: float
    current_percentile: float


class GrangerBlock(CamelModel):
    status: Literal["available", "unavailable"]
    message: str


class FedModelCalculatedValues(CamelModel):
    pe_ratio: float
    pe_basis: str
    earnings_yield_pct: float
    earnings_yield_source: str
    earnings_yield_as_of: str
    earnings_yield_is_proxy: bool
    treasury_yield_pct: float
    treasury_yield_as_of: str
    treasury_yield_source: str
    fed_model_spread_pct: float
    date_alignment_days: int
    neutral_band_pct: float
    relative_valuation: Literal[
        "equities_relatively_more_attractive",
        "bonds_relatively_more_attractive",
        "approximately_equal",
    ]
    treasury_stats: TreasuryStats | None = None
    granger_causality: GrangerBlock | None = None


class FedModelHistoricalSeries(CamelModel):
    """Only REAL observations are listed; nothing is back-filled or estimated."""

    earnings_yield: list[DatePoint] = []
    treasury_yield: list[DatePoint] = []
    fed_model_spread: list[DatePoint] = []
    earnings_yield_history_available: bool = False
    earnings_yield_history_note: str | None = None


class FedModelChartPoint(CamelModel):
    date: str
    earnings_yield_pct: float | None = None
    treasury_yield_pct: float | None = None
    fed_model_spread_pct: float | None = None


class FedModelChartData(CamelModel):
    series: list[FedModelChartPoint]
    latest_comparison: list[LabeledValue]  # earnings yield / treasury yield / spread


class FedModelResult(StrategyEnvelope):
    parameters: FedModelParams
    calculated_values: FedModelCalculatedValues | None = None
    historical_series: FedModelHistoricalSeries | None = None
    chart_data: FedModelChartData | None = None
