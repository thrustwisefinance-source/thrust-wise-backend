"""
Set 3 request / response models.

Follows the existing conventions (app.schemas.common.CamelModel -> camelCase
JSON, wrapped by ApiResponse[T] in the router). Nothing in the existing
schema modules is modified.

Adding a strategy
-----------------
1. Add its response model here.
2. Add ONE field to the matching ``<Category>Set3Strategies`` model below
   (field name == the strategy's registry key).
"""

from __future__ import annotations

import datetime
import re
from typing import Literal

from pydantic import Field, model_validator

from app.schemas.common import CamelModel

# ---------------------------------------------------------------------------
# Request: Randomized Investment Simulation parameters
# ---------------------------------------------------------------------------

_SYMBOL_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")
PROBABILITY_TOLERANCE = 1e-9


class RandomizedSimulationParams(CamelModel):
    """Validated parameters (built from query params by the router)."""

    initial_capital: float = Field(10_000.0, gt=0, le=1e12)
    start_date: datetime.date | None = None
    end_date: datetime.date | None = None
    # Daily action: BUY with p=buy_probability, SELL with p=sell_probability,
    # otherwise HOLD (p = 1 - buy - sell). Not silently renormalised: the sum
    # must be <= 1 so the stated probabilities are the ones actually used.
    buy_probability: float = Field(0.5, ge=0, le=1)
    sell_probability: float = Field(0.5, ge=0, le=1)
    # Upper bound on the share of CASH spent in one buy (the actual share is
    # drawn at random up to this bound).
    max_purchase_allocation_percent: float = Field(50.0, gt=0, le=100)
    # Upper bound on the share of HELD SHARES sold in one sell.
    max_sell_percent: float = Field(50.0, gt=0, le=100)
    iterations: int = Field(100, ge=1, le=1000)
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    benchmark_symbol: str = "SPY"
    commission_percent: float = Field(0.05, ge=0, le=5)  # % of trade value
    commission_per_trade: float = Field(0.0, ge=0, le=1000)  # flat, per trade
    chart_max_points: int = Field(400, ge=10, le=2000)

    @model_validator(mode="after")
    def _validate(self) -> "RandomizedSimulationParams":
        total = self.buy_probability + self.sell_probability
        if total <= 0:
            raise ValueError(
                "buy_probability + sell_probability must be greater than 0"
            )
        if total > 1 + PROBABILITY_TOLERANCE:
            raise ValueError(
                "buy_probability + sell_probability must not exceed 1 "
                f"(got {total:.4f}); the remainder is the hold probability"
            )
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError("start_date must not be after end_date")
        symbol = self.benchmark_symbol.strip().upper()
        if not _SYMBOL_RE.match(symbol):
            raise ValueError("benchmark_symbol must be 1-10 chars of A-Z, 0-9, '.', '-'")
        self.benchmark_symbol = symbol
        return self


# ---------------------------------------------------------------------------
# Response: Randomized Investment Simulation
# ---------------------------------------------------------------------------


class RandomizedSimulationEffectiveParams(CamelModel):
    """Parameters actually used (defaults / seed resolved)."""

    initial_capital: float
    buy_probability: float
    sell_probability: float
    hold_probability: float
    max_purchase_allocation_percent: float
    max_sell_percent: float
    iterations: int
    seed_used: int
    seed_supplied: bool
    reproducible: bool
    benchmark_symbol: str
    commission_percent: float
    commission_per_trade: float
    price_basis: str
    execution_assumption: str


class RandomizedSimulationPeriod(CamelModel):
    start_date: str
    end_date: str
    trading_days: int
    years: float


class BuyAndHoldResult(CamelModel):
    """Buy & Hold over the SAME period, capital and cost model."""

    symbol: str
    initial_capital: float
    shares: int
    final_value: float
    total_return_percent: float
    annualized_return_percent: float | None = None
    max_drawdown_percent: float
    total_commission: float


class RandomizedSimulationAggregate(CamelModel):
    """Statistics ACROSS all simulation iterations."""

    iterations: int
    initial_capital: float
    average_final_value: float
    median_final_value: float
    average_return_percent: float
    median_return_percent: float
    best_return_percent: float
    worst_return_percent: float
    std_dev_return_percent: float
    average_annualized_return_percent: float | None = None
    median_annualized_return_percent: float | None = None
    average_max_drawdown_percent: float
    median_max_drawdown_percent: float
    worst_max_drawdown_percent: float
    average_trades: float
    percent_outperforming_buy_and_hold: float
    percent_outperforming_benchmark: float | None = None
    average_difference_vs_buy_and_hold_points: float
    median_difference_vs_buy_and_hold_points: float
    average_difference_vs_benchmark_points: float | None = None


class RandomizedSimulationIteration(CamelModel):
    """Result of ONE simulation run (own cash / share balance)."""

    iteration: int
    final_value: float
    final_cash: float
    final_shares: int
    total_return_percent: float
    annualized_return_percent: float | None = None
    max_drawdown_percent: float
    buy_trades: int
    sell_trades: int
    total_commission: float
    difference_vs_buy_and_hold_points: float
    beat_buy_and_hold: bool
    beat_benchmark: bool | None = None


class RandomizedSimulationChartPoint(CamelModel):
    """Portfolio value (in currency) at one date, for the frontend chart."""

    date: str
    buy_and_hold: float
    benchmark: float | None = None
    mean: float
    median: float
    p5: float
    p95: float
    best_iteration: float
    worst_iteration: float
    median_iteration: float


class RandomizedSimulationSamplePaths(CamelModel):
    """Which iteration the best / worst / median chart columns come from."""

    best_iteration: int
    worst_iteration: int
    median_iteration: int


class RandomizedInvestmentSimulation(CamelModel):
    name: str
    category: Literal["quantitative"] = "quantitative"
    symbol: str
    parameters: RandomizedSimulationEffectiveParams
    period: RandomizedSimulationPeriod
    buy_and_hold: BuyAndHoldResult
    benchmark: BuyAndHoldResult | None = None
    aggregate: RandomizedSimulationAggregate
    iteration_results: list[RandomizedSimulationIteration]
    sample_paths: RandomizedSimulationSamplePaths
    chart_data: list[RandomizedSimulationChartPoint]
    warnings: list[str] = []
    notes: list[str] = []


# ---------------------------------------------------------------------------
# Per-category ``strategies`` containers
#   one optional field per registered strategy; field name == registry key
# ---------------------------------------------------------------------------


class TechnicalSet3Strategies(CamelModel):
    pass


class QuantitativeSet3Strategies(CamelModel):
    randomized_investment_simulation: RandomizedInvestmentSimulation | None = None


class RegimeSet3Strategies(CamelModel):
    pass


class CalendarSet3Strategies(CamelModel):
    pass


class PortfolioSet3Strategies(CamelModel):
    pass


# ---------------------------------------------------------------------------
# Per-category response bodies (wrapped in ApiResponse[...] by the router)
# ---------------------------------------------------------------------------


class Set3StrategyInfo(CamelModel):
    key: str
    title: str


class Set3CategoryBase(CamelModel):
    symbol: str
    set_name: str = "set3"
    category: str
    registered_strategies: list[Set3StrategyInfo] = []
    # strategy key -> why it is null (insufficient data, failure, ...)
    unavailable: dict[str, str] = {}


class TechnicalSet3Response(Set3CategoryBase):
    category: str = "technical"
    strategies: TechnicalSet3Strategies = TechnicalSet3Strategies()


class QuantitativeSet3Response(Set3CategoryBase):
    category: str = "quantitative"
    strategies: QuantitativeSet3Strategies = QuantitativeSet3Strategies()


class RegimeSet3Response(Set3CategoryBase):
    category: str = "regime"
    strategies: RegimeSet3Strategies = RegimeSet3Strategies()


class CalendarSet3Response(Set3CategoryBase):
    category: str = "calendar"
    strategies: CalendarSet3Strategies = CalendarSet3Strategies()


class PortfolioSet3Response(Set3CategoryBase):
    category: str = "portfolio"
    strategies: PortfolioSet3Strategies = PortfolioSet3Strategies()
