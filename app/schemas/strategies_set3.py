"""
Router-level schemas for the additive Set 3 quantitative endpoint
(GET /api/etfs/{symbol}/strategies/quantitative_set3, app/routers/strategies_set3.py).

The strategy result models live next to their implementations
(app/strategies/set_3_strategies/schemas/quantitative.py); this module only
composes them into the category response. The existing
app/schemas/strategy_set3.py is not modified: the new response is a SUPERSET of
the existing QuantitativeSet3Response (same top-level fields, same
``randomizedInvestmentSimulation`` entry, plus the two new strategies).
"""

from __future__ import annotations

from app.schemas.common import CamelModel
from app.schemas.strategy_set3 import RandomizedInvestmentSimulation, Set3CategoryBase
from app.strategies.set_3_strategies.schemas.quantitative import (
    AVP_STRATEGY_ID,
    FED_STRATEGY_ID,
    AsymmetricVolatilityParams,
    AsymmetricVolatilityResult,
    FedModelParams,
    FedModelResult,
)

RANDOMIZED_STRATEGY_ID = "randomized_investment_simulation"
QUANTITATIVE_SET3_STRATEGY_KEYS: tuple[str, ...] = (
    RANDOMIZED_STRATEGY_ID,
    AVP_STRATEGY_ID,
    FED_STRATEGY_ID,
)

__all__ = [
    "AsymmetricVolatilityParams",
    "FedModelParams",
    "QuantitativeSet3ExtendedResponse",
    "QuantitativeSet3ExtendedStrategies",
    "QUANTITATIVE_SET3_STRATEGY_KEYS",
]


class QuantitativeSet3ExtendedStrategies(CamelModel):
    randomized_investment_simulation: RandomizedInvestmentSimulation | None = None
    asymmetric_volatility_persistence: AsymmetricVolatilityResult | None = None
    fed_model_relative_valuation: FedModelResult | None = None


class QuantitativeSet3ExtendedResponse(Set3CategoryBase):
    category: str = "quantitative"
    strategies: QuantitativeSet3ExtendedStrategies = QuantitativeSet3ExtendedStrategies()
