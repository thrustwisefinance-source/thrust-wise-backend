"""
Set 3 strategy architecture - completely separate from app.services.strategies.

Layout (one module per category; each exposes a ``<CATEGORY>_STRATEGIES``
registry that the Set 3 router iterates):

    technical.py      technical indicators / price action / trend
    quantitative.py   statistical & stochastic models
    regime.py         market regimes / volatility / risk-on-off
    calendar.py       seasonality & calendar effects
    portfolio.py      allocation / optimisation / rebalancing
    common.py         shared Set 3 helpers (cache keys, registry primitives)
"""

from app.strategies.set_3_strategies.calendar import CALENDAR_STRATEGIES
from app.strategies.set_3_strategies.common import Set3Strategy
from app.strategies.set_3_strategies.portfolio import PORTFOLIO_STRATEGIES
from app.strategies.set_3_strategies.quantitative import QUANTITATIVE_STRATEGIES
from app.strategies.set_3_strategies.regime import REGIME_STRATEGIES
from app.strategies.set_3_strategies.technical import TECHNICAL_STRATEGIES

CATEGORY_REGISTRIES: dict[str, dict[str, Set3Strategy]] = {
    "technical": TECHNICAL_STRATEGIES,
    "quantitative": QUANTITATIVE_STRATEGIES,
    "regime": REGIME_STRATEGIES,
    "calendar": CALENDAR_STRATEGIES,
    "portfolio": PORTFOLIO_STRATEGIES,
}

__all__ = ["CATEGORY_REGISTRIES"]
