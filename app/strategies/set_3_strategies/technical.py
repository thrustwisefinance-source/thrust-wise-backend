"""
Set 3 - TECHNICAL strategies.

Technical indicators, price action, moving averages, trend analysis, etc.

How to add a strategy
---------------------
1. Write a pure function `compute_<name>(ctx: Set3Context) -> dict`
   (raise Set3StrategyUnavailable when there is not enough data).
2. Register it in TECHNICAL_STRATEGIES below.
3. Add the response model + one field to TechnicalSet3Strategies in
   app/schemas/strategy_set3.py (field name == registry key).
No router change is needed.
"""

from __future__ import annotations

from app.strategies.set_3_strategies.common import Set3Strategy

TECHNICAL_STRATEGIES: dict[str, Set3Strategy] = {}
