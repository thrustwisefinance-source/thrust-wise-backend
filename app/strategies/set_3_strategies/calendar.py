"""
Set 3 - CALENDAR strategies.

Seasonality, monthly cycles, weekday effects and other calendar-based methods.

Note: this module is named `calendar` but is always imported by its full
package path (app.strategies.set_3_strategies.calendar), so it does not shadow
the standard-library `calendar` module. Never add this folder itself to
sys.path.

How to add a strategy
---------------------
1. Write a pure function `compute_<name>(ctx: Set3Context) -> dict`
   (raise Set3StrategyUnavailable when there is not enough data).
2. Register it in CALENDAR_STRATEGIES below.
3. Add the response model + one field to CalendarSet3Strategies in
   app/schemas/strategy_set3.py (field name == registry key).
No router change is needed.
"""

from __future__ import annotations

from app.strategies.set_3_strategies.common import Set3Strategy

CALENDAR_STRATEGIES: dict[str, Set3Strategy] = {}
