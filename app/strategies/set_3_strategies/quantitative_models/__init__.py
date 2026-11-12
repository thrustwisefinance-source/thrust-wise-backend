"""
Implementations of the additive Set 3 quantitative strategies.

NOTE ON NAMING: the task layout asks for ``set_3_strategies/quantitative/``,
but the repository already has ``set_3_strategies/quantitative.py`` (the live
registry module). A package named ``quantitative`` would shadow that module
and break the existing endpoint, so the new strategy implementations live in
this sibling package instead. Nothing existing was moved or renamed.
"""

NEW_QUANTITATIVE_STRATEGY_IDS = (
    "asymmetric_volatility_persistence",
    "fed_model_relative_valuation",
)
