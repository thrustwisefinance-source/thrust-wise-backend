"""
Shared scaffolding for the additive Set 3 strategies.

``StrategyDescriptor`` carries the static identity every strategy result
exposes (id, name, category, description, calculation version).
``run_isolated`` runs one strategy so that an unexpected exception becomes an
``error`` result for THAT strategy only - sibling strategies are unaffected.

This module is independent of the registry primitives in ``common.py`` (which
belong to the pre-existing Set 3 registry); it only borrows nothing from it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

R = TypeVar("R")

CATEGORY_QUANTITATIVE = "quantitative"


@dataclass(frozen=True)
class StrategyDescriptor:
    strategy_id: str
    name: str
    category: str
    subcategory: str
    description: str
    calculation_version: str


async def run_isolated(
    descriptor: StrategyDescriptor,
    run: Callable[[], Awaitable[R]],
    on_error: Callable[[str], R],
) -> R:
    """Await ``run()``; on ANY exception log it and return ``on_error(msg)``.

    The message handed to ``on_error`` is deliberately generic (no internals
    leak to API clients); the full traceback goes to the server log.
    """
    try:
        return await run()
    except Exception:  # noqa: BLE001 - isolation boundary by design
        logger.exception("Set 3 strategy %s failed", descriptor.strategy_id)
        return on_error("Calculation failed unexpectedly; please try again later.")
