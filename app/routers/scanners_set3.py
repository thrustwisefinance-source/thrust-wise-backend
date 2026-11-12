"""
Set 3 stock scanner endpoint (separate from the ETF strategies and from the
CANSLIM scanner router, neither of which is modified).

    POST /api/scanners/quantitative_set3

Technical-Fundamental Composite Scoring: each stock gets a 0-4 score from four
independent conditions plus a separate "RSI below 30" flag. A failure for one
symbol never prevents the others from being processed.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.scanners import universe as scanner_universe
from app.scanners.set_3_scanners.quantitative import technical_fundamental_composite as tfcs
from app.scanners.set_3_scanners.schemas.quantitative import MAX_SYMBOLS
from app.schemas.common import ApiResponse
from app.schemas.scanners_set3 import (
    QuantitativeScannerSet3Request,
    QuantitativeScannerSet3Response,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scanners", tags=["scanners-set3"])

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


@router.post(
    "/quantitative_set3",
    response_model=ApiResponse[QuantitativeScannerSet3Response],
)
async def run_quantitative_set3(
    body: QuantitativeScannerSet3Request,
    db: AsyncSession = Depends(get_db),
):
    """Score one or more stocks (Technical-Fundamental Composite Scoring)."""
    symbols = list(body.symbols)
    universe_key: str | None = None
    notes: list[str] = []

    if body.universe:
        source = scanner_universe.get_universe_source(body.universe)
        if source is None:
            raise HTTPException(status_code=404, detail=f"Unknown universe: {body.universe}")
        universe_key = source.key
        members = await scanner_universe.get_stock_universe(db, source.key)
        if not members:
            if not symbols:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"The '{source.key}' universe has not been ingested yet; supply "
                        "'symbols' explicitly or run the scanner universe ingestion first."
                    ),
                )
            notes.append(f"Universe '{source.key}' is not ingested; only the supplied symbols were scanned.")
        for m in members[: body.universe_limit]:
            s = str(m.symbol).upper()
            if _SYMBOL_RE.match(s) and s not in symbols:
                symbols.append(s)
        if members and len(members) > body.universe_limit:
            notes.append(
                f"Universe '{source.key}' has {len(members)} members; the first "
                f"{body.universe_limit} (alphabetical) were scanned (universeLimit)."
            )

    if not symbols:
        raise HTTPException(status_code=422, detail="No valid symbols to scan.")
    if len(symbols) > MAX_SYMBOLS:
        raise HTTPException(
            status_code=422,
            detail=f"At most {MAX_SYMBOLS} symbols can be scanned per request (got {len(symbols)}).",
        )

    data = await tfcs.scan_symbols(symbols, body, universe=universe_key)
    data.notes = notes + data.notes
    failed = data.summary.error + data.summary.unavailable
    message = (
        f"{failed} of {data.summary.requested} symbol(s) could not be scored; see each result's error / dataAvailability."
        if failed
        else None
    )
    return ApiResponse(data=data, message=message)
