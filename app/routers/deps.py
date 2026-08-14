"""
Shared router helpers: symbol resolution and price loading.
"""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import ETF_REGISTRY, EtfStaticMeta
from app.models import DailyPrice


def resolve_meta(symbol: str) -> EtfStaticMeta:
    meta = ETF_REGISTRY.get(symbol.upper())
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Unknown ETF symbol: {symbol}")
    return meta


async def load_prices(
    db: AsyncSession, symbol: str, limit: int | None = None
) -> list[DailyPrice]:
    """Daily bars for a symbol, ascending by date. `limit` takes the most recent N."""
    stmt = (
        select(DailyPrice)
        .where(DailyPrice.symbol == symbol)
        .order_by(DailyPrice.date.desc())
    )
    if limit:
        stmt = stmt.limit(limit)
    rows = (await db.scalars(stmt)).all()
    return list(reversed(rows))


def require_prices(prices: list[DailyPrice], symbol: str) -> None:
    if not prices:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Price data for {symbol} has not been ingested yet. Try again shortly."
            ),
        )
