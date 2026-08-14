"""
Matches the frontend EtfSnapshot shape used by the Explorer cards
(constants/etf-snapshots.ts).
"""

from app.schemas.common import CamelModel


class EtfSnapshot(CamelModel):
    symbol: str
    name: str
    category: str
    issuer: str
    price: float
    change_amount: float
    change_percent: float
    expense_ratio: float
    risk_level: str
    sparkline: list[float]


class EtfQuote(CamelModel):
    """Latest end-of-day quote for /etfs/{symbol}/quote."""

    symbol: str
    price: float
    change_amount: float
    change_percent: float
    open: float
    high: float
    low: float
    previous_close: float
    volume: int
    as_of: str  # ISO date of the latest trading day
