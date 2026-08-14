"""
Aggregate summary for the dashboard endpoint.
"""

from app.schemas.common import CamelModel
from app.schemas.etf_snapshot import EtfSnapshot


class DashboardMover(CamelModel):
    symbol: str
    name: str
    change_percent: float


class MarketIndex(CamelModel):
    """Index-level widget: S&P 500 / NASDAQ Composite / Dow Jones."""

    symbol: str  # e.g. "GSPC"
    name: str
    level: float  # index points, not a dollar price
    change_amount: float
    change_percent: float
    sparkline: list[float]


class DashboardSummary(CamelModel):
    as_of: str  # ISO date of the latest trading day
    total_etfs: int
    top_performer: DashboardMover | None
    worst_performer: DashboardMover | None
    average_expense_ratio: float
    indices: list[MarketIndex] = []
    snapshots: list[EtfSnapshot]
