"""
Side-by-side comparison rows for /etfs/compare?symbols=VOO,SPY.
"""

from app.schemas.common import CamelModel
from app.schemas.etf_details import PerformanceStats, RiskMetrics


class EtfComparisonEntry(CamelModel):
    symbol: str
    name: str
    issuer: str
    category: str
    price: float
    change_percent: float
    expense_ratio: float
    risk_level: str
    dividend_yield: float
    aum: float
    performance_stats: PerformanceStats
    risk_metrics: RiskMetrics


class CompareResult(CamelModel):
    entries: list[EtfComparisonEntry]
    # Pairwise correlation of daily returns, aligned to the entries order
    correlation_matrix: list[list[float]]


class AllocationItem(CamelModel):
    symbol: str
    weight: float  # percent of portfolio, normalized to sum to 100


class AllocationMetrics(CamelModel):
    annualized_return: float  # CAGR of the blended portfolio, percent
    volatility: float  # annualized, percent
    sharpe_ratio: float
    max_drawdown: float  # percent (negative)
    risk_score: float  # rule-based 1-10


class AllocationResult(CamelModel):
    allocations: list[AllocationItem]
    metrics: AllocationMetrics
    # Constituent correlations, aligned to the allocations order
    correlation_matrix: list[list[float]]
