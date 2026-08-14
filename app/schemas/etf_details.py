"""
Matches the frontend EtfDetails shape (constants/etf-details.ts) and the
Recharts PerformancePoint shape (constants/etf-performance-series.ts).
"""

from app.schemas.common import CamelModel


class Holding(CamelModel):
    name: str
    weight: float  # percent of fund, e.g. 6.9


class PerformanceStats(CamelModel):
    ytd: float
    one_year: float
    three_year: float
    five_year: float
    annualized: float


class RiskMetrics(CamelModel):
    risk_score: float  # rule-based 1-10
    volatility: float  # annualized, percent
    beta: float  # vs SPY
    sharpe_ratio: float
    max_drawdown: float = 0.0  # peak-to-trough decline, percent


class PerformancePoint(CamelModel):
    label: str
    value: float


# Chart series grouped by range, e.g. {"1M": [...], "3M": [...], ...}
PerformanceSeries = dict[str, list[PerformancePoint]]


class EtfDetails(CamelModel):
    # Snapshot-level fields
    symbol: str
    name: str
    category: str
    issuer: str
    price: float
    change_amount: float
    change_percent: float
    expense_ratio: float
    risk_level: str

    # Detail fields
    fund_objective: str
    fund_type: str
    asset_class: str
    inception_date: str
    aum: float
    dividend_yield: float
    holdings: list[Holding]
    performance_stats: PerformanceStats
    risk_metrics: RiskMetrics
    education: str
    related_symbols: list[str]
