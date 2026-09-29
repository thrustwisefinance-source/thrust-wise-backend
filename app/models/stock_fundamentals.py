"""
SQLAlchemy model caching the specific EODHD Fundamentals API fields the
CANSLIM stock scanner needs (see app/scanners/data.py for extraction and
app/scanners/canslim.py for how each field maps to a CANSLIM criterion).

This is deliberately a narrow, typed cache of *already-extracted* fields
(not a raw JSON blob of the whole EODHD fundamentals payload) so that
"what data was this criterion computed from" stays explicit and
queryable, and so the scanner never has to re-parse EODHD's nested
payload shape on every request.

`fundamentals_as_of` is the period the underlying financial data covers
(e.g. the most recent fiscal quarter end date reported by EODHD) —
distinct from `fetched_at`, which is when ThrustWise pulled it. See the
"DATA FRESHNESS" requirement in the CANSLIM scanner docs: quarterly
fundamentals must never be presented as if they were as fresh as the
daily price feed.
"""

import datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class StockFundamentalsSnapshot(Base):
    __tablename__ = "stock_fundamentals_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "symbol", name="uq_stock_fundamentals_snapshots_symbol"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)

    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(150), nullable=True)

    # --- C: Current Quarterly Earnings (EODHD Earnings.History) ---
    # Most recent reported quarter and the same quarter one year prior,
    # stored explicitly (not just the computed growth rate) so the
    # scanner result stays explainable per the "no black-box score" spec.
    quarterly_eps_current: Mapped[float | None] = mapped_column(Float, nullable=True)
    quarterly_eps_current_period: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )  # e.g. "2026-06-30"
    quarterly_eps_year_ago: Mapped[float | None] = mapped_column(Float, nullable=True)
    quarterly_eps_year_ago_period: Mapped[str | None] = mapped_column(
        String(10), nullable=True
    )

    # --- A: Annual Earnings Growth (EODHD Earnings.Annual + Highlights) ---
    # JSON-encoded list of {"year": int, "eps": float}, most recent last —
    # kept as a small serialized field rather than a child table since it
    # is always read/written as one unit alongside its parent snapshot.
    annual_eps_history_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    return_on_equity_ttm: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- S: Supply and Demand (EODHD SharesStats) ---
    shares_float: Mapped[float | None] = mapped_column(Float, nullable=True)
    shares_outstanding: Mapped[float | None] = mapped_column(Float, nullable=True)
    percent_insiders: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- I: Institutional Sponsorship (EODHD Holders.Institutions + SharesStats) ---
    institutional_holders_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    percent_institutions: Mapped[float | None] = mapped_column(Float, nullable=True)

    # --- Freshness bookkeeping (see module docstring) ---
    fundamental_period: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # e.g. "2026-Q2"
    fundamentals_as_of: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    fetched_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=text("now()")
    )
