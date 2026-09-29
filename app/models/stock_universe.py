"""
SQLAlchemy model for stock-universe membership (e.g. "sp500", "nasdaq100").

This is the persisted equivalent of app.constants.ETF_REGISTRY, but for
individual equities used by the CANSLIM stock scanner
(app/scanners/universe.py) — NOT for the ETF Explorer/strategies. Rows
are ingested from EODHD's index-fundamentals endpoint (see
app.scanners.ingestion.refresh_universe) rather than hard-coded, so the
constituent list can be refreshed as index membership changes without a
code deploy.

`is_current` lets a refresh mark stale rows (symbols dropped from the
index) without losing history — a refresh flips every existing row for
the universe to `is_current=False` first, then upserts the current
constituent list back to `True`, so a mid-refresh failure never leaves
the table silently half-updated with no way to tell old from new.
"""

import datetime

from sqlalchemy import Boolean, DateTime, String, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class StockUniverseMember(Base):
    __tablename__ = "stock_universe_members"
    __table_args__ = (
        UniqueConstraint(
            "universe", "symbol", name="uq_stock_universe_members_universe_symbol"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    universe: Mapped[str] = mapped_column(String(30), nullable=False)  # e.g. "sp500"
    symbol: Mapped[str] = mapped_column(String(10), nullable=False)  # plain, e.g. "AAPL"
    name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(150), nullable=True)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False, default="US")
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    source: Mapped[str] = mapped_column(
        String(30), nullable=False, default="eodhd_index_fundamentals"
    )
    updated_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=text("now()")
    )

    def eodhd_symbol(self) -> str:
        return f"{self.symbol}.{self.exchange}"
