"""
Audit trail for ingestion runs (startup, scheduled, and admin-triggered).
Matches the table created in alembic revision 0002.
"""

import datetime

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"
    __table_args__ = (Index("ix_ingestion_runs_started_at", "started_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trigger: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # startup|schedule|manual
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # running|success|partial|failed
    symbols_ok: Mapped[int] = mapped_column(Integer, server_default="0", nullable=False)
    symbols_failed: Mapped[int] = mapped_column(
        Integer, server_default="0", nullable=False
    )
    rows_inserted: Mapped[int] = mapped_column(
        Integer, server_default="0", nullable=False
    )
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
