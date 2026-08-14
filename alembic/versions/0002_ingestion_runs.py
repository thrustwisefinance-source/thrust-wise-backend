"""add ingestion_runs, drop etf_metadata, drop redundant index, add created_at

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-09

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add ingestion_runs audit table
    op.create_table(
        "ingestion_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trigger", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("symbols_ok", sa.Integer(), server_default="0", nullable=False),
        sa.Column("symbols_failed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rows_inserted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_ingestion_runs_started_at",
        "ingestion_runs",
        ["started_at"],
        unique=False,
    )

    # 2. Drop the etf_metadata table (dead — written every boot, never read)
    op.drop_table("etf_metadata")

    # 3. Drop redundant index on daily_prices (duplicates the unique constraint index)
    op.drop_index("ix_daily_prices_symbol_date", table_name="daily_prices")

    # 4. Add created_at column to daily_prices for ingest forensics
    op.add_column(
        "daily_prices",
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    # Reverse: remove created_at
    op.drop_column("daily_prices", "created_at")

    # Recreate redundant index
    op.create_index("ix_daily_prices_symbol_date", "daily_prices", ["symbol", "date"])

    # Recreate etf_metadata
    op.create_table(
        "etf_metadata",
        sa.Column("symbol", sa.String(length=10), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("issuer", sa.String(length=100), nullable=False),
        sa.Column("subtitle", sa.String(length=300), nullable=False),
        sa.Column("risk_level", sa.String(length=20), nullable=False),
        sa.Column("expense_ratio", sa.Float(), nullable=False),
        sa.Column("inception_date", sa.String(length=10), nullable=False),
        sa.Column("aum_usd", sa.Float(), nullable=False),
        sa.Column("dividend_yield", sa.Float(), nullable=False),
        sa.Column("fund_objective", sa.String(length=500), nullable=False),
        sa.Column("fund_type", sa.String(length=100), nullable=False),
        sa.Column("asset_class", sa.String(length=100), nullable=False),
        sa.Column("related_symbols", sa.String(length=100), nullable=False),
    )

    # Drop ingestion_runs
    op.drop_index("ix_ingestion_runs_started_at", table_name="ingestion_runs")
    op.drop_table("ingestion_runs")
