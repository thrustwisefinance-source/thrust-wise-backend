"""initial tables: etf_metadata, daily_prices

Revision ID: 0001
Revises:
Create Date: 2026-07-09

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
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

    op.create_table(
        "daily_prices",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open", sa.Float(), nullable=False),
        sa.Column("high", sa.Float(), nullable=False),
        sa.Column("low", sa.Float(), nullable=False),
        sa.Column("close", sa.Float(), nullable=False),
        sa.Column("adjusted_close", sa.Float(), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False, server_default="0"),
        sa.UniqueConstraint("symbol", "date", name="uq_daily_prices_symbol_date"),
    )
    op.create_index("ix_daily_prices_symbol_date", "daily_prices", ["symbol", "date"])


def downgrade() -> None:
    op.drop_index("ix_daily_prices_symbol_date", table_name="daily_prices")
    op.drop_table("daily_prices")
    op.drop_table("etf_metadata")
