"""add stock_universe_members and stock_fundamentals_snapshots (CANSLIM stock scanner)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-29

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "stock_universe_members",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("universe", sa.String(length=30), nullable=False),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=True),
        sa.Column("sector", sa.String(length=100), nullable=True),
        sa.Column("industry", sa.String(length=150), nullable=True),
        sa.Column("exchange", sa.String(length=20), nullable=False, server_default="US"),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "source",
            sa.String(length=30),
            nullable=False,
            server_default="eodhd_index_fundamentals",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "universe", "symbol", name="uq_stock_universe_members_universe_symbol"
        ),
    )
    op.create_index(
        "ix_stock_universe_members_universe_current",
        "stock_universe_members",
        ["universe", "is_current"],
    )

    op.create_table(
        "stock_fundamentals_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("symbol", sa.String(length=10), nullable=False),
        sa.Column("company_name", sa.String(length=200), nullable=True),
        sa.Column("sector", sa.String(length=100), nullable=True),
        sa.Column("industry", sa.String(length=150), nullable=True),
        sa.Column("quarterly_eps_current", sa.Float(), nullable=True),
        sa.Column("quarterly_eps_current_period", sa.String(length=10), nullable=True),
        sa.Column("quarterly_eps_year_ago", sa.Float(), nullable=True),
        sa.Column("quarterly_eps_year_ago_period", sa.String(length=10), nullable=True),
        sa.Column("annual_eps_history_json", sa.Text(), nullable=True),
        sa.Column("return_on_equity_ttm", sa.Float(), nullable=True),
        sa.Column("shares_float", sa.Float(), nullable=True),
        sa.Column("shares_outstanding", sa.Float(), nullable=True),
        sa.Column("percent_insiders", sa.Float(), nullable=True),
        sa.Column("institutional_holders_count", sa.Integer(), nullable=True),
        sa.Column("percent_institutions", sa.Float(), nullable=True),
        sa.Column("fundamental_period", sa.String(length=20), nullable=True),
        sa.Column("fundamentals_as_of", sa.Date(), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.UniqueConstraint(
            "symbol", name="uq_stock_fundamentals_snapshots_symbol"
        ),
    )


def downgrade() -> None:
    op.drop_table("stock_fundamentals_snapshots")
    op.drop_index(
        "ix_stock_universe_members_universe_current",
        table_name="stock_universe_members",
    )
    op.drop_table("stock_universe_members")
