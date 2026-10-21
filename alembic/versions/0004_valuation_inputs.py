"""add valuation_inputs_json to stock_fundamentals_snapshots

Graham valuation / Greenblatt Magic Formula need EBIT, balance-sheet
components, enterprise value inputs and a forward growth estimate, none of
which the existing snapshot columns hold. One nullable Text column
(JSON, with per-value period/provenance); no new table.

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-21

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "stock_fundamentals_snapshots",
        sa.Column("valuation_inputs_json", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("stock_fundamentals_snapshots", "valuation_inputs_json")
