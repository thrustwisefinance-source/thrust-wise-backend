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

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # IF NOT EXISTS (PostgreSQL): safe if the column was already added
    # out-of-band. Nullable, no default => metadata-only change, existing rows
    # are not rewritten and simply read back NULL.
    op.execute(
        "ALTER TABLE stock_fundamentals_snapshots "
        "ADD COLUMN IF NOT EXISTS valuation_inputs_json TEXT"
    )


def downgrade() -> None:
    op.drop_column("stock_fundamentals_snapshots", "valuation_inputs_json")
