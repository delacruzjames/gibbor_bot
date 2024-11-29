"""Change 'value' column in 'prices' table from String to Float

Revision ID: 13dd022b3ce4
Revises: 11dcfbe8b2da
Create Date: 2024-11-29 17:24:27.781689

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '13dd022b3ce4'
down_revision: Union[str, None] = '11dcfbe8b2da'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'prices',
        'value',
        existing_type=sa.String(),
        type_=sa.Float(),
        existing_nullable=False,
        postgresql_using='value::float'
    )


def downgrade() -> None:
    # Revert 'value' column back to String
    op.alter_column(
        'prices',
        'value',
        existing_type=sa.Float(),
        type_=sa.String(),
        existing_nullable=False,
        postgresql_using='value::varchar'
    )
