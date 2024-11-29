"""Revert 'timestamp' column in 'prices' table back to DateTime

Revision ID: afc744c8a121
Revises: 13dd022b3ce4
Create Date: 2024-11-29 17:28:32.314821

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'afc744c8a121'
down_revision: Union[str, None] = '13dd022b3ce4'
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
    # Reapply the erroneous change if needed
    op.alter_column(
        'prices',
        'timestamp',
        existing_type=sa.DateTime(),
        type_=sa.String(),
        existing_nullable=False,
        postgresql_using='timestamp::varchar'
    )
