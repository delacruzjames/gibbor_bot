"""Update timestamp column to DateTime and value column to String

Revision ID: 36b7eb5f11db
Revises: 64b01c0ef413
Create Date: 2024-11-22 16:30:06.298026

"""
from alembic import op
import sqlalchemy as sa
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '36b7eb5f11db'
down_revision: Union[str, None] = '64b01c0ef413'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade the database schema by altering the `value` column in `prices`
    from `DOUBLE PRECISION` to `String`, and the `timestamp` column from
    `VARCHAR` to `DateTime`.
    """
    # Alter `value` column to String
    op.alter_column(
        'prices',
        'value',
        existing_type=sa.DOUBLE_PRECISION(precision=53),
        type_=sa.String(),
        existing_nullable=False
    )

    # Explicitly cast the `timestamp` column using a SQL expression
    op.execute(
        """
        ALTER TABLE prices 
        ALTER COLUMN timestamp 
        TYPE TIMESTAMP WITHOUT TIME ZONE 
        USING timestamp::timestamp without time zone
        """
    )


def downgrade() -> None:
    """
    Downgrade the database schema by reverting the `value` column in `prices`
    from `String` to `DOUBLE PRECISION`, and the `timestamp` column from
    `DateTime` to `VARCHAR`.
    """
    # Revert `timestamp` column to VARCHAR
    op.alter_column(
        'prices',
        'timestamp',
        existing_type=sa.DateTime(),
        type_=sa.VARCHAR(),
        existing_nullable=False
    )

    # Revert `value` column to DOUBLE PRECISION
    op.alter_column(
        'prices',
        'value',
        existing_type=sa.String(),
        type_=sa.DOUBLE_PRECISION(precision=53),
        existing_nullable=False
    )
