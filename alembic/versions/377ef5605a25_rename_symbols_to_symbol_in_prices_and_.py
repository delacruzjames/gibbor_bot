"""Rename symbols to symbol in prices and trades tables

Revision ID: 377ef5605a25
Revises: afc744c8a121
Create Date: 2024-11-30 01:44:23.106250

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import reflection
import logging

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '377ef5605a25'
down_revision: Union[str, None] = 'afc744c8a121'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger('alembic.env')

def column_exists(table_name, column_name, connection):
    inspector = reflection.Inspector.from_engine(connection)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns

def upgrade() -> None:
    connection = op.get_bind()

    # Rename 'symbols' to 'symbol' in 'prices' table if 'symbols' exists
    if column_exists('prices', 'symbols', connection):
        op.alter_column('prices', 'symbols', new_column_name='symbol')
        logger.info("Renamed 'symbols' to 'symbol' in 'prices' table.")
    else:
        logger.info("'symbols' column does not exist in 'prices' table. Skipping rename.")

    # Rename 'symbols' to 'symbol' in 'trades' table if 'symbols' exists
    if column_exists('trades', 'symbols', connection):
        op.alter_column('trades', 'symbols', new_column_name='symbol')
        logger.info("Renamed 'symbols' to 'symbol' in 'trades' table.")
    else:
        logger.info("'symbols' column does not exist in 'trades' table. Skipping rename.")


def downgrade() -> None:
    connection = op.get_bind()

    # Rename 'symbol' back to 'symbols' in 'prices' table if 'symbol' exists
    if column_exists('prices', 'symbol', connection):
        op.alter_column('prices', 'symbol', new_column_name='symbols')
        logger.info("Reverted 'symbol' to 'symbols' in 'prices' table.")
    else:
        logger.info("'symbol' column does not exist in 'prices' table. Skipping rename.")

    # Rename 'symbol' back to 'symbols' in 'trades' table if 'symbol' exists
    if column_exists('trades', 'symbol', connection):
        op.alter_column('trades', 'symbol', new_column_name='symbols')
        logger.info("Reverted 'symbol' to 'symbols' in 'trades' table.")
    else:
        logger.info("'symbol' column does not exist in 'trades' table. Skipping rename.")
