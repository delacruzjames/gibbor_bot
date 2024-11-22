import os
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import your SQLAlchemy Base metadata
from main import Base  # Replace `main` with the module where your Base is defined

# Alembic configuration object
config = context.config

# Configure logging
fileConfig(config.config_file_name)

# Dynamically fetch the database URL from the environment variable
# Retrieve the DATABASE_URL environment variable
database_url = os.getenv("DATABASE_URL")

# Transform Heroku's default 'postgres://' to 'postgresql+psycopg2://'
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+psycopg2://", 1)

# Set the database URL dynamically in the Alembic config
config.set_main_option("sqlalchemy.url", database_url)


# Set the metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
