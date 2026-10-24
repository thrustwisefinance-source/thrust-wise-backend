from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings
from app.database import Base, sync_migration_url
import app.models  # noqa: F401 — register models on Base.metadata

config = context.config

# Alembic runs sync (psycopg2); normalize whatever URL format was supplied
# (asyncpg-style ssl=, Neon's sslmode=/channel_binding=, either scheme).
# Escape % for configparser interpolation.
config.set_main_option(
    "sqlalchemy.url", sync_migration_url(settings.database_url).replace("%", "%%")
)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        # Bound every migration query so a hung DB fails the deploy fast
        # instead of blocking it forever
        connection.exec_driver_sql("SET statement_timeout = '30s'")
        connection.exec_driver_sql("SET lock_timeout = '15s'")
        # SQLAlchemy 2.0 autobegins a transaction on the statements above.
        # Alembic treats a pre-existing transaction as "external" and never
        # commits it, so without this commit `alembic upgrade head` logs every
        # migration yet rolls all of them back when the connection closes.
        # (SET is session-level, so the timeouts survive the commit.)
        connection.commit()
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
