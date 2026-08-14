from typing import AsyncGenerator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Hosted Postgres URLs (Neon) arrive in libpq format:
#   postgresql://...?sslmode=require&channel_binding=require
# asyncpg and psycopg2 disagree on those params, so normalize per driver
# instead of requiring users to hand-edit the connection string.


def asyncpg_url(url_str: str) -> str:
    """App-runtime URL: asyncpg driver, no SSL-related query params at all.

    SQLAlchemy's asyncpg dialect forwards every remaining query-string key
    verbatim as a kwarg to asyncpg.connect() (`opts.update(url.query)`), so
    leaving any ssl-related key in the query string is fragile — asyncpg
    only accepts `ssl=<bool | ssl.SSLContext>`, never `sslmode=<str>`.
    SSL is decided by `asyncpg_ssl_required()` below and passed explicitly
    through `connect_args` instead, so nothing SSL-related survives here.
    """
    url = make_url(url_str.strip().strip("'\""))
    query = dict(url.query)
    for key in ("channel_binding", "sslmode", "ssl", "sslrootcert", "sslcert", "sslkey"):
        query.pop(key, None)
    url = url.set(drivername="postgresql+asyncpg", query=query)
    return url.render_as_string(hide_password=False)


def asyncpg_ssl_required(url_str: str) -> bool:
    """Whether the original URL requested SSL (from a `sslmode`/`ssl` query
    param), used to build asyncpg's `connect_args["ssl"]` explicitly."""
    url = make_url(url_str.strip().strip("'\""))
    query = dict(url.query)
    mode = query.get("sslmode") or query.get("ssl")
    if mode is None:
        return False
    return str(mode).strip().lower() not in ("disable", "false", "0", "off", "")


def sync_migration_url(url_str: str) -> str:
    """Alembic URL: psycopg2 driver, sslmode= param, fail-fast timeouts."""
    url = make_url(url_str.strip().strip("'\""))
    query = dict(url.query)
    ssl = query.pop("ssl", None)
    if ssl and "sslmode" not in query:
        # asyncpg accepts booleans; libpq wants a mode name
        query["sslmode"] = "require" if ssl in ("true", "True", "1") else ssl
    # Never hang a deploy on TCP/TLS: bound connection time (libpq param).
    # Statement/lock timeouts are SET explicitly in alembic/env.py — putting
    # them in ?options= breaks libpq's URI parser (space encodes as '+').
    query.setdefault("connect_timeout", "10")
    url = url.set(drivername="postgresql", query=query)
    return url.render_as_string(hide_password=False)


_app_db_url = asyncpg_url(settings.database_url)

# Fail fast instead of hanging when the DB is unreachable
_connect_args: dict = {"timeout": 10, "command_timeout": 30}
if asyncpg_ssl_required(settings.database_url):
    _connect_args["ssl"] = True
if "-pooler" in (make_url(_app_db_url).host or ""):
    # Neon's pooled endpoint (PgBouncer, transaction mode) breaks asyncpg's
    # prepared-statement cache; disable it so pooled URLs work too
    _connect_args["statement_cache_size"] = 0

engine = create_async_engine(
    _app_db_url,
    echo=settings.environment == "development",
    pool_pre_ping=True,
    connect_args=_connect_args,
    # Serverless Postgres (Neon) suspends compute when idle, killing pooled
    # connections; recycle them proactively so requests never inherit one.
    pool_recycle=300,
    pool_size=5,
    max_overflow=5,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session