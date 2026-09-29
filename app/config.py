from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    eodhd_api_token: str

    environment: str = "development"
    cors_origins: str = "http://localhost:3000"

    # Cache TTL — EOD data is stale after 24 h on trading days
    cache_ttl_seconds: int = 86400
    # How far back to seed historical OHLCV data
    ingestion_history_years: int = 5

    # Annual risk-free rate used for Sharpe ratio (decimal, e.g. 0.045 = 4.5%)
    risk_free_rate: float = 0.045

    # --- Operational gates ---
    # Set to false in production; Alembic owns the schema there.
    auto_create_tables: bool = True
    # Set to false for extra replicas so only one instance runs the scheduler.
    run_scheduler: bool = True
    # When set, enables POST /api/admin/refresh guarded by X-Admin-Token header.
    # When empty, the admin endpoint returns 404 (invisible).
    admin_api_token: str = ""

    # --- CANSLIM stock scanner ingestion (app/scanners/) ---
    # Comma-separated universe keys (see app.scanners.universe.UNIVERSE_REGISTRY)
    # to refresh automatically on the nightly scanner schedule below.
    # Empty by default: a ~500-stock universe refresh is a meaningfully
    # larger/slower job than the six-ETF nightly ingestion, so it is
    # opt-in rather than running unconditionally on every deploy. Use
    # POST /api/admin/scanners/refresh for an on-demand first ingestion.
    scanner_universes_to_ingest: str = ""
    # Set to false for extra replicas so only one instance runs the
    # scanner scheduler (same convention as run_scheduler above).
    run_scanner_scheduler: bool = False

    @property
    def scanner_universes_to_ingest_list(self) -> list[str]:
        return [u.strip() for u in self.scanner_universes_to_ingest.split(",") if u.strip()]

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]


settings = Settings()
