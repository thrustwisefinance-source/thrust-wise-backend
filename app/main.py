import asyncio
import logging
from contextlib import asynccontextmanager


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession


from app.config import settings


from app.database import Base, engine, get_db
from app.exceptions import register_exception_handlers
from app.middleware import RequestIDMiddleware
from app.routers import admin, canslim, compare, dashboard, etfs, health, strategy
from app.services import cache, ingestion


from pythonjsonlogger import jsonlogger


logger = logging.getLogger()
logger.setLevel(logging.INFO)


log_handler = logging.StreamHandler()
if settings.environment == "production":
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
else:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
log_handler.setFormatter(formatter)
logger.addHandler(log_handler)


# Silence httpx/httpcore — they log full request URLs at INFO, which
# includes the EODHD API token as a query parameter.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)
logger.info("========== CORS DEBUG ==========")
logger.info("Loaded CORS_ORIGINS: %s", settings.cors_origins)
logger.info("Parsed CORS list: %s", settings.cors_origins_list)
logger.info("================================")


# --- Rate limiter ---
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])



async def _startup_ingestion() -> None:
    """Backfill history on first boot; no-op when data is current."""
    try:
        await ingestion.ingest_all(trigger="startup")
        logger.info("Startup ingestion complete")
    except Exception:  # noqa: BLE001
        logger.exception("Startup ingestion failed — API will serve stored data")



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Dev convenience; production migrations run through Alembic
    if settings.auto_create_tables:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


    # Don't block startup on the EODHD backfill
    ingest_task = asyncio.create_task(_startup_ingestion())


    scheduler = None
    if settings.run_scheduler:
        scheduler = ingestion.create_scheduler()
        scheduler.start()


    yield


    if scheduler is not None:
        scheduler.shutdown(wait=False)
    ingest_task.cancel()
    await cache.close()
    await engine.dispose()



app = FastAPI(
    title="Thrustwise Finance API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)


# --- Rate limiting ---
app.state.limiter = limiter



@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "data": None,
            "success": False,
            "message": "Rate limit exceeded. Please slow down.",
        },
    )


# --- Exception handlers (error envelope) ---
register_exception_handlers(app)


# --- Middleware ---
app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Admin-Token", "X-Request-ID"],
)


# --- Routers ---
# /etfs/compare must be registered before /etfs/{symbol}
app.include_router(compare.router, prefix="/api")
app.include_router(etfs.router, prefix="/api")
app.include_router(strategy.router, prefix="/api")
app.include_router(canslim.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(admin.router, prefix="/api")
app.include_router(health.router)


# --- Route alias: /api/compare → same handler as /api/etfs/compare ---
from app.routers.compare import _compare_handler  # noqa: E402
from app.schemas import ApiResponse, CompareResult  # noqa: E402
from fastapi import Depends, Query  # noqa: E402


@app.get(
    "/api/compare",
    response_model=ApiResponse[CompareResult],
    tags=["compare"],
)
async def compare_alias(
    symbols: str = Query(..., description="Comma-separated, e.g. VOO,SPY,QQQ"),
    db: AsyncSession = Depends(get_db),
):
    """Alias for /api/etfs/compare — frontend compatibility."""
    return await _compare_handler(symbols=symbols, db=db)


# --- Root routes ---
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Thrustwise Finance API",
        "version": "1.0.0",
        "status": "ok",
        "docs": "/api/docs",
        "health": "/health/live",
    }


@app.get("/api", include_in_schema=False)
async def api_root():
    return {
        "service": "Thrustwise Finance API",
        "version": "1.0.0",
        "endpoints": [
            "/api/etfs",
            "/api/etfs/{symbol}",
            "/api/etfs/{symbol}/quote",
            "/api/etfs/{symbol}/performance",
            "/api/etfs/{symbol}/strategies",
            "/api/etfs/compare",
            "/api/compare",
            "/api/strategies/canslim",
            "/api/dashboard",
            "/api/dashboard/market",
            "/health/live",
            "/health/ready",
        ],
        "docs": "/api/docs",
    }