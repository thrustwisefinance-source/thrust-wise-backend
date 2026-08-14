# Deployment Runbook — Railway + Neon + Redis

Production topology: **Railway** runs the FastAPI container (built from the
`Dockerfile` per `railway.toml`), **Neon** hosts Postgres, **Redis** runs as a
Railway service (or Upstash). Neon and Redis need no build files — they are
configured entirely through environment variables on the Railway service.

## Neon (Postgres)

- Project: `thrustwise-finance`, region `ap-southeast-1` (Singapore).
- Schema is managed exclusively by Alembic; `railway.toml` runs
  `alembic upgrade head` before every boot. Never enable
  `AUTO_CREATE_TABLES` in production.
- **Connection string rules** (Neon's copied URL needs two edits):
  1. Scheme: `postgresql://` → `postgresql+asyncpg://`
  2. `sslmode=require` → `ssl=require` (asyncpg spelling; Alembic translates
     it back internally)
  - `channel_binding=require` may be left in or removed — the app strips it
    automatically (`app/database.py::_asyncpg_safe_url`), asyncpg would
    otherwise reject it.
- Use the **direct** (non-pooled) endpoint, not the `-pooler` one: the app's
  own pool is capped at 10 connections, and Neon's PgBouncer breaks asyncpg
  prepared statements.
- Free-tier autosuspend is handled in code (`pool_recycle=300` +
  `pool_pre_ping`); expect ~0.5s extra latency on the first query after idle.
- The database is fully rebuildable: an empty Neon DB + migrations + one
  startup ingestion (≈1 min) restores everything from EODHD. Backups are a
  nice-to-have, not critical, in Phase 1.

## Railway (API + Redis)

- Deploy the service in Railway's **Southeast Asia** region so it sits next to
  Neon Singapore (cross-region DB round-trips multiply cold-request latency).
- Add a **Redis** service in the same Railway project; reference its private
  URL. The API works without Redis (circuit breaker → compute from Postgres),
  but caching hides Neon latency after the first request.

### Required service variables

| Variable | Value |
|---|---|
| `DATABASE_URL` | Neon URL, transformed per the rules above |
| `REDIS_URL` | Railway Redis private URL (`redis://...railway.internal:6379`) or Upstash `rediss://` URL |
| `EODHD_API_TOKEN` | Company EODHD account token |
| `ENVIRONMENT` | `production` (JSON logs, no SQL echo) |
| `CORS_ORIGINS` | `https://thrustwise-finance-ui.vercel.app` (comma-append other origins as needed) |
| `AUTO_CREATE_TABLES` | `false` |
| `RUN_SCHEDULER` | `true` on exactly one instance; `false` on any extra replicas |
| `ADMIN_API_TOKEN` | Long random secret; unset ⇒ `/api/admin/refresh` returns 404 |

## Boot sequence (what happens on deploy)

1. Railway builds the Docker image (`.dockerignore` keeps `.env`, `.venv`,
   `.git` out of it).
2. `alembic upgrade head` applies any pending migrations against Neon.
3. Uvicorn starts; `/health/live` turns 200 (Railway healthcheck passes).
4. Startup ingestion backfills/tops-up daily prices from EODHD in the
   background; `/health/ready` turns 200 once data is present and fresh
   (≤5 days old).
5. Nightly refresh runs at 22:30 UTC Mon–Fri; caches are invalidated after
   each run.

## Frontend wiring (Vercel)

Set `NEXT_PUBLIC_API_BASE_URL=https://<railway-service-domain>/api` in the
Vercel project env vars, then **redeploy the frontend** (NEXT_PUBLIC_ vars are
baked at build time).

## Rollback

Deploy = new image. Rollback = redeploy the previous Railway deployment.
Keep migrations backward-compatible for one release so an old image can run
against a newer schema.
