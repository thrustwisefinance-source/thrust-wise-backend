# Thrustwise Finance — Backend

FastAPI backend serving the Thrustwise Finance UI. Ingests end-of-day OHLCV
data for six ETFs (VOO, VTI, SPY, QQQ, GLD, SHY) from EODHD into Postgres,
computes analytics with a rule-based engine, and caches responses in Redis.

## API

All responses use the frontend's `ApiResponse<T>` envelope:
`{ "data": ..., "success": true, "message": null }` with camelCase keys.

| Endpoint | Returns |
|---|---|
| `GET /api/etfs?category=Equity` | `EtfSnapshot[]` for Explorer cards (chips: All, Equity, Index, Technology, Gold, Treasury) |
| `GET /api/etfs/{symbol}` | Full `EtfDetails` (stats, risk metrics, holdings, education) |
| `GET /api/etfs/{symbol}/quote` | Latest EOD quote |
| `GET /api/etfs/{symbol}/performance` | `Record<"1M"\|"3M"\|"6M"\|"1Y"\|"5Y", PerformancePoint[]>` for Recharts |
| `GET /api/etfs/compare?symbols=VOO,SPY` | Side-by-side comparison entries (2-4 symbols) |
| `GET /api/dashboard` | Aggregate summary (top/worst performer, snapshots) |
| `GET /api/health` | Liveness + ingested row count |

The frontend reads the backend origin from `NEXT_PUBLIC_API_BASE_URL`
(never hardcoded):

- Local dev: `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api` in the
  frontend's `.env.local`
- Vercel (https://thrustwise-finance-ui.vercel.app): set
  `NEXT_PUBLIC_API_BASE_URL=https://<deployed-backend-host>/api` in the
  Vercel project's environment variables once the backend is hosted, then
  redeploy (NEXT_PUBLIC_ vars are baked in at build time).

## Deployment configuration

All configuration comes from environment variables (see `.env.example`);
nothing is hardcoded in the repo and `.env` is gitignored. For a hosted
backend set:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Company Postgres, `postgresql+asyncpg://...` |
| `REDIS_URL` | Company Redis (optional — API degrades gracefully without it) |
| `EODHD_API_TOKEN` | Company EODHD account token |
| `CORS_ORIGINS` | Must include `https://thrustwise-finance-ui.vercel.app` |
| `ENVIRONMENT` | `production` (disables SQL echo logging) |

## Running locally

### Option A: Docker Compose (Postgres + Redis + API)

```bash
docker compose up --build
```

### Option B: Bare metal

1. Start Postgres and Redis (e.g. `docker compose up db redis`).
2. Create a venv and install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in `EODHD_API_TOKEN` and `DATABASE_URL`.
4. Run migrations (optional in dev — startup also runs `create_all`):
   ```bash
   alembic upgrade head
   ```
5. Start the API:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

On first boot the app backfills 5 years of daily history from EODHD in the
background (a minute or so for 6 symbols), then refreshes nightly at
22:30 UTC on weekdays. Until a symbol's backfill lands, its endpoints
return `503`.

## Architecture

- `app/constants.py` — ETF registry mirroring the frontend's `constants/etfs.ts`,
  plus static holdings/education content.
- `app/models/` — SQLAlchemy models: `etf_metadata`, `daily_prices`
  (unique on symbol+date).
- `app/schemas/` — Pydantic models with camelCase aliases matching the
  TypeScript interfaces.
- `app/services/eodhd_client.py` — async EODHD EOD API wrapper.
- `app/services/ingestion.py` — startup backfill + APScheduler nightly refresh;
  invalidates cache after each run.
- `app/services/analytics.py` — computes snapshots, quotes, chart series
  (downsampled to ≤30 points/range), performance stats (YTD/1Y/3Y/5Y/CAGR),
  and risk metrics (annualized volatility, beta vs SPY, Sharpe, 1-10 risk score).
- `app/services/cache.py` — Redis JSON cache, degrades gracefully if Redis
  is down; TTL 24h (EOD data changes once per trading day).
