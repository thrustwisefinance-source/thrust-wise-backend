# Thrustwise Finance — Backend Transformation Plan (Phase 1)

**Date:** 2026-07-09
**Scope:** Public ETF research platform, Phase 1 (Explorer, Details, Compare, Market Dashboard). Rule-based analytics only. EODHD EOD data, 6 symbols.
**Repo audited:** `Thrustwise Finance/backend` (this repository).

---

## 0. Framing note

"Farm Fresh" in the planning brief refers to this repository itself — the
freshly created Thrustwise backend (2026-07-09) — not a separate legacy
codebase. The audit target is therefore this repo, which is already aligned to
the frontend contract.

Consequences for this plan:
- There is **no foreign domain logic to remove or isolate** (audit item 14: n/a).
- The gap analysis is **current backend vs. Phase-1 production requirements**.
- The correct strategy is **incremental hardening of a young, clean codebase**, not
  migration. Nothing here warrants a rewrite.

---

## 1. Backend Audit Summary

### 1.1 Project structure (current)

```
backend/
├── app/
│   ├── main.py            # FastAPI app, lifespan (create_all, seed, ingest, scheduler)
│   ├── config.py          # pydantic-settings, .env-driven
│   ├── constants.py       # ETF registry (6 symbols), filter tags, holdings, education
│   ├── database.py        # async engine + sessionmaker + Base
│   ├── models/            # DailyPrice, EtfMetadata
│   ├── schemas/           # camelCase Pydantic models matching TS interfaces
│   ├── services/          # eodhd_client (httpx), analytics (pandas), cache (redis), ingestion (APScheduler)
│   └── routers/           # etfs, compare, dashboard + deps helpers
├── alembic/               # env.py + 0001_initial migration
├── docker-compose.yml     # db (pg16), redis, api
├── Dockerfile             # single-stage python:3.12-slim, runs as root
├── requirements.txt       # pinned versions
└── README.md
```

### 1.2 Audit findings by area

| # | Area | State | Verdict |
|---|---|---|---|
| 1 | Structure | Clean layered FastAPI (routers → services → models) | **Keep** |
| 2 | Frameworks | FastAPI 0.111, SQLAlchemy 2 async, Pydantic v2, APScheduler, pandas, httpx, redis-py | **Keep** — matches target stack |
| 3 | DB models | `daily_prices` (unique symbol+date), `etf_metadata` | Keep prices; `etf_metadata` is **write-only dead weight** (never read — routers use the in-code registry) |
| 4 | Endpoints | `/api/etfs[?category]`, `/api/etfs/{symbol}`, `/quote`, `/performance`, `/api/etfs/compare`, `/api/dashboard`, `/api/health` | Working, verified against live data; naming/robustness gaps below |
| 5 | Middleware | CORS only | Gap: no request IDs, no timing, no rate limit |
| 6 | Auth/session | None (correct for Phase 1 — public read-only API) | Keep; add admin-token guard for refresh trigger only |
| 7 | Caching | Redis JSON w/ 60s circuit breaker, 24h TTL, prefix invalidation after ingest | **Good base**; no SWR, no stampede lock, no warmup |
| 8 | Scheduler | In-process APScheduler, cron 22:30 UTC Mon–Fri + startup backfill | Works, but **duplicates jobs under multi-worker/replicas**; no run auditing, no overlap guard |
| 9 | Config | pydantic-settings, all secrets in gitignored `.env`, none in code | **Keep**; add `ADMIN_API_TOKEN`, `RUN_SCHEDULER`, `AUTO_CREATE_TABLES` |
| 10 | Docker/CI | Dockerfile + compose exist; **no `.dockerignore`, no CI at all** | **Critical fix** (see 1.3) + build CI from scratch |
| 11 | Logging/errors | `logging.basicConfig`, plain text; FastAPI default error bodies | Gaps: error envelope mismatch, httpx URL logging leaks token, no structured logs |
| 12 | Tests | **None** | Must add before risky refactors |
| 13 | Reusable domain logic | Registry, schemas, analytics core, ingestion skeleton, cache — **~85% of repo is production-reusable** | Keep |
| 14 | Farm Fresh–specific logic | **None exists** | n/a |
| 15 | Debt/anti-patterns | Itemized below | — |

### 1.3 Defects and debt (evidence-based, ranked)

**Severity: critical**
1. **`Dockerfile` `COPY . .` with no `.dockerignore`** — a `docker build` today bakes
   `.env` (live EODHD token) and the 11k-file `.venv` into image layers. Anyone with
   image pull access gets the token. Fix same day.
2. **Zero commits** — no baseline, no reviewability, no rollback. Everything is
   untracked working files.

**Severity: high**
3. **Error responses violate the contract.** Success bodies use
   `{data, success, message}`; errors return FastAPI's `{"detail": ...}` with no
   `success: false`. Frontend error handling will misparse failures.
4. **`GET /etfs` is all-or-nothing** — `require_prices()` raises 503 mid-loop, so one
   symbol without data blanks the whole Explorer. Should degrade per-symbol.
5. **Scheduler multiplies with workers** — `uvicorn --workers 4` or 2 replicas ⇒ 4–2×
   nightly ingestions racing (idempotent inserts save the data, but EODHD calls and
   cache invalidations multiply). Needs `RUN_SCHEDULER` env gate.
6. **httpx logs full request URLs at INFO** — including `?api_token=...`. Token leaks
   into any INFO-level log sink. Silence `httpx`/`httpcore` loggers to WARNING.
7. **No tests, no CI** — the analytics engine (returns, beta, Sharpe, drawdown) is
   exactly the kind of code that regresses silently.

**Severity: medium**
8. **`create_all()` runs unconditionally at startup** — fights Alembic in prod; gate
   to development.
9. **CORS overly broad** — `allow_methods=["*"], allow_headers=["*"],
   allow_credentials=True` for a read-only, credential-free API. Minimize to
   `GET` + no credentials.
10. **Missing Phase-1 metrics** — max drawdown, correlation, allocation-comparison
    support are in scope but not implemented.
11. **No ingestion audit** — runs are invisible except in logs; no partial-failure
    state, no retries/backoff on EODHD calls.
12. **Health endpoint conflates live/ready** — single `/api/health` does a DB count;
    orchestrators need a dependency-free liveness probe and a readiness probe with
    freshness.
13. **`etf_metadata` table dead** — seeded every boot, read by nothing. Delete (the
    in-code registry is canonical) and reuse the slot for an `ingestion_runs` table.
14. **Redundant index** — `ix_daily_prices_symbol_date` duplicates the index implied
    by `uq_daily_prices_symbol_date`. Drop.
15. **No rate limiting** on a public API.

**Severity: low / cosmetic**
16. `category` filter: unknown chip returns empty 200 instead of 422.
17. `datetime.date.today()` uses server TZ; ingestion window math should be UTC-anchored.
18. Docker image runs as root, single-stage, no `HEALTHCHECK`.
19. Local dev is Python 3.10, image is 3.12 — standardize CI on 3.12.
20. Per-request `httpx.AsyncClient` construction in the EODHD client (6 calls/day — harmless, tidy later).
21. `/etfs` and `/dashboard` issue 6 sequential per-symbol queries (fine at n=6; not worth a window-function rewrite in Phase 1).

**Contract risks (carried from build):** `holdings` item shape (`{name, weight}`),
`education` (single string), `aum` (raw USD number), and the compare/dashboard
payloads were **designed, not confirmed** against the private frontend repo. Verify
against `constants/etf-details.ts` before freezing the contract.

---

## 2. Gap Analysis (current vs Phase-1 target)

### 2.1 Endpoint disposition

| Current endpoint | Target | Disposition |
|---|---|---|
| `GET /api/etfs?category=` | `GET /etfs` | **Reuse.** Add chip enum validation + per-symbol degradation |
| `GET /api/etfs/{symbol}` | `GET /etfs/{symbol}` | **Reuse** as-is |
| `GET /api/etfs/{symbol}/quote` | keep-or-fold decision | **Keep** (decision + rationale in §4.8) |
| `GET /api/etfs/{symbol}/performance` | same | **Reuse** as-is |
| `GET /api/etfs/compare?symbols=` | `GET /compare?symbols=` | **Reuse + alias.** One handler mounted at both paths until frontend confirms which it calls |
| `GET /api/dashboard` | `GET /dashboard/market` | **Rename + alias** (same pattern) |
| `GET /api/health` | `GET /health/live` + `GET /health/ready` | **Split**; keep `/api/health` as alias of ready |
| — (missing) | `POST /admin/refresh` | **Add**, admin-token-guarded, 202 + audit row |
| — (missing) | correlation / allocation-comparison support | **Add** to analytics + compare payload |

Nothing to delete, merge, or version-split. **Versioning:** defer URL versioning; the
frontend's base URL is env-configured, so `/api` → `/api/v1` is a config change at
deploy time. Adopt `/api/v1` when the backend is first hosted, not before.

### 2.2 Capability gaps

| Capability | Current | Gap |
|---|---|---|
| Ingestion | Backfill + nightly cron, idempotent upserts | Retries/backoff, integrity checks, audit table, run locking |
| Analytics | price/change, sparkline, range series, YTD/1Y/3Y/5Y/CAGR, vol, beta, Sharpe, risk score | max drawdown, correlation matrix, allocation comparison |
| Caching | Get/set/invalidate + circuit breaker | SWR semantics, post-ingest warmup, stampede lock |
| API hygiene | Pydantic response models, envelope on success | Envelope on error, request IDs, rate limits, structured logs |
| Ops | Dockerfile, compose | `.dockerignore`, non-root image, CI/CD, secret/dependency scans, staging flow |
| Tests | none | analytics unit tests, API integration tests, ingestion tests |

---

## 3. Target Architecture

Keep the current layering; add the missing operational tissue. No repositories
framework, no CQRS, no microservices — one deployable API + one optional worker flag.

```
backend/
├── app/
│   ├── main.py                  # app factory, lifespan, router mounting
│   ├── config.py                # settings (+ admin token, scheduler/table gates)
│   ├── constants.py             # ETF registry = single source of truth (unchanged)
│   ├── database.py              # async engine/session (unchanged)
│   ├── exceptions.py            # NEW: handlers → ApiResponse envelope on every error
│   ├── middleware.py            # NEW: X-Request-ID + access timing log
│   ├── models/
│   │   ├── daily_price.py       # keep (drop redundant index via migration)
│   │   └── ingestion_run.py     # NEW (replaces etf_metadata)
│   ├── repositories/
│   │   └── prices.py            # NEW: moved query helpers from routers/deps.py
│   ├── schemas/                 # unchanged + compare gains correlation fields
│   ├── services/
│   │   ├── eodhd_client.py      # + retry/backoff/timeout budget, silenced URL logs
│   │   ├── ingestion.py         # + audit rows, integrity checks, run lock
│   │   ├── analytics.py         # + max_drawdown, correlation, allocation compare
│   │   ├── cache.py             # + get_or_compute w/ per-key lock, warmup hook
│   │   └── scheduler.py         # split from ingestion; gated by RUN_SCHEDULER
│   └── routers/
│       ├── etfs.py / compare.py / dashboard.py   # tightened
│       ├── health.py            # NEW: /health/live, /health/ready
│       └── admin.py             # NEW: POST /admin/refresh
├── tests/
│   ├── unit/test_analytics.py   # deterministic price fixtures
│   ├── unit/test_cache.py
│   └── api/test_endpoints.py    # httpx ASGI client + Postgres service
├── alembic/versions/            # 0002: ingestion_runs, drop etf_metadata + dup index
├── .github/workflows/ci.yml     # lint, test, migration check, build, scans
├── .dockerignore                # NEW — before any image is ever built
├── Dockerfile                   # multi-stage, non-root, HEALTHCHECK
└── docs/BACKEND_TRANSFORMATION_PLAN.md
```

**Observability hooks:** request-ID middleware, JSON logs in production
(`python-json-logger`), latency measured in middleware and logged per route,
ingestion outcomes queryable via `ingestion_runs`, staleness surfaced in
`/health/ready`. Defer metrics servers (Prometheus) until there's somewhere to
scrape them.

---

## 4. Endpoint Contract Plan

Envelope everywhere, including errors:
`{ "data": T | null, "success": bool, "message": string | null }` (camelCase keys).
All public endpoints are read-only GET, rate-limited (default 60 req/min/IP via
slowapi; tune at the edge later), and validated by Pydantic response models.

### 4.1 `GET /api/etfs?category={chip}` — public
- **Purpose:** Explorer cards.
- **Params:** `category` optional, enum `All|Equity|Index|Technology|Gold|Treasury`; anything else → 422.
- **Response:** `ApiResponse<EtfSnapshot[]>`.
- **Caching:** `tw:v1:snapshot:all:{chip}`, TTL 26h, invalidated + rewarmed post-ingest.
- **Freshness:** EOD; stale acceptable until next nightly run.
- **Robustness:** symbol with no data is skipped, `message` notes the omission; empty DB → 503 with envelope.

### 4.2 `GET /api/etfs/{symbol}` — public
- **Params:** path symbol, uppercased, must be in registry → else 404 envelope.
- **Response:** `ApiResponse<EtfDetails>` (incl. `performanceStats`, `riskMetrics` — riskMetrics gains `maxDrawdown`).
- **Caching:** `tw:v1:details:{symbol}`, TTL 26h + post-ingest invalidation/warmup.

### 4.3 `GET /api/etfs/{symbol}/quote` — public
- Latest EOD OHLCV + change vs previous close. Cache `tw:v1:quote:{symbol}`.

### 4.4 `GET /api/etfs/{symbol}/performance` — public
- `Record<"1M"|"3M"|"6M"|"1Y"|"5Y", {label, value}[]>`, ≤ ~31 points/range. Cache per symbol.

### 4.5 `GET /api/etfs/compare?symbols=A,B[,C,D]` — public (alias: `GET /api/compare`)
- **Validation:** 2–4 distinct registry symbols; else 422.
- **Response:** `ApiResponse<EtfComparisonEntry[]>`; each entry gains `maxDrawdown`;
  payload gains a `correlationMatrix: number[][]` (aligned to request order) to
  support allocation-comparison UI.
- **Caching:** `tw:v1:compare:{sorted-symbols}` — sorted key ⇒ order-insensitive hit.

### 4.6 `GET /api/dashboard/market` — public (alias: `GET /api/dashboard`)
- Aggregate: `asOf`, movers, average expense ratio, snapshots. Cache `tw:v1:dashboard`.

### 4.7 Health + admin
- **`GET /health/live`** — public/infra. No dependencies; returns 200 if the process serves requests. Excluded from rate limits.
- **`GET /health/ready`** — infra. Checks DB connectivity and data freshness (latest `daily_prices.date` within 5 calendar days); 503 when not ready. Reports last ingestion run status.
- **`POST /api/admin/refresh`** — **admin-only.** Guard: `X-Admin-Token` header, constant-time compare against `ADMIN_API_TOKEN` env (endpoint returns 404 when the var is unset, so it cannot exist accidentally). Returns 202 + run ID; refuses (409) if a run is already active. Strictly rate-limited (e.g. 3/hour). This is the frontend-triggerable "refresh mechanism": ops or a protected proxy route can poke it; public users never can.

### 4.8 Decision: keep `/etfs/{symbol}/quote`?
**Keep it.** The frontend already calls it; it is one cached Redis read; and
`EtfDetails` already embeds price/change so the details page costs no extra call.
Folding it into `/etfs` would save nothing and break a shipped consumer. Revisit only
if Phase 2 introduces intraday quotes with a different provider path.

---

## 5. Data Model & Persistence

### 5.1 `daily_prices` — keep
- Surrogate `id` PK (keep; churn of moving to composite PK buys nothing).
- `UNIQUE (symbol, date)` — dedupe + the only index needed for `WHERE symbol = ? ORDER BY date` lookups. **Drop** the redundant explicit index in migration 0002.
- Add `created_at TIMESTAMPTZ DEFAULT now()` for ingest forensics.
- `volume BIGINT`, prices `FLOAT` (double precision) — adequate for analytics; NUMERIC unnecessary in Phase 1 (no accounting semantics).

### 5.2 `etf_metadata` — **drop**
Written every boot, read never; the code registry is canonical and versioned with the
API that interprets it. Delete model, seeding, and table (migration 0002). If BI ever
needs it, re-add as a materialized export — don't keep dead schema "just in case."

### 5.3 `ingestion_runs` — **add**
```
id BIGSERIAL PK
started_at / finished_at TIMESTAMPTZ
trigger  TEXT  (startup | schedule | manual)
status   TEXT  (running | success | partial | failed)
symbols_ok INT, symbols_failed INT, rows_inserted INT
detail   TEXT NULL          -- first error summary
INDEX (started_at DESC)
```
Powers `/health/ready`, the admin endpoint's 409 logic, and ops queries.

### 5.4 Derived metrics — do **not** precompute in Postgres
6 symbols × ~1,300 rows compute in milliseconds with pandas; Redis warmup after
ingestion gives precomputed behavior without a second storage schema to migrate.
Revisit only if symbol count grows 100×.

### 5.5 Async/pooling
Async engine + asyncpg already in place. Set explicit `pool_size=5,
max_overflow=5` (public read API, tiny fan-out), keep `pool_pre_ping=True`.
Production DDL happens only through Alembic (`AUTO_CREATE_TABLES=false`); CI runs
`alembic upgrade head` against a service Postgres to catch drift.

---

## 6. EODHD Ingestion & Freshness

**Client choice — deliberate deviation from the brief:** the official `eodhd`
library is synchronous; inside an async server it would block the event loop or need
`asyncio.to_thread` wrapping. The existing async httpx client hits the identical REST
endpoint (`/api/eod/{SYMBOL}.US`). **Keep httpx**, and add what the library also
wouldn't give us: retries and backoff. (If company policy mandates the official
client, wrap it in `to_thread` inside `eodhd_client.py` — the interface stays the
same either way; it's a one-file swap.)

Flow (per nightly run and startup):
1. Acquire run lock (Postgres advisory lock keyed on a constant — safe across replicas; APScheduler `max_instances=1` guards within-process).
2. Insert `ingestion_runs` row (`running`).
3. Per symbol (isolated try/except): window = `last_date + 1 → today (UTC)`; skip if empty. Bootstrap = today − (5y + 30d buffer).
4. Fetch with timeout 30s, 3 retries, exponential backoff (1s → 4s → 16s + jitter). 4xx = no retry (fail symbol); 5xx/timeout = retry.
5. **Integrity checks** before insert: drop rows where `close <= 0` or `high < low`; warn if a >5-trading-day gap appears; warn if `|close change| > 20%` day-over-day (split/bad-tick sentinel — log, don't block).
6. Idempotent insert: `ON CONFLICT (symbol, date) DO NOTHING` (already in place).
7. Finalize run row: `success` / `partial` (some symbols failed) / `failed` (all failed). Log one structured line per symbol: rows fetched/inserted/rejected.
8. On any inserts: invalidate `tw:v1:*` keys, then **warm** snapshot/details/performance/dashboard caches.

Symbol normalization stays where it is: plain symbols (`VOO`) everywhere internally;
`.US` suffix applied only at the EODHD call site via `meta.eodhd_symbol()`.

Freshness model: EOD data + 26h cache TTL + nightly refresh at 22:30 UTC. The
frontend's refresh mechanism reads `asOf` fields (already present in dashboard/quote)
and can surface "data as of {date}"; stale-while-revalidate below guarantees it never
waits on recompute.

---

## 7. Caching Plan (Redis)

- **Key convention:** `tw:v1:{resource}:{qualifier}` — the `v1` segment is bumped when payload shapes change, making stale-shape poisoning impossible across deploys.
  - `tw:v1:snapshot:all:{chip}` · `tw:v1:details:{sym}` · `tw:v1:quote:{sym}` · `tw:v1:performance:{sym}` · `tw:v1:compare:{sorted syms}` · `tw:v1:dashboard`
  - Yes — explorer list, details, compare, dashboard cache **separately**: different shapes, different invalidation cheapness, and per-key warmup.
- **TTL:** 26h (one daily cycle + slack) — the nightly invalidate-and-warm is the real freshness mechanism; TTL is the safety net.
- **Stale-while-revalidate:** post-ingest **warmup** makes the public path effectively always-warm. For cold-start/miss stampedes, `get_or_compute(key, fn)` takes a per-key `asyncio.Lock` (sufficient for a single instance; swap to Redis `SET NX PX=30000` lock when replicas > 1). Losers of the lock race wait for the winner's value rather than recomputing.
- **Redis down:** existing 60s circuit breaker stays — API computes from Postgres, sub-100ms verified. No sensitive payloads are cached (all data is public EOD).
- **Invalidation:** prefix-scan delete after each ingest run with inserts (already implemented), followed by warmup.

---

## 8. Security Hardening Checklist

**API**
- [x] Pydantic validation + `response_model` on every route (in place)
- [ ] Error envelope handlers (`HTTPException`, `RequestValidationError`, catch-all 500 with no internals leaked)
- [ ] `X-Request-ID` middleware; ID echoed in responses and logs
- [ ] slowapi rate limiting: 60/min/IP public, 3/hour admin refresh; health endpoints exempt
- [ ] Input normalization: symbols uppercased + registry-validated (in place); category enum-validated; compare list bounded 2–4 (in place)
- [ ] Pagination: n/a (6 fixed ETFs) — compare bound is the only fan-out limiter needed
- [ ] Public vs internal separation: only `/api/admin/*` is privileged; guarded by header token + 404-when-unconfigured + rate limit

**CORS / browser**
- [ ] `allow_origins` strict allowlist from env (in place: localhost + Vercel domain)
- [ ] `allow_methods=["GET", "POST"]` (POST solely for admin), `allow_headers=["Content-Type", "X-Admin-Token"]`, `allow_credentials=False` — no cookies exist in Phase 1

**Future auth readiness**
- No auth in Phase 1 (correct). When Phase 2 adds users: prefer short-lived JWT in `Authorization: Bearer` (no CSRF surface); if cookies are chosen instead, they must be `HttpOnly; Secure; SameSite=Lax` + CSRF tokens. Nothing in the current design blocks either.

**Database**
- [ ] Prod runtime role: `SELECT/INSERT/UPDATE/DELETE` on app tables only; DDL reserved for a migration role
- [ ] `AUTO_CREATE_TABLES=false` in prod; Alembic is the only schema authority
- [x] No raw SQL anywhere; all access via SQLAlchemy expressions (in place)
- [ ] Company-managed Postgres for hosting (per stakeholder constraint: no personal services)

**Redis**
- [ ] Private network only, never public; AUTH + TLS if managed provider
- [x] Only public EOD payloads cached (in place)

**Secrets**
- [x] `.env` gitignored, never committed (verified — repo has no commits yet)
- [ ] **`.dockerignore` excluding `.env`, `.venv`, `.git` — before any image build** (critical)
- [ ] Silence `httpx`/`httpcore` INFO logs (token appears in request-URL log lines)
- [ ] CI secrets via GitHub Actions encrypted secrets; EODHD token rotation = env swap + restart (confirm token is a company EODHD account, not personal — flagged previously)

**Logging/monitoring**
- [ ] JSON structured logs in prod; request ID + route + status + latency per request
- [ ] Ingestion job logs + `ingestion_runs` audit trail
- [ ] Staleness alert: `/health/ready` flips 503 when data > 5 days old — wire uptime monitor to it
- [ ] No secrets in logs (covered by httpx silencing + no token interpolation anywhere)

---

## 9. CI/CD + Ops Checklist (GitHub Actions)

Pipeline (`.github/workflows/ci.yml`), Python 3.12 to match the container:
1. **Lint/format:** `ruff check` + `ruff format --check`
2. **Secret scan:** gitleaks
3. **Dependency scan:** `pip-audit` (non-blocking warning initially, blocking after triage)
4. **Tests:** pytest with Postgres 16 + Redis 7 service containers; EODHD mocked — CI never spends API credits
5. **Migration check:** `alembic upgrade head` against service Postgres, then `alembic check` for model/migration drift
6. **Build:** `docker build` (multi-stage, non-root, HEALTHCHECK) — proves the image
7. **Deploy:** manual-approval staging job when hosting exists; deploy = new image tag; **rollback = redeploy previous tag** (DB migrations kept backward-compatible one release)

Release discipline: `main` protected, PRs require green CI; Alembic revisions
append-only.

---

## 10. Refactor Roadmap

**R0 — Same-day quick wins (no behavior risk)**
1. `git init` baseline commit *(prerequisite for everything — review, rollback, CI)*
2. `.dockerignore` (`.env`, `.venv`, `.git`, `__pycache__`, `docs`, `tests`) — **critical**
3. Silence `httpx`/`httpcore` loggers → WARNING — token leak
4. Exception handlers → error envelope (`exceptions.py`)
5. Tighten CORS (GET+admin-POST only, no credentials)
6. Gate `create_all` behind `AUTO_CREATE_TABLES` (default true in dev, false in prod)
7. Gate scheduler behind `RUN_SCHEDULER` (default true; false for extra replicas)
8. Validate `category` against chip enum
9. `/etfs` degrades per-symbol instead of 503-ing the list

**R1 — Short (1–3 days)**
10. Tests: analytics unit suite (fixed price fixtures asserting returns/vol/beta/Sharpe/drawdown numerically), API integration tests, ingestion integrity tests
11. CI workflow (§9)
12. `ingestion_runs` audit table + retry/backoff + integrity checks + advisory-lock run guard (migration 0002, incl. dropping `etf_metadata` + duplicate index)
13. Health split: `/health/live`, `/health/ready` (+freshness)
14. `POST /api/admin/refresh` with token guard
15. Rate limiting (slowapi)
16. Request-ID middleware + structured logging
17. Analytics additions: `max_drawdown` (riskMetrics + compare), `correlation_matrix` (compare payload)

**R2 — Medium (≤ 1 week)**
18. Cache `get_or_compute` with per-key locking + post-ingest warmup (SWR semantics)
19. Dockerfile: multi-stage, non-root `USER`, `HEALTHCHECK` on `/health/live`
20. Move query helpers `routers/deps.py` → `repositories/prices.py`; split `scheduler.py` from `ingestion.py`
21. Staging deploy job + uptime monitor on `/health/ready`
22. Frontend contract verification pass (holdings/education/aum/compare/dashboard shapes) once repo access exists — adjust schemas if actuals differ

**R3 — Structural (only when triggered)**
23. Separate ingestion worker process — trigger: >1 API replica
24. Redis-based distributed cache lock — trigger: >1 API replica
25. `/api/v1` prefix — trigger: first hosted deploy (frontend base URL is env-driven, so it's a config-time rename)

**Disposition summary** — *Keep:* structure, registry, schemas, analytics core, cache
core, compose, Alembic. *Wrap:* compare/dashboard route aliases, EODHD client behind
retry layer. *Rename:* `/dashboard` → `/dashboard/market` (aliased). *Split:*
health; scheduler-from-ingestion. *Remove:* `etf_metadata`, duplicate index. *Rewrite:*
nothing — no component is weak enough to justify it.

**Risky-change rule:** items 17, 18, and any `_return_since`/window-math edits land
only after R1 item 10 (tests) is merged.

---

## 11. Follow-up Implementation Prompt

> Copy-paste this to Claude Code to execute the plan:

```
Implement the Thrustwise Finance backend hardening plan in
docs/BACKEND_TRANSFORMATION_PLAN.md. Work in the backend repo root.
Execute phases R0 → R1 → R2 in order (roadmap §10); skip R3 (not yet triggered).

Ground rules:
- Preserve the public API contract: ApiResponse envelope with camelCase keys;
  do not change existing success payload shapes.
- New/renamed routes must keep the old paths as aliases (same handler):
  /api/etfs/compare + /api/compare; /api/dashboard + /api/dashboard/market;
  /api/health stays as an alias of /health/ready.
- All new config via pydantic-settings env vars with safe defaults:
  ADMIN_API_TOKEN (unset ⇒ admin endpoint 404s), RUN_SCHEDULER=true,
  AUTO_CREATE_TABLES=true locally / false in prod examples.
- Migration 0002 must: add ingestion_runs, drop etf_metadata (and its model,
  seeding code, and Alembic references), drop index ix_daily_prices_symbol_date,
  add created_at to daily_prices.
- Write tests before touching analytics math. Analytics tests use synthetic
  price fixtures with hand-computed expected values (returns, volatility, beta,
  sharpe, max drawdown, correlation). API tests run against Postgres via
  docker-compose db + fakeredis or the circuit-breaker path; mock EODHD with
  recorded JSON fixtures — never call the live API in tests.
- CI: .github/workflows/ci.yml per plan §9 (ruff, gitleaks, pip-audit, pytest
  with pg16+redis7 services, alembic upgrade+check, docker build).
- After each phase: run the test suite, boot the server, and smoke-test
  /api/etfs, /api/etfs/VOO, /api/etfs/compare?symbols=VOO,SPY,
  /api/dashboard/market, /health/live, /health/ready, and POST
  /api/admin/refresh (with and without token). Then commit with a conventional
  message (feat/fix/chore scope per phase).
- Definition of done: all R0+R1+R2 checklist items checked off in the plan doc
  (update the doc's checkboxes), tests green, image builds without .env inside
  (verify with: docker build + docker run --rm IMAGE ls -la /app | grep -v .env).
```

---

*Plan authored 2026-07-09. Repo state at audit: 0 commits, all files untracked,
endpoints verified live against EODHD data ingested into local Postgres 18.*
