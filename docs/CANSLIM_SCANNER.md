# CANSLIM Stock Scanner

## 1. What this is

A **stock scanner**, not a strategy. It screens a real stock universe
(S&P 500 today) against William O'Neil's CANSLIM methodology, using
actual EODHD fundamental and market data — never buy/sell/hold signals,
never fabricated data. Results are descriptive: which criteria passed,
failed, or couldn't be evaluated, with the underlying numbers and the
rule each one was checked against.

## 2. Separate from ETF strategy analytics

| | Strategies (existing) | Scanners (new) |
|---|---|---|
| Router | `app/routers/strategy.py`, `app/routers/canslim.py` | `app/routers/scanners.py` |
| Service code | `app/services/*.py` | `app/scanners/*.py` |
| Universe | ThrustWise's 6 ETFs (`app.constants.ETF_REGISTRY`) | S&P 500 (extensible; `app.scanners.universe`) |
| API | `/api/strategies/*` (untouched, still works) | `/api/scanners/*` (new) |

`app/services/canslim.py` and `GET /api/strategies/canslim` are the
**pre-existing, unrelated** ETF-universe CANSLIM-like proxy score. It is
left exactly as it was — not modified, not renamed, not repurposed.
Nothing in this document changes it or its behavior.

## 3. Where the CANSLIM methodology comes from

The supplied source article ("CANSLIM: William O'Neil's Battle-Tested
Framework..."). Every threshold that article states explicitly is
preserved exactly:

| Letter | Article's rule | Preserved as |
|---|---|---|
| C | ≥25% quarterly EPS growth YoY | `CANSLIM_C_MIN_QUARTERLY_EPS_GROWTH = 0.25` |
| A | ≥25% annual EPS growth over 3–5 years; ROE ideally >17% | `CANSLIM_A_MIN_ANNUAL_EPS_GROWTH = 0.25`, `CANSLIM_A_MIN_ROE = 0.17` |
| S | Float ideally under 25,000,000 shares | `CANSLIM_S_MAX_FLOAT_SHARES = 25_000_000` |
| L | RS Rating ≥ 80 (outperforming ≥80% of the market) | `CANSLIM_L_MIN_PERCENTILE = 80.0` |
| M | Market direction from the S&P 500 index vs. its 200-day SMA (article's own addendum) | `CANSLIM_M_SMA_PERIOD = 200`, evaluated on `GSPC.INDX` |

Where the article is **qualitative only** (no exact number given), this
codebase makes a documented, disclosed choice rather than inventing one
silently — see each criterion's docstring in `app/scanners/canslim.py`
for the full reasoning:

- **N ("near a 52-week high")** — the article's prose gives no
  percentage; its own reference Python script uses `price / 52w_high >=
  0.85`. That number is reused (attributed to the article's own code,
  not copied from the legacy ETF module, which coincidentally used the
  same figure for the same reason). The qualitative "new
  products/management" component is explicitly **not** scored.
- **I ("a few institutional investors")** — no number given anywhere in
  the article. `CANSLIM_I_MIN_INSTITUTIONAL_HOLDERS = 3` is used as a
  disclosed interpretive minimum, clearly labeled as such in every
  result's explanation text — not presented as an official CANSLIM
  figure. "Increasing sponsorship" and "over-owned" have no article
  threshold and are informational only, never gating pass/fail.
- **L's lookback window** — the article gives the 80th-percentile
  threshold but not the lookback period. 252 trading days (~12 months)
  is used, matching how IBD's own RS Rating is conventionally described
  (a documented choice, not a fabricated "article number").
- **A's growth basis** — "25% over 3–5 years" doesn't disambiguate
  annualized vs. total growth. This implementation uses **total**
  growth from the earliest to the latest EPS in the available 3-5 year
  window — a documented judgment call, stated in the module docstring.

## 4. What's directly calculated vs. what's not available

| Criterion | Calculated from | Limitation |
|---|---|---|
| C | EODHD `Earnings.History` | Zero/negative year-ago EPS → `NOT_APPLICABLE`, never a misleading percentage |
| A | EODHD `Earnings.Annual` + `Highlights.ReturnOnEquityTTM` | Requires ≥3 years of annual EPS on file |
| N | Ingested EODHD daily price history | Only the price-highs component; new products/management are **not evaluated** (no verifiable data source) |
| S | EODHD `SharesStats.SharesFloat` | Insider buying / buybacks are **not evaluated** — no ingested insider-transaction time series or historical shares-outstanding series |
| L | Ingested EODHD daily price history, ranked cross-sectionally | This is ThrustWise's **own calculated percentile**, explicitly never called an official IBD RS Rating |
| I | EODHD `Holders.Institutions` count + `SharesStats.PercentInstitutions` | "Increasing over time" and "over-owned" are informational only |
| M | Ingested S&P 500 index (`GSPC`) price history | Computed **once per scan**, applied identically to every stock — never per-stock |

Every criterion's status is one of `PASS` / `FAIL` / `UNAVAILABLE` /
`NOT_APPLICABLE`. Missing data is **never** silently converted to
`FAIL` — see `criteriaUnavailable` and `dataQuality` on each result.

## 5. Data source: EODHD

Reuses the existing `app/services/eodhd_client.py`, extended (not
duplicated) with:

- `fetch_fundamentals(symbol)` — `GET /api/fundamentals/{symbol}`
- `fetch_bulk_fundamentals(exchange, symbols)` — `GET /api/bulk-fundamentals/{exchange}?symbols=...`, used to fetch many stocks' fundamentals in one HTTP call (chunked at 50 symbols) instead of one request per symbol
- `fetch_index_fundamentals(index_symbol)` — same endpoint, used for an index's `Components` section

No `yfinance`. No fabricated fields. A field EODHD doesn't return for a
symbol stays `None` all the way through to the API response's
`UNAVAILABLE` status.

## 6. How the S&P 500 universe is obtained

`app/scanners/universe.py` does **not** hard-code a ticker list (unlike
the source article's own reference script). Instead:

1. `refresh_universe(db, "sp500")` calls EODHD's index-fundamentals
   endpoint for `GSPC.INDX`, whose `Components` section lists every
   current constituent with `{Code, Name, Sector, Industry, Exchange}`.
2. Constituents are upserted into `stock_universe_members`
   (`app.models.StockUniverseMember`), a full stale-then-fresh replace
   per universe so a partial refresh never leaves the table in an
   ambiguous state.
3. `get_stock_universe(db, "sp500")` reads the current, active members
   back out for the scanner to use.

Adding NASDAQ 100 or Russell 1000 means adding an entry to
`UNIVERSE_REGISTRY` (and, if EODHD has an index for it, its
`.INDX` symbol) — no changes to the scoring or router code.

## 7. How relative strength (L) is calculated

For every stock with ≥252 trading days of ingested price history,
compute its trailing ~12-month total return. Rank all such stocks in
the scanned universe by that return (best first) and convert each rank
to a percentile: the fraction of the rest of the universe it
outperformed. A stock at or above the 80th percentile passes L. This is
surfaced in every result as **"Calculated Relative Strength
Percentile"** — explicitly not an official IBD RS Rating.

## 8. How market direction (M) is calculated

Once per scan (not per stock): the S&P 500 index's (`GSPC`, already
ingested via `app.constants.INDEX_REGISTRY` for the existing dashboard)
own daily close vs. its own 200-day SMA. If the index is above its
200-day SMA, the market is a confirmed uptrend and every stock in the
scan gets `M = PASS`; otherwise every stock gets `M = FAIL`. This one
result is embedded in every stock's `criteria.m` and also surfaced once
at the top level of the scan response (`marketDirection`).

## 9. Data freshness

Every result carries:

- `priceDate` — the date of the most recent ingested daily bar used
- `fundamentalAsOf` — the fiscal period the fundamentals data covers
  (e.g. latest annual EPS's fiscal year-end, or latest quarter)
- `fundamentalFetchedAt` — when ThrustWise last pulled that
  fundamentals data from EODHD
- `dataUpdatedAt` — when this particular scan was computed

Quarterly/annual fundamentals are never presented as if they were as
fresh as the daily price feed.

## 10. Performance

A ~500-symbol scan makes:
- **~10 HTTP calls** for fundamentals (bulk-fundamentals, chunked at 50
  symbols/call) — not 500.
- **Up to 500 HTTP calls for price history**, one per symbol (EODHD has
  no bulk historical-range endpoint), run with bounded concurrency
  (`PRICE_INGESTION_CONCURRENCY = 8`) during ingestion, not on every
  scan request — the scan itself reads price history back out of
  Postgres (`daily_prices`, the same table ETF ingestion already uses).
- The computed scan is cached in Redis under `tw:v1:scanner:canslim:{universe}`
  (a dedicated namespace — never `tw:v1:screener:canslim`, which is the
  unrelated legacy ETF cache key) via the existing
  `app.services.cache.get_or_compute`.

## 11. API endpoints

| Method & path | Purpose |
|---|---|
| `GET /api/scanners/universes` | List supported universes and whether each has been ingested |
| `GET /api/scanners/canslim` | Full CANSLIM scan for a universe, with filters (`universe`, `min_criteria_passed`, `only_complete`, `sector`, `industry`, `limit`, `offset`) |
| `GET /api/scanners/canslim/{symbol}` | Single-symbol convenience lookup against the same cached scan |
| `POST /api/admin/scanners/refresh?universe=sp500` | Admin-only (same `X-Admin-Token` convention as `/api/admin/refresh`): ingest/refresh universe membership, fundamentals, and price history, then bust the scan cache |

`GET /api/strategies/canslim` (the legacy ETF proxy) is unchanged and
still works.

## 12. Ingestion into the database

Two new tables (Alembic revision `0003`):

- `stock_universe_members` — which symbols are in which universe, with
  sector/industry, refreshed from EODHD
- `stock_fundamentals_snapshots` — the specific EODHD fundamentals
  fields each CANSLIM criterion needs, with `fetched_at` /
  `fundamentals_as_of` for freshness

Stock daily price history reuses the **existing** `daily_prices` table
and `eodhd_client.fetch_eod_history` — no new price-storage model.

Ingestion is triggered via `POST /api/admin/scanners/refresh`, or
automatically on a nightly schedule if `RUN_SCANNER_SCHEDULER=true` and
`SCANNER_UNIVERSES_TO_INGEST=sp500` (both opt-in/off by default, since a
~500-stock refresh is materially heavier than the existing 6-ETF nightly
ingestion).

## 13. Known data limitations

- N's qualitative component (new products, services, management) is not
  evaluated — no verifiable data source.
- S's insider-buying and buyback sub-signals are not evaluated — no
  ingested insider-transaction time series or historical shares-outstanding
  series.
- I's "increasing sponsorship over time" is not evaluated — would
  require historical institutional-ownership snapshots, which are not
  ingested; only a current holder count/ownership percentage is used.
- L is ThrustWise's own calculated percentile, not IBD's proprietary RS
  Rating.
- A's "25% over 3-5 years" is implemented as total growth across the
  available window (a documented interpretation, see §3), not an
  annualized rate.
