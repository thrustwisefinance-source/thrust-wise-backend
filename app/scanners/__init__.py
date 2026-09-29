"""
Stock scanners — cross-sectional screens over a stock universe (e.g.
S&P 500), architecturally separate from app/services/strategies.py
(single-symbol technical indicators for ThrustWise's six ETFs) and from
the legacy app/services/canslim.py (ETF-universe CANSLIM-like proxy
score, kept as-is for backward compatibility — see that module's
docstring).

    STRATEGIES  -> app/services/*.py, app/routers/strategy.py, app/routers/canslim.py
                   Single-symbol / ETF-universe analytics.

    SCANNERS    -> app/scanners/*.py, app/routers/scanners.py
                   Cross-sectional stock screens (CANSLIM today; the
                   package is structured so more screeners can be added
                   later without touching strategies code).

EODHD IS NOT USED ANYWHERE IN THIS PACKAGE. All stock-level data (S&P
500 constituent list, fundamentals, historical prices, including the
S&P 500 index itself for the M criterion) comes from yfinance/Yahoo
Finance and Wikipedia — see app.scanners.yfinance_client and
app.scanners.sp500_fallback. EODHD remains in use only by the separate,
untouched OLD ETF/index pipeline (app.services.ingestion,
app.services.eodhd_client) backing GET /api/strategies/* and the ETF
Explorer.

Modules:
    universe.py         Stock-universe abstraction (get_stock_universe)
                        backed by app.models.StockUniverseMember,
                        refreshed from Wikipedia's S&P 500 constituent
                        table (sp500_fallback.py) — NOT a hard-coded
                        ticker list, NOT EODHD.
    sp500_fallback.py   Free Wikipedia-based S&P 500 constituent-list
                        fetch used by universe.py.
    yfinance_client.py  Low-level, resilient async wrapper around the
                        synchronous `yfinance` library (retry/backoff,
                        bounded concurrency, per-symbol failure
                        isolation) — the package's sole stock-data HTTP
                        client.
    data.py             Per-symbol yfinance data retrieval + extraction
                        into the typed inputs each CANSLIM criterion
                        needs, backed by
                        app.models.StockFundamentalsSnapshot and the
                        existing app.models.DailyPrice table.
    canslim.py          Pure scoring functions per CANSLIM letter (no
                        I/O), implementing the source article's
                        methodology — unchanged by the EODHD -> yfinance
                        migration.
    ingestion.py        Populates/refreshes the universe + fundamentals
                        + price tables from yfinance (separate from, and
                        no longer reusing, app.services.ingestion /
                        eodhd_client).
"""
