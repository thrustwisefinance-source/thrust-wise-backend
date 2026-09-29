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

Modules:
    universe.py   Stock-universe abstraction (get_stock_universe) backed
                  by app.models.StockUniverseMember, refreshed from EODHD
                  index fundamentals — NOT a hard-coded ticker list.
    data.py       Bulk/efficient EODHD data retrieval + extraction into
                  the typed inputs each CANSLIM criterion needs, backed
                  by app.models.StockFundamentalsSnapshot and the
                  existing app.models.DailyPrice table.
    canslim.py    Pure scoring functions per CANSLIM letter (no I/O),
                  implementing the source article's methodology.
    ingestion.py  Populates/refreshes the universe + fundamentals + price
                  tables from EODHD (separate from, but reusing, the
                  existing app.services.ingestion / eodhd_client).
"""
