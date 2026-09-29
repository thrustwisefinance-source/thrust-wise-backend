"""
Free S&P 500 constituent-list source for the CANSLIM stock scanner.

This module name predates a later decision to remove EODHD from the
scanner pipeline entirely (kept as-is rather than renamed, to avoid
churning an unrelated import path) — it is NOT a fallback any more, it
is THE constituent source app.scanners.universe.refresh_universe uses
for "sp500", unconditionally. No EODHD call is made anywhere in that
path, hidden or otherwise; EODHD is used only by the separate OLD
ETF/index ingestion (app.services.ingestion, app.services.eodhd_client),
which this module does not import.

Wikipedia's "List of S&P 500 companies" page is community-maintained
against actual index changes and is the same underlying source many
other open tools use for this exact purpose. This module is never
consulted for per-stock fundamentals or prices — only for the
constituent list itself (those come from yfinance; see
app.scanners.yfinance_client / app.scanners.data).
"""

from __future__ import annotations

import io
import logging

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


async def fetch_sp500_constituents_from_wikipedia() -> list[dict]:
    """
    Returns [{Code, Name, Sector, Industry, Exchange}, ...] — the exact
    shape app.scanners.universe.refresh_universe iterates over.

    Raises httpx.HTTPStatusError / RuntimeError on failure; callers
    should refuse to wipe the existing universe on failure (see
    refresh_universe).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            WIKIPEDIA_SP500_URL,
            headers={
                "User-Agent": "ThrustWiseBackend/1.0 (scanner universe ingestion fallback)"
            },
        )
        response.raise_for_status()

    tables = pd.read_html(io.StringIO(response.text))
    df = tables[0]  # first table on that page is the constituents table

    out: list[dict] = []
    for _, row in df.iterrows():
        # Wikipedia uses "BRK.B" style; yfinance (and most US-equity data
        # vendors) use "BRK-B". Normalize the dot to a hyphen so symbols
        # line up with app.scanners.yfinance_client's ticker lookups.
        raw_symbol = str(row.get("Symbol", "")).strip().upper()
        symbol = raw_symbol.replace(".", "-")
        if not symbol:
            continue
        out.append(
            {
                "Code": symbol,
                "Name": str(row.get("Security", "")).strip() or None,
                "Sector": str(row.get("GICS Sector", "")).strip() or None,
                "Industry": str(row.get("GICS Sub-Industry", "")).strip() or None,
                "Exchange": "US",
            }
        )

    if not out:
        raise RuntimeError(
            "Wikipedia S&P 500 table fetched but parsed to zero rows — "
            "the page structure may have changed."
        )
    return out
