"""
Free fallback source for the S&P 500 constituent list.

Used only when EODHD's Index Components data returns 403 for GSPC.INDX
(see app.scanners.universe.refresh_universe) — which, per EODHD's own
docs, means the account's plan/add-ons don't include Index
Constituents data, not that anything in this codebase is misconfigured.
EODHD's regular EOD/fundamentals-per-symbol endpoints are unaffected by
this; only the index-level "Components" lookup is gated differently.

Wikipedia's "List of S&P 500 companies" page is community-maintained
against actual index changes and is the same underlying source many
other open tools use for this exact purpose. This is a fallback only:
EODHD is always tried first, and this module is never consulted for
per-stock fundamentals or prices — only for the constituent list itself.
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
    Returns [{Code, Name, Sector, Industry, Exchange}, ...] shaped like the
    subset of EODHD's index-fundamentals "Components" entries that
    app.scanners.universe.refresh_universe reads — a drop-in replacement
    for `list(payload["Components"].values())`.

    Raises httpx.HTTPStatusError / RuntimeError on failure; callers should
    treat that the same as an EODHD ingestion failure (refuse to wipe the
    existing universe — see refresh_universe).
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
        # Wikipedia uses "BRK.B" style; EODHD/most US-equity data vendors use
        # "BRK-B". Normalize the dot to a hyphen so symbols line up with the
        # rest of the pipeline (daily_prices, fundamentals lookups, etc.).
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