"""
Single source of truth for ETF metadata, mirroring the
frontend's constants/etfs.ts exactly so both sides stay in sync.
"""

from dataclasses import dataclass, field

ETF_SYMBOLS_US = [
    "GLD.US",
    "QQQ.US",
    "SHY.US",
    "SPY.US",
    "VOO.US",
    "VTI.US",
]

# Strip the .US suffix to get the plain symbol the frontend uses
PLAIN_SYMBOLS: list[str] = [s.replace(".US", "") for s in ETF_SYMBOLS_US]

# Benchmark used for beta calculations
BENCHMARK_SYMBOL = "SPY"


@dataclass(frozen=True)
class MarketIndexMeta:
    """A market index tracked for the dashboard (EODHD .INDX feed)."""

    symbol: str  # plain, e.g. "GSPC"
    name: str

    def eodhd_symbol(self) -> str:
        return f"{self.symbol}.INDX"


# Dashboard market-summary indices (stored in daily_prices like ETFs,
# but never exposed as explorable products)
INDEX_REGISTRY: dict[str, MarketIndexMeta] = {
    "GSPC": MarketIndexMeta(symbol="GSPC", name="S&P 500"),
    "IXIC": MarketIndexMeta(symbol="IXIC", name="NASDAQ Composite"),
    "DJI": MarketIndexMeta(symbol="DJI", name="Dow Jones Industrial Average"),
}

# Filter chips shown in the ETF Explorer
FILTER_CATEGORIES = ["All", "Equity", "Index", "Technology", "Gold", "Treasury"]


@dataclass(frozen=True)
class EtfStaticMeta:
    symbol: str  # plain, e.g. "VOO"
    name: str
    category: str
    issuer: str
    subtitle: str  # short description for cards
    risk_level: str  # "Low" | "Moderate" | "High"
    expense_ratio: float
    inception_date: str  # ISO date
    aum_usd: float  # USD
    dividend_yield: float
    fund_objective: str
    fund_type: str
    asset_class: str
    related_symbols: list[str]
    # Which Explorer filter chips this ETF belongs to
    tags: list[str] = field(default_factory=list)

    def eodhd_symbol(self) -> str:
        return f"{self.symbol}.US"


ETF_REGISTRY: dict[str, EtfStaticMeta] = {
    "VOO": EtfStaticMeta(
        symbol="VOO",
        name="Vanguard S&P 500 ETF",
        category="US Large-Cap Equity",
        issuer="Vanguard",
        subtitle="Low-cost exposure to the 500 largest US companies.",
        risk_level="Moderate",
        expense_ratio=0.03,
        inception_date="2010-09-07",
        aum_usd=450_000_000_000,
        dividend_yield=1.3,
        fund_objective=(
            "Tracks the S&P 500 Index, providing exposure to 500 of the "
            "largest publicly traded US companies."
        ),
        fund_type="Passive / Index Fund",
        asset_class="US Large-Cap Equity",
        related_symbols=["SPY", "VTI", "QQQ"],
        tags=["Equity", "Index"],
    ),
    "VTI": EtfStaticMeta(
        symbol="VTI",
        name="Vanguard Total Stock Market ETF",
        category="US Total Market Equity",
        issuer="Vanguard",
        subtitle="Broad exposure across the entire US stock market.",
        risk_level="Moderate",
        expense_ratio=0.03,
        inception_date="2001-05-24",
        aum_usd=380_000_000_000,
        dividend_yield=1.4,
        fund_objective=(
            "Tracks the CRSP US Total Market Index, covering large-, mid-, "
            "small-, and micro-cap stocks."
        ),
        fund_type="Passive / Index Fund",
        asset_class="US Total Market Equity",
        related_symbols=["VOO", "SPY", "QQQ"],
        tags=["Equity", "Index"],
    ),
    "SPY": EtfStaticMeta(
        symbol="SPY",
        name="SPDR S&P 500 ETF Trust",
        category="US Large-Cap Equity",
        issuer="State Street",
        subtitle="The original ETF tracking the S&P 500 index.",
        risk_level="Moderate",
        expense_ratio=0.0945,
        inception_date="1993-01-22",
        aum_usd=520_000_000_000,
        dividend_yield=1.25,
        fund_objective=(
            "Tracks the S&P 500 Index. The world's most liquid ETF and a "
            "benchmark for US large-cap equities."
        ),
        fund_type="Passive / Index Fund",
        asset_class="US Large-Cap Equity",
        related_symbols=["VOO", "VTI", "QQQ"],
        tags=["Equity", "Index"],
    ),
    "QQQ": EtfStaticMeta(
        symbol="QQQ",
        name="Invesco QQQ Trust",
        category="US Large-Cap Growth / Tech",
        issuer="Invesco",
        subtitle="Concentrated exposure to the Nasdaq-100's top tech names.",
        risk_level="High",
        expense_ratio=0.20,
        inception_date="1999-03-10",
        aum_usd=250_000_000_000,
        dividend_yield=0.6,
        fund_objective=(
            "Tracks the Nasdaq-100 Index, concentrated in technology and "
            "growth-oriented large-cap companies."
        ),
        fund_type="Passive / Index Fund",
        asset_class="US Large-Cap Growth / Technology",
        related_symbols=["VOO", "SPY", "VTI"],
        tags=["Equity", "Index", "Technology"],
    ),
    "GLD": EtfStaticMeta(
        symbol="GLD",
        name="SPDR Gold Shares",
        category="Commodities — Gold",
        issuer="State Street",
        subtitle="A convenient way to hold physical gold in a brokerage account.",
        risk_level="Moderate",
        expense_ratio=0.40,
        inception_date="2004-11-18",
        aum_usd=57_000_000_000,
        dividend_yield=0.0,
        fund_objective=(
            "Tracks the price of gold bullion. Each share represents ~1/10 "
            "of an ounce of gold."
        ),
        fund_type="Commodity / Physical Gold",
        asset_class="Commodities",
        related_symbols=["SHY", "VOO", "SPY"],
        tags=["Gold"],
    ),
    "SHY": EtfStaticMeta(
        symbol="SHY",
        name="iShares 1-3 Year Treasury Bond ETF",
        category="Short-Term US Treasuries",
        issuer="iShares",
        subtitle="Short-duration US Treasuries built for capital preservation.",
        risk_level="Low",
        expense_ratio=0.15,
        inception_date="2002-07-22",
        aum_usd=24_000_000_000,
        dividend_yield=4.8,
        fund_objective=(
            "Tracks an index of US Treasury bonds with remaining maturities "
            "between 1 and 3 years."
        ),
        fund_type="Passive / Fixed Income",
        asset_class="Short-Term US Treasuries",
        related_symbols=["GLD", "VOO", "VTI"],
        tags=["Treasury"],
    ),
}


# ---------------------------------------------------------------------------
# Static content not derivable from EODHD end-of-day data.
# Holdings are top-10 snapshots (approximate weights); education is the
# plain-language explainer shown on the details page.
# ---------------------------------------------------------------------------

ETF_HOLDINGS: dict[str, list[dict]] = {
    "VOO": [
        {"name": "Apple Inc.", "weight": 6.9},
        {"name": "Microsoft Corp.", "weight": 6.5},
        {"name": "NVIDIA Corp.", "weight": 6.1},
        {"name": "Amazon.com Inc.", "weight": 3.7},
        {"name": "Alphabet Inc.", "weight": 3.5},
        {"name": "Meta Platforms Inc.", "weight": 2.4},
        {"name": "Berkshire Hathaway Inc.", "weight": 1.7},
        {"name": "Broadcom Inc.", "weight": 1.6},
        {"name": "Tesla Inc.", "weight": 1.4},
        {"name": "Eli Lilly & Co.", "weight": 1.3},
    ],
    "VTI": [
        {"name": "Apple Inc.", "weight": 5.9},
        {"name": "Microsoft Corp.", "weight": 5.6},
        {"name": "NVIDIA Corp.", "weight": 5.2},
        {"name": "Amazon.com Inc.", "weight": 3.2},
        {"name": "Alphabet Inc.", "weight": 3.0},
        {"name": "Meta Platforms Inc.", "weight": 2.0},
        {"name": "Berkshire Hathaway Inc.", "weight": 1.4},
        {"name": "Broadcom Inc.", "weight": 1.4},
        {"name": "Tesla Inc.", "weight": 1.2},
        {"name": "Eli Lilly & Co.", "weight": 1.1},
    ],
    "SPY": [
        {"name": "Apple Inc.", "weight": 6.9},
        {"name": "Microsoft Corp.", "weight": 6.5},
        {"name": "NVIDIA Corp.", "weight": 6.1},
        {"name": "Amazon.com Inc.", "weight": 3.7},
        {"name": "Alphabet Inc.", "weight": 3.5},
        {"name": "Meta Platforms Inc.", "weight": 2.4},
        {"name": "Berkshire Hathaway Inc.", "weight": 1.7},
        {"name": "Broadcom Inc.", "weight": 1.6},
        {"name": "Tesla Inc.", "weight": 1.4},
        {"name": "Eli Lilly & Co.", "weight": 1.3},
    ],
    "QQQ": [
        {"name": "Apple Inc.", "weight": 8.8},
        {"name": "Microsoft Corp.", "weight": 8.3},
        {"name": "NVIDIA Corp.", "weight": 7.9},
        {"name": "Amazon.com Inc.", "weight": 5.4},
        {"name": "Broadcom Inc.", "weight": 5.1},
        {"name": "Meta Platforms Inc.", "weight": 4.9},
        {"name": "Alphabet Inc.", "weight": 4.8},
        {"name": "Tesla Inc.", "weight": 3.0},
        {"name": "Costco Wholesale Corp.", "weight": 2.6},
        {"name": "Netflix Inc.", "weight": 2.2},
    ],
    "GLD": [
        {"name": "Physical Gold Bullion", "weight": 100.0},
    ],
    "SHY": [
        {"name": "US Treasury Notes 1-2 Year", "weight": 55.0},
        {"name": "US Treasury Notes 2-3 Year", "weight": 43.0},
        {"name": "Cash & Equivalents", "weight": 2.0},
    ],
}

# ---------------------------------------------------------------------------
# Portfolio-comparison-only price sources.
#
# VT and VXUS are NOT explorable Explorer products (they are intentionally
# absent from ETF_REGISTRY), so they must not appear in the ETF Explorer,
# /etfs/{symbol} details, quote, compare, or allocation endpoints. They are
# needed purely as historical-price inputs for the "VT vs VTI + VXUS
# Portfolio Comparison" strategy (see services/portfolio_comparison.py),
# which also uses VTI's price history already covered by ETF_REGISTRY.
#
# Kept as a separate lightweight registry (mirroring the existing
# INDEX_REGISTRY pattern) so app.services.ingestion can pull daily bars for
# them the same way it does for every other symbol, without touching
# ETF_REGISTRY or any of the six existing technical-analysis strategies.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PortfolioComponentMeta:
    """Minimal metadata for a symbol that is only ever used as a raw
    historical-price input to a portfolio-analytics strategy — never
    rendered as a standalone Explorer product."""

    symbol: str  # plain, e.g. "VT"
    name: str

    def eodhd_symbol(self) -> str:
        return f"{self.symbol}.US"


PORTFOLIO_COMPARISON_REGISTRY: dict[str, PortfolioComponentMeta] = {
    "VT": PortfolioComponentMeta(symbol="VT", name="Vanguard Total World Stock ETF"),
    "VXUS": PortfolioComponentMeta(
        symbol="VXUS", name="Vanguard Total International Stock ETF"
    ),
}


# ---------------------------------------------------------------------------
# Strategy-only price sources (same pattern as PORTFOLIO_COMPARISON_REGISTRY
# above).
#
# TLT is NOT an explorable Explorer product (intentionally absent from
# ETF_REGISTRY), so it must not appear in the ETF Explorer, /etfs/{symbol}
# details, quote, compare, or allocation endpoints. It is needed purely as
# a historical-price input for the "TLT Monthly Cycle" calendar strategy
# (see services/strategies.py, compute_tlt_monthly_cycle), which is
# surfaced on the existing Strategies section for every symbol (the same
# way the cross-ETF Risk-On/Risk-Off strategy is), not as a standalone
# Explorer product.
#
# Kept as its own lightweight registry (mirroring PORTFOLIO_COMPARISON_
# REGISTRY) so app.services.ingestion pulls daily bars for it the same way
# it does for every other symbol, without touching ETF_REGISTRY or any of
# the existing technical-analysis strategies.
# ---------------------------------------------------------------------------

STRATEGY_ONLY_REGISTRY: dict[str, PortfolioComponentMeta] = {
    "TLT": PortfolioComponentMeta(
        symbol="TLT", name="iShares 20+ Year Treasury Bond ETF"
    ),
}


ETF_EDUCATION: dict[str, str] = {
    "VOO": (
        "VOO is often recommended as a core portfolio holding. By owning it "
        "you effectively own a slice of the 500 largest US companies at one "
        "of the lowest costs in the industry. Its returns closely mirror "
        "the overall US large-cap stock market."
    ),
    "VTI": (
        "VTI goes a step broader than an S&P 500 fund: it holds the entire "
        "investable US stock market, including mid-, small-, and micro-cap "
        "companies. It suits investors who want maximum diversification in "
        "a single US equity fund."
    ),
    "SPY": (
        "SPY was the first US-listed ETF and remains the most heavily traded "
        "security in the world. Its unrivaled liquidity makes it popular with "
        "active traders, though long-term investors often prefer cheaper "
        "alternatives like VOO that track the same index."
    ),
    "QQQ": (
        "QQQ concentrates on the 100 largest non-financial companies listed "
        "on the Nasdaq, which makes it heavily tilted toward technology. It "
        "has historically delivered higher returns than broad-market funds, "
        "but with noticeably larger swings along the way."
    ),
    "GLD": (
        "GLD lets you own gold without storing bars or coins: each share is "
        "backed by physical bullion held in vaults. Investors typically use "
        "it as a hedge against inflation and equity-market stress rather "
        "than as a growth holding — it pays no income."
    ),
    "SHY": (
        "SHY holds short-term US Treasury bonds, among the safest assets "
        "available. Its price barely moves compared to stocks, making it a "
        "common choice for parking cash, dampening portfolio volatility, or "
        "earning yield with minimal risk."
    ),
}