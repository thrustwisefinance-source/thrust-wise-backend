"""
Request / response models for the Technical-Fundamental Composite Scoring
scanner (POST /api/scanners/quantitative_set3).

CamelModel: snake_case in Python, camelCase JSON; the request accepts either
spelling (``populate_by_name``).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, field_validator, model_validator

from app.schemas.common import CamelModel

SCANNER_ID = "technical_fundamental_composite_scoring"
MAX_SYMBOLS = 100
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")

ConditionStatus = Literal["passed", "failed", "unavailable", "not_meaningful"]
CalculationStatus = Literal["complete", "partial", "unavailable", "error"]


class ScannerRequest(CamelModel):
    symbols: list[str] = Field(default_factory=list, max_length=MAX_SYMBOLS)
    universe: str | None = Field(None, max_length=40)
    universe_limit: int = Field(25, ge=1, le=MAX_SYMBOLS)
    ma_period: int = Field(50, ge=2, le=400)
    rsi_period: int = Field(14, ge=2, le=100)
    rsi_upper_threshold: float = Field(70.0, gt=0, le=100)
    rsi_opportunity_threshold: float = Field(30.0, gt=0, lt=100)
    forward_pe_threshold: float = Field(20.0, gt=0, le=1000)
    debt_to_equity_threshold: float = Field(1.0, gt=0, le=100)
    use_cache: bool = True

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalise_symbols(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            v = [s for s in v.split(",")]
        out: list[str] = []
        for raw in v:
            if not isinstance(raw, str):
                raise ValueError("symbols must be strings")
            s = raw.strip().upper()
            if not s:
                continue
            if not _SYMBOL_RE.match(s):
                raise ValueError(
                    f"Invalid symbol '{raw}': use 1-10 characters of A-Z, 0-9, '.', '-'"
                )
            if s not in out:
                out.append(s)
        return out

    @model_validator(mode="after")
    def _check(self) -> "ScannerRequest":
        if not self.symbols and not self.universe:
            raise ValueError("Provide at least one symbol in 'symbols' or a 'universe'")
        if self.rsi_opportunity_threshold >= self.rsi_upper_threshold:
            raise ValueError(
                "rsi_opportunity_threshold must be lower than rsi_upper_threshold"
            )
        return self


class ConditionResult(CamelModel):
    name: str
    rule: str
    threshold: float
    value: float | None = None
    status: ConditionStatus
    passed: bool | None = None  # None = not evaluated (missing data)
    score: int | None = None  # 1 / 0, or None when the condition was not evaluated
    message: str | None = None


class OpportunityFlag(CamelModel):
    """Separate analytical flag - it never adds a point to the 0-4 score."""

    name: str = "rsi_below_opportunity_threshold"
    threshold: float
    rsi: float | None = None
    flagged: bool | None = None  # None = RSI unavailable
    message: str | None = None


class IndividualScores(CamelModel):
    price_above_ma: int | None = None
    rsi_below_upper: int | None = None
    forward_pe_below_threshold: int | None = None
    debt_to_equity_below_threshold: int | None = None


class ScannerDataAvailability(CamelModel):
    price_history_observations: int = 0
    required_price_observations: int = 0
    price_history_available: bool = False
    forward_pe_available: bool = False
    debt_to_equity_available: bool = False
    messages: list[str] = []


class EffectiveParameters(CamelModel):
    ma_period: int
    rsi_period: int
    rsi_upper_threshold: float
    rsi_opportunity_threshold: float
    forward_pe_threshold: float
    debt_to_equity_threshold: float


class StockScoreResult(CamelModel):
    symbol: str
    as_of_date: str | None = None
    current_price: float | None = None
    moving_average: float | None = None
    rsi: float | None = None
    forward_pe: float | None = None
    debt_to_equity: float | None = None
    debt_to_equity_provider_value: float | None = None
    price_above_ma: ConditionResult | None = None
    rsi_below_upper: ConditionResult | None = None
    forward_pe_below_threshold: ConditionResult | None = None
    debt_to_equity_below_threshold: ConditionResult | None = None
    individual_scores: IndividualScores = IndividualScores()
    total_score: int | None = None
    max_score: int = 4
    evaluated_conditions: int = 0
    score_complete: bool = False
    rsi_opportunity: OpportunityFlag | None = None
    data_availability: ScannerDataAvailability = ScannerDataAvailability()
    calculation_status: CalculationStatus
    error: str | None = None


class ScanSummary(CamelModel):
    requested: int
    complete: int
    partial: int
    unavailable: int
    error: int
    score_distribution: dict[str, int] = {}
    rsi_opportunity_count: int = 0


class ScannerResponse(CamelModel):
    scanner_id: str = SCANNER_ID
    scanner_name: str = "Technical-Fundamental Composite Scoring"
    category: str = "quantitative"
    calculation_version: str
    as_of: str  # request date, UTC
    parameters: EffectiveParameters
    universe: str | None = None
    results: list[StockScoreResult]
    summary: ScanSummary
    notes: list[str] = []
