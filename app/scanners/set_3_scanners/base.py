"""Shared scaffolding for Set 3 stock scanners."""

from __future__ import annotations

from dataclasses import dataclass

CATEGORY_QUANTITATIVE = "quantitative"


@dataclass(frozen=True)
class ScannerDescriptor:
    scanner_id: str
    name: str
    category: str
    description: str
    calculation_version: str
