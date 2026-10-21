"""
Unit tests for the two fundamental-valuation QUANTITATIVE strategies
(services.strategies Strategies 27-28) and their wiring into the existing
GET /api/etfs/{symbol}/strategies route:

- compute_graham_valuation  (`graham_valuation`)
- compute_magic_formula_metrics / rank_magic_formula / screen_magic_formula
  (`greenblatt_magic_formula`)
- app.scanners.data valuation-input extraction (partial provider data)
- route integration (no new endpoint, ETF + stock behaviour, regression)

No live network / database / Redis access anywhere in this file.
"""

import asyncio
import datetime
import json
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.scanners import data as scanner_data
from app.services import strategies as st

TODAY = datetime.date(2026, 10, 21)


# ---------------------------------------------------------------------------
# Graham valuation:  EPS x (8.5 + 2g) x (4.4 / Y)
# ---------------------------------------------------------------------------


def _graham(**overrides):
    kwargs = dict(
        eps=5.0,
        eps_period="TTM",
        growth_percent=8.0,
        growth_basis=st.GRAHAM_GROWTH_BASIS_FORWARD,
        aaa_yield_percent=5.1,
        price=80.0,
    )
    kwargs.update(overrides)
    return st.compute_graham_valuation(**kwargs)


class TestGrahamValuation:
    def test_hand_verified_estimate(self):
        # 5 x (8.5 + 2*8) x (4.4 / 5.1) = 5 x 24.5 x 0.862745... = 105.6863
        r = _graham()
        assert r["status"] == "available"
        assert r["estimated_value"] == pytest.approx(105.69, abs=0.005)

    def test_percent_units_not_fractions(self):
        # 8 means 8% (multiplier 24.5). Passing 0.08 would give 8.66 - a
        # wildly different (wrong) answer, proving units are percent.
        assert _graham(growth_percent=8.0)["estimated_value"] == pytest.approx(
            105.69, abs=0.01
        )
        assert _graham(growth_percent=0.08)["estimated_value"] == pytest.approx(
            5 * (8.5 + 0.16) * 4.4 / 5.1, abs=0.01
        )
        # Yield 5.1 (percent) vs 0.051 (a fraction) differ 100x.
        assert _graham(aaa_yield_percent=0.051)["estimated_value"] == pytest.approx(
            105.6863 * 100, rel=1e-4
        )

    def test_yield_equal_to_baseline_removes_rate_adjustment(self):
        r = _graham(aaa_yield_percent=4.4)
        assert r["estimated_value"] == pytest.approx(5 * 24.5, abs=0.005)

    def test_percentage_difference_definition(self):
        r = _graham()  # value 105.6863, price 80
        assert r["difference_abs"] == pytest.approx(25.69, abs=0.01)
        assert r["upside_downside_percent"] == pytest.approx(
            (105.6863 - 80) / 80 * 100, abs=0.01
        )
        below = _graham(price=200.0)  # estimate below market -> negative
        assert below["upside_downside_percent"] < 0
        assert below["difference_abs"] < 0

    def test_sensitivity_scenarios_hand_verified(self):
        s = {x["scenario"]: x for x in _graham()["sensitivity"]}
        assert s["No growth (g = 0)"]["estimated_value"] == pytest.approx(
            36.67, abs=0.01
        )
        assert s["Half of applied growth"]["growth_percent"] == 4.0
        assert s["Half of applied growth"]["estimated_value"] == pytest.approx(
            71.18, abs=0.01
        )
        assert s["75% of applied growth"]["estimated_value"] == pytest.approx(
            88.43, abs=0.01
        )
        # every scenario is at or below the base estimate (conservative)
        base = _graham()["estimated_value"]
        assert all(x["estimated_value"] <= base for x in s.values())

    def test_sensitivity_never_improves_negative_growth(self):
        for x in _graham(growth_percent=-2.0)["sensitivity"]:
            assert x["growth_percent"] <= -2.0

    def test_margin_of_safety(self):
        r = _graham(margin_of_safety_percent=30.0)
        assert r["margin_of_safety_price"] == pytest.approx(105.6863 * 0.7, abs=0.01)
        assert r["price_within_margin_of_safety"] is False  # 80 > 73.98
        assert _graham(price=70.0)["price_within_margin_of_safety"] is True
        assert _graham(margin_of_safety_percent=0.0)[
            "margin_of_safety_price"
        ] == pytest.approx(105.69, abs=0.01)

    def test_invalid_margin_of_safety_is_omitted_with_warning(self):
        r = _graham(margin_of_safety_percent=150.0)
        assert r["status"] == "available"
        assert r["margin_of_safety_price"] is None
        assert any("Margin of safety" in w for w in r["data_quality_warnings"])

    @pytest.mark.parametrize("field", ["eps", "growth_percent", "aaa_yield_percent"])
    def test_missing_input_is_unavailable_not_defaulted(self, field):
        r = _graham(**{field: None})
        assert r["status"] == "unavailable"
        assert r["estimated_value"] is None
        assert r["unavailable_reasons"]

    @pytest.mark.parametrize("bad_yield", [0.0, -1.0])
    def test_zero_or_negative_yield_is_unavailable_without_dividing(self, bad_yield):
        r = _graham(aaa_yield_percent=bad_yield)
        assert r["status"] == "unavailable"
        assert r["estimated_value"] is None

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), "abc", True]
    )
    def test_non_finite_or_non_numeric_inputs_are_unavailable(self, bad):
        for field in ("eps", "growth_percent", "aaa_yield_percent"):
            r = _graham(**{field: bad})
            assert r["status"] == "unavailable", (field, bad)
            assert r["estimated_value"] is None

    @pytest.mark.parametrize("eps", [0.0, -1.5])
    def test_non_positive_eps_is_unavailable(self, eps):
        assert _graham(eps=eps)["status"] == "unavailable"

    def test_growth_so_negative_multiplier_not_positive(self):
        assert _graham(growth_percent=-10.0)["status"] == "unavailable"

    def test_growth_is_capped_and_raw_value_kept(self):
        r = _graham(growth_percent=40.0)
        assert r["growth_percent_raw"] == 40.0
        assert r["growth_percent_applied"] == st.GRAHAM_MAX_GROWTH_PERCENT
        assert any("cap" in w for w in r["data_quality_warnings"])
        assert r["estimated_value"] == pytest.approx(
            5 * (8.5 + 2 * st.GRAHAM_MAX_GROWTH_PERCENT) * 4.4 / 5.1, abs=0.01
        )

    def test_missing_price_keeps_estimate_but_drops_comparison(self):
        for price in (None, 0.0, -3.0, float("nan")):
            r = _graham(price=price)
            assert r["status"] == "available"
            assert r["estimated_value"] == pytest.approx(105.69, abs=0.01)
            assert r["market_price"] is None
            assert r["upside_downside_percent"] is None
            assert r["difference_abs"] is None
            assert r["price_within_margin_of_safety"] is None
            assert any("price" in w.lower() for w in r["data_quality_warnings"])

    def test_historical_proxy_growth_is_flagged(self):
        r = _graham(growth_basis=st.GRAHAM_GROWTH_BASIS_HISTORICAL)
        assert r["growth_basis"] == st.GRAHAM_GROWTH_BASIS_HISTORICAL
        assert any("HISTORICAL" in w for w in r["data_quality_warnings"])

    def test_provenance_and_disclaimer_present(self):
        r = _graham(
            eps_basis="TTM EPS",
            growth_source="consensus",
            growth_horizon="next 5 years",
            aaa_yield_source="FRED AAA",
            aaa_yield_as_of="2026-10-20",
            price_as_of="2026-10-20 (end-of-day close)",
        )
        assert (r["eps_period"], r["eps_basis"]) == ("TTM", "TTM EPS")
        assert (
            r["growth_source"] == "consensus" and r["growth_horizon"] == "next 5 years"
        )
        assert (r["aaa_yield_source"], r["aaa_yield_as_of"]) == (
            "FRED AAA",
            "2026-10-20",
        )
        assert r["market_price_as_of"] == "2026-10-20 (end-of-day close)"
        assert "not a guaranteed intrinsic value" in r["disclaimer"]
        assert r["formula"] == "EPS x (8.5 + 2g) x (4.4 / Y)"

    def test_result_is_json_serialisable(self):
        json.dumps(_graham())
        json.dumps(_graham(eps=None))


# ---------------------------------------------------------------------------
# Greenblatt Magic Formula
# ---------------------------------------------------------------------------


def _cand(symbol, ebit, ev, ca, cash, cl, debt, ppe, sector="Technology", **extra):
    base = dict(
        symbol=symbol,
        company_name=f"{symbol} Inc",
        sector=sector,
        industry="X",
        ebit=ebit,
        ebit_period="2026-06-30",
        ebit_label="EBIT",
        reported_enterprise_value=ev,
        current_assets=ca,
        cash=cash,
        current_liabilities=cl,
        current_debt=debt,
        net_ppe=ppe,
        balance_sheet_period="2026-06-30",
        financial_currency="USD",
        price_currency="USD",
    )
    base.update(extra)
    return base


# Hand-computed:  A: EY 100/1000 = 10%,  NWC (500-100)-(300-50) = 150, IC 450, ROC 22.2222%
#                 B: EY 200/4000 = 5%,   NWC (300-50)-(200-0)   = 50,  IC 100, ROC 200%
#                 C: EY 50/400   = 12.5%, NWC (100-10)-(60-10)   = 40,  IC 100, ROC 50%
A = _cand("AAA", 100, 1000, 500, 100, 300, 50, 300)
B = _cand("BBB", 200, 4000, 300, 50, 200, 0, 50)
C = _cand("CCC", 50, 400, 100, 10, 60, 10, 60)


class TestMagicFormulaMetrics:
    def test_earnings_yield(self):
        assert st.compute_magic_formula_metrics(A, TODAY)[
            "earnings_yield_percent"
        ] == pytest.approx(10.0)
        assert st.compute_magic_formula_metrics(C, TODAY)[
            "earnings_yield_percent"
        ] == pytest.approx(12.5)

    def test_return_on_capital_and_components(self):
        r = st.compute_magic_formula_metrics(A, TODAY)
        assert r["net_working_capital"] == 150
        assert r["net_fixed_assets"] == 300
        assert r["invested_capital"] == 450
        assert r["return_on_capital_percent"] == pytest.approx(
            100 / 450 * 100, abs=1e-4
        )
        assert st.compute_magic_formula_metrics(B, TODAY)[
            "return_on_capital_percent"
        ] == pytest.approx(200.0)

    def test_ev_provenance_reported_vs_computed(self):
        r = st.compute_magic_formula_metrics(A, TODAY)
        assert "provider-reported" in r["enterprise_value_source"]
        comp = dict(
            A,
            reported_enterprise_value=None,
            market_cap=900,
            total_debt=300,
            total_cash=200,
        )
        r2 = st.compute_magic_formula_metrics(comp, TODAY)
        assert r2["enterprise_value"] == 1000
        assert "Computed" in r2["enterprise_value_source"]
        assert r2["earnings_yield_percent"] == pytest.approx(10.0)

    def test_provenance_fields(self):
        r = st.compute_magic_formula_metrics(A, TODAY)
        assert r["ebit_period"] == "2026-06-30" and r["ebit_basis"] == "EBIT"
        assert r["balance_sheet_period"] == "2026-06-30"

    @pytest.mark.parametrize("ev", [0, -500])
    def test_zero_or_negative_ev_is_not_valid_not_attractive(self, ev):
        r = st.compute_magic_formula_metrics(
            dict(A, reported_enterprise_value=ev), TODAY
        )
        assert r["earnings_yield_status"] == "unavailable"
        assert r["earnings_yield_percent"] is None
        assert any("Enterprise value" in w for w in r["data_quality_warnings"])

    def test_missing_ebit(self):
        r = st.compute_magic_formula_metrics(dict(A, ebit=None), TODAY)
        assert (
            r["earnings_yield_status"] == r["return_on_capital_status"] == "unavailable"
        )

    @pytest.mark.parametrize(
        "field",
        ["current_assets", "cash", "current_liabilities", "current_debt", "net_ppe"],
    )
    def test_missing_balance_sheet_component_makes_roc_unavailable(self, field):
        r = st.compute_magic_formula_metrics(dict(A, **{field: None}), TODAY)
        assert r["return_on_capital_status"] == "unavailable"
        assert r["return_on_capital_percent"] is None
        assert r["earnings_yield_status"] == "available"  # independent factor

    def test_invalid_return_on_capital_denominator(self):
        # NWC = (100-50)-(400-0) = -350, NFA 100 -> IC = -250 <= 0
        r = st.compute_magic_formula_metrics(
            dict(
                A,
                current_assets=100,
                cash=50,
                current_liabilities=400,
                current_debt=0,
                net_ppe=100,
            ),
            TODAY,
        )
        assert r["invested_capital"] == -250
        assert r["return_on_capital_status"] == "unavailable"
        # zero denominator: NWC 0, NFA 0 -> must not divide by zero
        z = st.compute_magic_formula_metrics(
            dict(
                A,
                current_assets=100,
                cash=100,
                current_liabilities=50,
                current_debt=50,
                net_ppe=0,
            ),
            TODAY,
        )
        assert z["invested_capital"] == 0
        assert z["return_on_capital_percent"] is None

    def test_negative_net_ppe_is_invalid(self):
        r = st.compute_magic_formula_metrics(dict(A, net_ppe=-10), TODAY)
        assert r["return_on_capital_status"] == "unavailable"

    def test_negative_ebit_is_valid_but_unfavourable(self):
        r = st.compute_magic_formula_metrics(dict(A, ebit=-50), TODAY)
        assert (
            r["earnings_yield_status"] == r["return_on_capital_status"] == "available"
        )
        assert r["earnings_yield_percent"] == pytest.approx(-5.0)
        assert r["return_on_capital_percent"] < 0

    def test_non_finite_inputs_are_missing(self):
        r = st.compute_magic_formula_metrics(
            dict(A, ebit=float("nan"), net_ppe=float("inf")), TODAY
        )
        assert (
            r["earnings_yield_status"] == r["return_on_capital_status"] == "unavailable"
        )

    def test_stale_financials_warn(self):
        r = st.compute_magic_formula_metrics(dict(A, ebit_period="2024-01-31"), TODAY)
        assert any("stale" in w for w in r["data_quality_warnings"])

    def test_currency_mismatch_flagged(self):
        r = st.compute_magic_formula_metrics(dict(A, financial_currency="EUR"), TODAY)
        assert r["currency_mismatch"] is True


class TestMagicFormulaRanking:
    def _screen(self, cands):
        return st.screen_magic_formula(cands, "sp500", TODAY)

    def test_independent_factor_ranks_and_combined_rank(self):
        out = self._screen([A, B, C])
        by = {r["symbol"]: r for r in out["ranked"]}
        assert (
            by["CCC"]["earnings_yield_rank"],
            by["AAA"]["earnings_yield_rank"],
            by["BBB"]["earnings_yield_rank"],
        ) == (1, 2, 3)
        assert (
            by["BBB"]["return_on_capital_rank"],
            by["CCC"]["return_on_capital_rank"],
            by["AAA"]["return_on_capital_rank"],
        ) == (1, 2, 3)
        assert (
            by["CCC"]["combined_rank"],
            by["BBB"]["combined_rank"],
            by["AAA"]["combined_rank"],
        ) == (3, 4, 5)

    def test_sorted_by_combined_rank_ascending(self):
        out = self._screen([A, B, C])
        assert [r["symbol"] for r in out["ranked"]] == ["CCC", "BBB", "AAA"]
        assert [r["overall_position"] for r in out["ranked"]] == [1, 2, 3]
        combined = [r["combined_rank"] for r in out["ranked"]]
        assert combined == sorted(combined)

    def test_input_order_does_not_change_result(self):
        a = self._screen([A, B, C])["ranked"]
        b = self._screen([C, A, B])["ranked"]
        assert [(r["symbol"], r["combined_rank"]) for r in a] == [
            (r["symbol"], r["combined_rank"]) for r in b
        ]

    def test_deterministic_tie_handling(self):
        # ZZZ and AAA2 have identical metrics: they share ranks and are
        # ordered by symbol, regardless of input order.
        t1 = dict(A, symbol="ZZZ")
        t2 = dict(A, symbol="AAA2")
        for order in ([t1, t2, B, C], [C, B, t2, t1]):
            ranked = self._screen(order)["ranked"]
            tied = [r for r in ranked if r["symbol"] in ("ZZZ", "AAA2")]
            assert tied[0]["symbol"] == "AAA2" and tied[1]["symbol"] == "ZZZ"
            assert tied[0]["earnings_yield_rank"] == tied[1]["earnings_yield_rank"]
            assert tied[0]["combined_rank"] == tied[1]["combined_rank"]
        # competition ranking: two tied at EY rank 2 -> next gets rank 4
        ranked = {r["symbol"]: r for r in self._screen([t1, t2, B, C])["ranked"]}
        assert ranked["BBB"]["earnings_yield_rank"] == 4

    def test_missing_data_excluded_never_ranked(self):
        broken = dict(B, symbol="BAD", ebit=None)
        out = self._screen([A, C, broken])
        assert out["ranked_count"] == 2
        assert "BAD" not in [r["symbol"] for r in out["ranked"]]
        bad = next(r for r in out["unranked"] if r["symbol"] == "BAD")
        assert bad["ranked"] is False and bad["combined_rank"] is None
        assert bad["exclusion_reason"] == "insufficient_or_invalid_financial_data"
        assert out["excluded_by_reason"] == {
            "insufficient_or_invalid_financial_data": 1
        }

    def test_invalid_ev_and_denominator_not_given_favourable_rank(self):
        neg_ev = dict(A, symbol="NEGEV", reported_enterprise_value=-100)
        no_ic = dict(
            A,
            symbol="NOIC",
            current_assets=100,
            cash=50,
            current_liabilities=400,
            current_debt=0,
            net_ppe=100,
        )
        out = self._screen([A, B, C, neg_ev, no_ic])
        assert {r["symbol"] for r in out["ranked"]} == {"AAA", "BBB", "CCC"}
        assert out["excluded_count"] == 2

    def test_negative_ebit_ranks_last_on_its_factors(self):
        neg = dict(A, symbol="NEG", ebit=-50)
        out = self._screen([A, B, C, neg])
        by = {r["symbol"]: r for r in out["ranked"]}
        assert by["NEG"]["earnings_yield_rank"] == 4
        assert by["NEG"]["return_on_capital_rank"] == 4
        assert out["ranked"][-1]["symbol"] == "NEG"

    def test_financials_utilities_and_currency_mismatch_are_flagged(self):
        out = self._screen(
            [
                A,
                B,
                C,
                dict(A, symbol="BANK", sector="Financials"),
                dict(A, symbol="UTIL", sector="Utilities"),
                dict(A, symbol="EURO", financial_currency="EUR"),
            ]
        )
        assert out["excluded_by_reason"] == {
            "currency_mismatch": 1,
            "excluded_sector": 2,
        }
        assert out["ranked_count"] == 3

    def test_member_without_fundamentals_is_flagged(self):
        out = self._screen([A, B, {"symbol": "NONE", "no_data": True}])
        assert out["excluded_by_reason"] == {"no_fundamentals_data": 1}
        assert out["ranked_count"] == 2

    def test_one_failing_stock_does_not_crash_the_ranking(self, monkeypatch):
        real = st.compute_magic_formula_metrics

        def flaky(inp, today=None):
            if inp["symbol"] == "BOOM":
                raise RuntimeError("boom")
            return real(inp, today)

        monkeypatch.setattr(st, "compute_magic_formula_metrics", flaky)
        out = self._screen([A, B, dict(C, symbol="BOOM")])
        assert out["ranked_count"] == 2
        assert out["excluded_by_reason"] == {"calculation_error": 1}

    def test_empty_or_all_invalid_universe_is_unavailable(self):
        assert self._screen([])["status"] == "unavailable"
        assert self._screen([dict(A, ebit=None)])["status"] == "unavailable"

    def test_screen_is_deterministic_and_json_serialisable(self):
        a = self._screen([A, B, C])
        b = self._screen([A, B, C])
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ---------------------------------------------------------------------------
# Valuation-input extraction (app.scanners.data) incl. partial provider data
# ---------------------------------------------------------------------------


def _income_df():
    cols = [pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")]
    return pd.DataFrame(
        {cols[0]: [120.0, 130.0], cols[1]: [100.0, 110.0]},
        index=["EBIT", "Operating Income"],
    )


def _balance_df():
    cols = [pd.Timestamp("2025-09-30"), pd.Timestamp("2024-09-30")]
    return pd.DataFrame(
        {
            cols[0]: [500.0, 100.0, 300.0, np.nan, 300.0],
            cols[1]: [1.0, 1.0, 1.0, 1.0, 1.0],
        },
        index=[
            "Current Assets",
            "Cash Cash Equivalents And Short Term Investments",
            "Current Liabilities",
            "Current Debt And Capital Lease Obligation",
            "Net PPE",
        ],
    )


class TestValuationExtraction:
    def test_ebit_latest_period_and_label(self):
        e = scanner_data.extract_ebit(_income_df())
        assert e == {"value": 120.0, "period": "2025-09-30", "label": "EBIT"}

    def test_ebit_falls_back_to_operating_income_labelled(self):
        df = _income_df().drop(index="EBIT")
        e = scanner_data.extract_ebit(df)
        assert e["value"] == 130.0 and e["label"] == "Operating Income"

    def test_balance_sheet_uses_single_period_and_never_backfills(self):
        bs = scanner_data.extract_balance_sheet_components(_balance_df())
        assert bs["period"] == "2025-09-30"
        assert bs["current_assets"] == 500.0 and bs["net_ppe"] == 300.0
        # latest column's Current Debt is NaN -> None (NOT the older column's 1.0, NOT 0)
        assert bs["current_debt"] is None

    def test_forward_growth_percent_and_horizon(self):
        df = pd.DataFrame({"stockTrend": [0.12, 0.10]}, index=["+5y", "0y"])
        g = scanner_data.extract_forward_growth(df)
        assert g["percent"] == pytest.approx(12.0)
        assert "5 years" in g["horizon"] and "analyst" in g["source"]

    def test_forward_growth_missing_or_nan(self):
        assert scanner_data.extract_forward_growth(None) is None
        assert (
            scanner_data.extract_forward_growth(
                pd.DataFrame({"stockTrend": [0.1]}, index=["0y"])
            )
            is None
        )
        assert (
            scanner_data.extract_forward_growth(
                pd.DataFrame({"stockTrend": [np.nan]}, index=["+5y"])
            )
            is None
        )

    def test_historical_cagr_proxy(self):
        hist = [
            {"year": 2022, "eps": 1.0},
            {"year": 2023, "eps": 1.21},
            {"year": 2025, "eps": 1.4641},
        ]
        r = scanner_data.historical_eps_cagr_percent(hist)
        assert r["percent"] == pytest.approx(
            ((1.4641 / 1.0) ** (1 / 3) - 1) * 100, abs=1e-6
        )
        assert scanner_data.historical_eps_cagr_percent(hist[:2]) is None
        assert (
            scanner_data.historical_eps_cagr_percent(
                [
                    {"year": 2020, "eps": -1.0},
                    {"year": 2021, "eps": 1.0},
                    {"year": 2022, "eps": 2.0},
                ]
            )
            is None
        )

    def test_extract_inputs_with_all_providers_failing(self):
        out = scanner_data.extract_valuation_inputs(None, None, None, None)
        assert (
            out["ebit"] is None
            and out["balance_sheet"] is None
            and out["forward_growth"] is None
        )
        assert out["market_cap"] is None and out["trailing_eps"] is None
        json.dumps(out)

    def test_extract_inputs_cleans_nan_info_fields(self):
        out = scanner_data.extract_valuation_inputs(
            {
                "marketCap": float("nan"),
                "enterpriseValue": 1000,
                "trailingEps": "x",
                "currency": "USD",
            },
            None,
            None,
            None,
        )
        assert out["market_cap"] is None and out["trailing_eps"] is None
        assert out["enterprise_value"] == 1000.0

    def test_candidate_round_trip_into_metrics(self):
        inputs = scanner_data.extract_valuation_inputs(
            {"enterpriseValue": 1000.0, "currency": "USD", "financialCurrency": "USD"},
            _income_df(),
            _balance_df(),
            None,
        )
        cand = scanner_data.build_magic_formula_candidate(
            "ZZZ", "Z", "Tech", "X", inputs
        )
        row = st.compute_magic_formula_metrics(cand, datetime.date(2026, 1, 1))
        assert row["earnings_yield_percent"] == pytest.approx(12.0)
        # current debt missing -> ROC unavailable, flagged, EY still valid
        assert row["return_on_capital_status"] == "unavailable"

    def test_candidate_without_inputs_is_no_data(self):
        assert (
            scanner_data.build_magic_formula_candidate("A", None, None, None, None)[
                "no_data"
            ]
            is True
        )

    def test_partial_provider_failure_keeps_canslim_fields_intact(self, monkeypatch):
        """balance sheet + growth estimate fetches fail (None); the CANSLIM
        part of the extraction is unaffected and valuation inputs are partial."""

        async def ret(value):
            return value

        yc = scanner_data.yfinance_client
        monkeypatch.setattr(
            yc,
            "fetch_info",
            lambda s: ret(
                {"longName": "Z Corp", "sector": "Tech", "enterpriseValue": 5.0}
            ),
        )
        monkeypatch.setattr(yc, "fetch_earnings_dates", lambda s: ret(None))
        monkeypatch.setattr(yc, "fetch_annual_income_stmt", lambda s: ret(_income_df()))
        monkeypatch.setattr(yc, "fetch_institutional_holders", lambda s: ret(None))
        monkeypatch.setattr(yc, "fetch_balance_sheet", lambda s: ret(None))
        monkeypatch.setattr(yc, "fetch_growth_estimates", lambda s: ret(None))

        out = asyncio.run(scanner_data.fetch_and_extract_fundamentals_for_symbol("ZZZ"))
        assert out["company_name"] == "Z Corp" and out["sector"] == "Tech"
        assert "annual_eps_history" in out and "quarterly_eps_current" in out
        v = out["valuation_inputs"]
        assert v["ebit"]["value"] == 120.0
        assert v["balance_sheet"] is None and v["forward_growth"] is None


# ---------------------------------------------------------------------------
# Route integration: existing GET /api/etfs/{symbol}/strategies, no new endpoint
# ---------------------------------------------------------------------------

# Every (method, path) the API exposed BEFORE this change (19 operations).
ORIGINAL_ROUTES = {
    ("GET", "/api/compare"),
    ("GET", "/api/dashboard"),
    ("GET", "/api/dashboard/market"),
    ("GET", "/api/etfs"),
    ("GET", "/api/etfs/allocation"),
    ("GET", "/api/etfs/compare"),
    ("GET", "/api/etfs/{symbol}"),
    ("GET", "/api/etfs/{symbol}/performance"),
    ("GET", "/api/etfs/{symbol}/quote"),
    ("GET", "/api/etfs/{symbol}/strategies"),
    ("GET", "/api/scanners/canslim"),
    ("GET", "/api/scanners/canslim/{symbol}"),
    ("GET", "/api/scanners/universes"),
    ("GET", "/api/strategies/canslim"),
    ("GET", "/health/live"),
    ("GET", "/health/ready"),
    ("POST", "/api/admin/refresh"),
    ("POST", "/api/admin/scanners/refresh"),
    ("POST", "/api/portfolio/optimization"),
}


@pytest.fixture
def api(monkeypatch):
    from app.database import get_db
    from app.main import app
    from app.routers import strategy as router
    from app.services import cache

    async def _no_cache_get(key):
        return None

    async def _noop_set(key, value, ttl=None):
        return None

    async def _passthrough(key, fn, ttl=None):
        return await fn()

    monkeypatch.setattr(cache, "get_json", _no_cache_get)
    monkeypatch.setattr(cache, "set_json", _noop_set)
    monkeypatch.setattr(cache, "get_or_compute", _passthrough)

    async def _db():
        yield None

    app.dependency_overrides[get_db] = _db
    yield SimpleNamespace(client=TestClient(app), router=router, app=app, cache=cache)
    app.dependency_overrides.clear()


def _members():
    mk = lambda sym, sector="Technology": SimpleNamespace(
        symbol=sym, name=f"{sym} Inc", sector=sector, industry="X", universe="sp500"
    )
    return [mk("AAA"), mk("BBB"), mk("CCC"), mk("BANK", "Financials"), mk("NEW")]


def _snapshot(sym, cand, sector="Technology", **inputs_over):
    inputs = {
        "schema_version": 1,
        "extracted_at": "2026-10-20T10:00:00+00:00",
        "price_currency": "USD",
        "financial_currency": "USD",
        "market_cap": None,
        "enterprise_value": cand["reported_enterprise_value"],
        "total_debt": None,
        "total_cash": None,
        "trailing_eps": 5.0,
        "ebit": {"value": cand["ebit"], "period": "2026-06-30", "label": "EBIT"},
        "balance_sheet": {
            "period": "2026-06-30",
            "current_assets": cand["current_assets"],
            "cash": cand["cash"],
            "current_liabilities": cand["current_liabilities"],
            "current_debt": cand["current_debt"],
            "net_ppe": cand["net_ppe"],
        },
        "forward_growth": {
            "percent": 8.0,
            "horizon": "next 5 years (per annum)",
            "source": "consensus",
        },
    }
    inputs.update(inputs_over)
    return SimpleNamespace(
        symbol=sym,
        company_name=f"{sym} Inc",
        sector=sector,
        industry="X",
        valuation_inputs_json=json.dumps(inputs),
        annual_eps_history_json=json.dumps(
            [
                {"year": 2023, "eps": 3.0},
                {"year": 2024, "eps": 4.0},
                {"year": 2025, "eps": 4.5},
            ]
        ),
    )


@pytest.fixture
def seeded(api, monkeypatch):
    snaps = {
        "AAA": _snapshot("AAA", A),
        "BBB": _snapshot("BBB", B),
        "CCC": _snapshot("CCC", C),
        "BANK": _snapshot("BANK", A, sector="Financials"),
    }

    async def get_universe(db, key):
        return _members()

    async def load_snaps(db, symbols):
        return {s: snaps[s] for s in symbols if s in snaps}

    async def find_member(db, symbol):
        return next((m for m in _members() if m.symbol == symbol.upper()), None)

    async def load_prices(db, symbol, limit=None):
        return [SimpleNamespace(close=80.0, date=datetime.date(2026, 10, 20))]

    r = api.router
    monkeypatch.setattr(r.scanner_universe, "get_stock_universe", get_universe)
    monkeypatch.setattr(r.scanner_data, "load_fundamentals_snapshots", load_snaps)
    monkeypatch.setattr(r, "_find_stock_member", find_member)
    monkeypatch.setattr(r, "load_prices", load_prices)
    monkeypatch.setattr(r.settings, "graham_aaa_yield_percent", 5.1)
    monkeypatch.setattr(r.settings, "graham_aaa_yield_as_of", "2026-10-20")
    return api


class TestRouteIntegration:
    def test_no_new_endpoints(self, api):
        paths = {
            (m.upper(), p) for p, v in api.app.openapi()["paths"].items() for m in v
        }
        assert paths == ORIGINAL_ROUTES

    def test_unknown_symbol_still_404(self, seeded, monkeypatch):
        async def none(db, symbol):
            return None

        monkeypatch.setattr(seeded.router, "_find_stock_member", none)
        resp = seeded.client.get("/api/etfs/NOPE/strategies")
        assert resp.status_code == 404
        # same error envelope/message as before this change
        assert resp.json() == {
            "data": None,
            "success": False,
            "message": "Unknown ETF symbol: NOPE",
        }

    def test_stock_symbol_returns_both_strategies(self, seeded):
        resp = seeded.client.get("/api/etfs/CCC/strategies")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        g = data["grahamValuation"]
        assert g["status"] == "available" and g["symbol"] == "CCC"
        # trailing EPS 5, g 8 (forward), Y 5.1 -> 105.69; price 80
        assert g["estimatedValue"] == pytest.approx(105.69, abs=0.01)
        assert g["growthBasis"] == "forward_analyst_consensus"
        assert g["aaaYieldAsOf"] == "2026-10-20" and g["marketPriceAsOf"].startswith(
            "2026-10-20"
        )
        mf = data["greenblattMagicFormula"]
        assert mf["status"] == "available"
        assert [r["symbol"] for r in mf["topRanked"]] == ["CCC", "BBB", "AAA"]
        assert mf["requestedSymbolResult"]["symbol"] == "CCC"
        assert mf["requestedSymbolResult"]["combinedRank"] == 3
        assert mf["universeSize"] == 5 and mf["rankedCount"] == 3
        assert mf["excludedByReason"] == {
            "excluded_sector": 1,
            "no_fundamentals_data": 1,
        }
        assert any(
            "no ingested valuation inputs" in w for w in mf["dataQualityWarnings"]
        )
        # category grouping: Quantitative only, no new top-level category
        assert set(data["quantitativeStrategies"]) >= {
            "grahamValuation",
            "greenblattMagicFormula",
        }
        assert "message" in resp.json() and "stock" in resp.json()["message"]

    def test_excluded_stock_gets_flagged_row_not_rank(self, seeded):
        data = seeded.client.get("/api/etfs/BANK/strategies").json()["data"]
        row = data["greenblattMagicFormula"]["requestedSymbolResult"]
        assert row["ranked"] is False and row["combinedRank"] is None
        assert row["exclusionReason"] == "excluded_sector"

    def test_graham_unavailable_without_configured_yield(self, seeded, monkeypatch):
        monkeypatch.setattr(seeded.router.settings, "graham_aaa_yield_percent", None)
        g = seeded.client.get("/api/etfs/AAA/strategies").json()["data"][
            "grahamValuation"
        ]
        assert g["status"] == "unavailable" and g["estimatedValue"] is None
        assert any("GRAHAM_AAA_YIELD_PERCENT" in r for r in g["unavailableReasons"])

    def test_graham_uses_labelled_historical_proxy_when_no_forward_growth(
        self, seeded, monkeypatch
    ):
        snap = _snapshot("AAA", A, forward_growth=None)

        async def load_snaps(db, symbols):
            return {"AAA": snap}

        monkeypatch.setattr(
            seeded.router.scanner_data, "load_fundamentals_snapshots", load_snaps
        )
        g = seeded.client.get("/api/etfs/AAA/strategies").json()["data"][
            "grahamValuation"
        ]
        assert g["status"] == "available"
        assert g["growthBasis"] == "historical_eps_cagr_proxy"
        assert any("HISTORICAL" in w for w in g["dataQualityWarnings"])

    def test_graham_unavailable_when_no_snapshot(self, seeded, monkeypatch):
        async def load_snaps(db, symbols):
            return {}

        monkeypatch.setattr(
            seeded.router.scanner_data, "load_fundamentals_snapshots", load_snaps
        )
        g = seeded.client.get("/api/etfs/NEW/strategies").json()["data"][
            "grahamValuation"
        ]
        assert g["status"] == "unavailable"

    def test_etf_symbol_keeps_existing_payload_and_adds_valuation(
        self, seeded, monkeypatch
    ):
        # Minimal stand-in for the cached ETF payload (all strategies null,
        # which the route already treats as "insufficient history").
        cached_etf = {
            "ema50_rsi": None,
            "macd": None,
            "evar_risk": None,
            "quantitative_strategies": {"evar_risk": None, "momentum_reversal": None},
            "technical_strategies": {"macd": None},
        }

        async def get_json(key):
            assert key == "tw:v10:strategies:QQQ"  # unchanged ETF cache key
            return cached_etf

        monkeypatch.setattr(seeded.cache, "get_json", get_json)
        resp = seeded.client.get("/api/etfs/QQQ/strategies")
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert (
            body["macd"] is None and body["evarRisk"] is None
        )  # existing flat fields untouched
        assert set(body["quantitativeStrategies"]) >= {"evarRisk", "momentumReversal"}
        assert (
            "Insufficient price history" in resp.json()["message"]
        )  # existing message logic intact
        assert body["grahamValuation"]["status"] == "unavailable"
        assert "not ETFs" in body["grahamValuation"]["unavailableReasons"][0]
        assert body["greenblattMagicFormula"]["status"] == "available"
        assert body["greenblattMagicFormula"]["requestedSymbolResult"] is None
        # the shared cached dict was not mutated (valuation is attached to a copy)
        assert "graham_valuation" not in cached_etf
        assert "graham_valuation" not in cached_etf["quantitative_strategies"]

    def test_magic_formula_unavailable_when_universe_not_ingested(
        self, api, monkeypatch
    ):
        async def empty(db, key):
            return []

        monkeypatch.setattr(api.router.scanner_universe, "get_stock_universe", empty)
        view = asyncio.run(api.router._load_magic_formula_view(None, "sp500", None))
        assert view["status"] == "unavailable"
        assert "not been ingested" in view["unavailable_reasons"][0]

    def test_magic_formula_unavailable_when_no_valuation_inputs_ingested(
        self, api, monkeypatch
    ):
        async def members(db, key):
            return _members()

        async def none(db, symbols):
            return {}

        monkeypatch.setattr(api.router.scanner_universe, "get_stock_universe", members)
        monkeypatch.setattr(
            api.router.scanner_data, "load_fundamentals_snapshots", none
        )
        view = asyncio.run(api.router._load_magic_formula_view(None, "sp500", None))
        assert view["status"] == "unavailable"
        assert "refresh" in view["unavailable_reasons"][0]

    def test_unexpected_failure_becomes_unavailable_not_500(self, api, monkeypatch):
        async def boom(db, key):
            raise RuntimeError("db down")

        monkeypatch.setattr(api.router.scanner_universe, "get_stock_universe", boom)
        view = asyncio.run(api.router._load_magic_formula_view(None, "sp500", None))
        assert view["status"] == "unavailable"
        assert "failed" in view["unavailable_reasons"][0]


# ---------------------------------------------------------------------------
# Regression: existing identifiers / categories / response structure
# ---------------------------------------------------------------------------


class TestRegression:
    def test_existing_category_fields_unchanged(self):
        from app.schemas import (
            CalendarStrategies,
            PortfolioStrategies,
            QuantitativeStrategies,
            RegimeStrategies,
            TechnicalStrategies,
        )

        assert list(TechnicalStrategies.model_fields) == [
            "ema50_rsi",
            "ema821",
            "macd",
            "bollinger_bands",
            "better_breakout",
            "sma_trend",
            "triple_ma_pullback",
            "mswing",
            "td_setup",
            "supertrend_confirmation",
            "ufx_vervoort_heiken_ashi",
            "trend_signals_trend_tide",
            "pivot_levels",
            "cpr",
            "ta_candles_trend_health",
        ]
        assert list(QuantitativeStrategies.model_fields)[:6] == [
            "evar_risk",
            "momentum_reversal",
            "squeeze_momentum",
            "soc_avalanche",
            "wavelet_trend_filter",
            "first_passage_time",
        ]
        # the two new strategies live in the EXISTING Quantitative category only
        assert list(QuantitativeStrategies.model_fields)[6:] == [
            "graham_valuation",
            "greenblatt_magic_formula",
        ]
        assert list(RegimeStrategies.model_fields) == [
            "risk_on_risk_off",
            "hmm_regime_switching",
            "factor_momentum",
        ]
        assert list(CalendarStrategies.model_fields) == ["tlt_monthly_cycle"]
        assert list(PortfolioStrategies.model_fields) == [
            "tqqq_tmf_ief_rebalancing",
            "vt_vs_vti_vxus",
        ]

    def test_strategy_analytics_only_gained_optional_fields(self):
        from app.schemas import StrategyAnalytics

        fields = StrategyAnalytics.model_fields
        assert "graham_valuation" in fields and "greenblatt_magic_formula" in fields
        assert all(not f.is_required() for f in fields.values())
        for original in (
            "ema50_rsi",
            "ema821",
            "macd",
            "bollinger_bands",
            "better_breakout",
            "sma_trend",
            "vt_vs_vti_vxus",
            "evar_risk",
            "risk_on_risk_off",
            "triple_ma_pullback",
            "tlt_monthly_cycle",
            "tqqq_tmf_ief_rebalancing",
            "mswing",
            "momentum_reversal",
            "factor_momentum",
            "hmm_regime_switching",
            "squeeze_momentum",
            "soc_avalanche",
            "wavelet_trend_filter",
            "first_passage_time",
            "td_setup",
            "supertrend_confirmation",
            "ufx_vervoort_heiken_ashi",
            "trend_signals_trend_tide",
            "pivot_levels",
            "cpr",
            "ta_candles_trend_health",
            "technical_strategies",
            "quantitative_strategies",
            "regime_strategies",
            "calendar_strategies",
            "portfolio_strategies",
        ):
            assert original in fields

    def test_existing_compute_functions_still_present(self):
        for name in (
            "compute_ema50_rsi_pullback",
            "compute_macd",
            "compute_evar_risk",
            "compute_factor_momentum",
            "compute_portfolio_optimization",
            "compute_cpr",
        ):
            assert callable(getattr(st, name))

    def test_portfolio_optimization_and_kelly_still_work(self):
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2023-01-02", periods=400)

        def prices(seed_shift):
            close = 100 * np.cumprod(
                1 + rng.normal(0.0004 + seed_shift, 0.01, len(idx))
            )
            return [
                SimpleNamespace(
                    date=d.date(),
                    open=c,
                    high=c,
                    low=c,
                    close=c,
                    adjusted_close=c,
                    volume=1,
                )
                for d, c in zip(idx, close)
            ]

        result = st.compute_portfolio_optimization(
            {"AAA": prices(0.0), "BBB": prices(0.0002)}, risk_free_rate_percent=1.0
        )
        assert result is not None
        assert {k["symbol"] for k in result["kelly_sizing"]} == {"AAA", "BBB"}
        assert math.isclose(
            sum(w["weight_percent"] for w in result["max_sharpe_portfolio"]["weights"]),
            100.0,
            abs_tol=0.5,
        )

    def test_model_gained_only_the_nullable_column(self):
        from app.models import StockFundamentalsSnapshot

        col = StockFundamentalsSnapshot.__table__.c.valuation_inputs_json
        assert col.nullable is True
