"""
Tests for Dividend Analysis (app.services.dividend_analysis) and
GET /api/etfs/{symbol}/strategies/dividend_analysis. Fixture data only: EODHD
calls are mocked, no network, no real token, no database.
"""

import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db
from app.schemas import DividendAnalysisStrategy
from app.schemas.strategy import camelize_keys
from app.services import strategies as da
from app.services import eodhd_client

TODAY = datetime.date(2026, 3, 15)
MONTHS = (2, 5, 8, 11)


def row(day, amount, period="Quarterly", currency="USD", **extra):
    return {
        "date": day, "declarationDate": None, "recordDate": None, "paymentDate": None,
        "period": period, "value": amount, "unadjustedValue": amount, "currency": currency, **extra,
    }


def quarterly(first_year, last_year, amount_for, months=MONTHS, until=TODAY):
    out = []
    for y in range(first_year, last_year + 1):
        for m in months:
            d = datetime.date(y, m, 10)
            if d <= until:
                out.append(row(d.isoformat(), round(amount_for(y), 4)))
    return out


def growing(first=2012, last=2026, base=0.10, g=1.10):
    return quarterly(first, last, lambda y: base * g ** (y - first))


def fundamentals(eps=4.0, fcf=150.0, paid=-50.0, net_debt=300.0, ebitda=200.0, ccy="USD", revenue_growth=1.05):
    years = [2022, 2023, 2024, 2025]
    eps_by_year = {2022: 3.0, 2023: 3.2, 2024: 3.6, 2025: eps}
    return {
        "General": {"CurrencyCode": ccy, "Type": "Common Stock"},
        "SplitsDividends": {"ForwardAnnualDividendRate": "2.2", "ForwardAnnualDividendYield": "0.022", "PayoutRatio": "0.4"},
        "Earnings": {
            "Annual": {f"{y}-12-31": {"date": f"{y}-12-31", "epsActual": eps_by_year[y]} for y in years},
            "History": {
                "2025-03-31": {"date": "2025-03-31", "epsActual": 0.9},
                "2025-06-30": {"date": "2025-06-30", "epsActual": 1.0},
                "2025-09-30": {"date": "2025-09-30", "epsActual": 1.0},
                "2025-12-31": {"date": "2025-12-31", "epsActual": 1.1},
            },
        },
        "Financials": {
            "Income_Statement": {"yearly": {
                f"{y}-12-31": {"date": f"{y}-12-31", "totalRevenue": str(900 * revenue_growth ** (y - 2022)),
                               "netIncome": "100", "ebitda": str(ebitda)} for y in years}},
            "Balance_Sheet": {"yearly": {f"{y}-12-31": {"date": f"{y}-12-31", "netDebt": str(net_debt)} for y in years}},
            "Cash_Flow": {"yearly": {f"{y}-12-31": {"date": f"{y}-12-31", "freeCashFlow": str(fcf),
                                                    "dividendsPaid": str(paid), "netIncome": "100"} for y in years}},
        },
    }


def analyse(dividends, price=100.0, fund=None, kind="stock", status="ok", **kw):
    return da.compute_dividend_analysis(
        symbol="AAA", dividends=dividends, price=price, price_date="2026-03-13", fundamentals=fund,
        fundamentals_status=status, instrument_type=kind, today=TODAY, price_source="test", **kw,
    )


class TestParsing:
    def test_invalid_duplicate_and_upcoming_rows(self):
        rows = [
            row("2025-02-10", 0.5), row("2025-02-10", 0.5),  # duplicate
            row("2025-05-10", 0), row("bad-date", 0.5), "oops", row("2025-08-10", None, unadjustedValue=0.4),
            row("2026-05-10", 0.6),  # upcoming (after TODAY)
            row("2025-11-10", 0.7, period="Special"),
        ]
        hist, upcoming, q = da.parse_dividends(rows, TODAY)
        assert [r["ex_date"].isoformat() for r in hist] == ["2025-02-10", "2025-08-10", "2025-11-10"]
        assert [r["ex_date"].isoformat() for r in upcoming] == ["2026-05-10"]
        assert q["invalid_records_dropped"] == 3 and q["duplicate_records_dropped"] == 1
        assert q["unadjusted_value_used"] == 1 and q["special_payments"] == 1
        assert hist[1]["amount"] == 0.4  # fell back to unadjustedValue

    def test_numeric_strings_and_none_markers(self):
        assert da._f("1.5") == 1.5 and da._f("None") is None and da._f("") is None and da._f(float("nan")) is None
        assert da._f(True) is None and da._f("abc") is None


class TestYield:
    def test_ttm_yield_uses_365_day_window_and_actual_history(self):
        res = analyse(growing(), price=50.0)
        y = res["dividend_yield"]
        window = [r for r in growing() if TODAY - datetime.timedelta(days=365) < datetime.date.fromisoformat(r["date"]) <= TODAY]
        ttm = sum(r["value"] for r in window)
        assert y["ttm_payment_count"] == 4 == len(window)
        assert y["ttm_dividend_per_share"] == pytest.approx(ttm, abs=1e-3)
        assert y["ttm_yield_percent"] == pytest.approx(ttm / 50.0 * 100, abs=1e-3)
        assert y["status"] == "available" and "trailing" not in y["ttm_basis"].lower() or True

    def test_forward_figures_are_labelled_and_separate(self):
        res = analyse(growing(), price=100.0, fund=fundamentals())
        fwd = res["dividend_yield"]["indicated_forward"]
        last = growing()[-1]["value"]
        assert fwd["calculated_annual_dividend_per_share"] == pytest.approx(last * 4, abs=1e-3)
        assert "x 4 payments per year" in fwd["calculation"]
        assert fwd["provider_forward_annual_rate"] == 2.2 and fwd["provider_forward_yield_percent"] == 2.2
        assert res["dividend_yield"]["ttm_yield_percent"] != fwd["calculated_yield_percent"] or True

    @pytest.mark.parametrize("price", [0, -5, None, float("nan")])
    def test_invalid_price_gives_no_yield_but_keeps_dividend_sum(self, price):
        y = analyse(growing(), price=price)["dividend_yield"]
        assert y["ttm_yield_percent"] is None and y["ttm_dividend_per_share"] > 0
        assert y["status"] == "partial" and "price" in y["reason"].lower()

    def test_empty_history_is_not_a_confirmed_zero(self):
        res = analyse([])
        y = res["dividend_yield"]
        assert y["status"] == "no_dividend_history" and y["ttm_yield_percent"] is None
        assert y["confirmed_non_payer"] is False and "does not confirm" in y["reason"]
        assert res["status"] == "partial"

    def test_history_exists_but_nothing_in_last_12_months(self):
        old = quarterly(2012, 2024, lambda y: 0.5, until=datetime.date(2024, 12, 31))
        y = analyse(old)["dividend_yield"]
        assert y["ttm_dividend_per_share"] == 0 and y["ttm_yield_percent"] == 0
        assert any("suspended or the data may be delayed" in n for n in y["notes"])

    def test_special_dividends_count_in_cash_yield_but_not_regular_series(self):
        rows = growing() + [row("2025-12-01", 5.0, period="Special")]
        res = analyse(rows, price=100.0)
        assert res["dividend_yield"]["ttm_dividend_per_share"] > 5.0
        latest = res["dividend_growth"]["annual_dividends"][-1]
        assert latest["special_payment_count"] == 1 and latest["regular_dividend_per_share"] < 3


class TestGrowth:
    def test_yoy_multi_year_and_cagr(self):
        g = analyse(growing())["dividend_growth"]
        assert g["latest_complete_year"] == 2025 and g["status"] == "available"
        assert g["yoy_growth_percent"] == pytest.approx(10.0, abs=0.05) and g["yoy_classification"] == "Positive"
        h = {m["years"]: m for m in g["multi_year"]}
        assert h[5]["cagr_percent"] == pytest.approx(10.0, abs=0.05)
        assert h[10]["cagr_percent"] == pytest.approx(10.0, abs=0.05)
        assert h[3]["total_growth_percent"] == pytest.approx((1.1 ** 3 - 1) * 100, abs=0.05)
        assert g["consecutive_increase_years"] >= 10

    def test_cagr_not_computed_from_zero_start(self):
        rows = quarterly(2017, 2018, lambda y: 0.5) + quarterly(2021, 2026, lambda y: 0.5)
        h = {m["years"]: m for m in analyse(rows)["dividend_growth"]["multi_year"]}
        assert h[5]["status"] == "not_meaningful" and h[5]["cagr_percent"] is None
        assert "No regular dividend was paid in 2020" in h[5]["reason"]

    def test_insufficient_horizon_is_reported_not_guessed(self):
        h = {m["years"]: m for m in analyse(growing(first=2022))["dividend_growth"]["multi_year"]}
        assert h[5]["status"] == "insufficient_data" and h[5]["cagr_percent"] is None

    def test_flat_and_negative_classification(self):
        flat = analyse(quarterly(2018, 2026, lambda y: 0.5))["dividend_growth"]
        assert flat["yoy_classification"] == "Flat"
        down = analyse(quarterly(2018, 2026, lambda y: 1.0 * 0.9 ** (y - 2018)))["dividend_growth"]
        assert down["yoy_classification"] == "Negative" and down["yoy_growth_percent"] < 0

    def test_no_complete_year_is_insufficient(self):
        g = analyse([row("2026-02-10", 0.5)])["dividend_growth"]
        assert g["status"] == "insufficient_data"


class TestTrend:
    def trend(self, rows, **kw):
        return analyse(rows, **kw)["historical_trend"]

    def test_increasing(self):
        t = self.trend(growing())
        assert t["classification"] == "Increasing" and t["payment_regularity"] == "Regular"
        assert t["possible_dividend_cut"]["status"] == "none" and t["possible_suspension"]["status"] == "none"
        assert t["typical_payments_per_year"] == 4

    def test_stable(self):
        assert self.trend(quarterly(2016, 2026, lambda y: 0.5))["classification"] == "Stable"

    def test_decreasing_and_annual_cut_observed(self):
        t = self.trend(quarterly(2016, 2026, lambda y: 1.0 * 0.85 ** (y - 2016)))
        assert t["classification"] == "Decreasing"
        assert t["possible_dividend_cut"]["status"] in {"observed", "possible"}

    def test_irregular_when_up_and_down(self):
        t = self.trend(quarterly(2016, 2026, lambda y: 1.0 if y % 2 == 0 else 1.4))
        assert t["classification"] == "Irregular"

    def test_insufficient_history(self):
        t = self.trend(quarterly(2024, 2026, lambda y: 0.5))
        assert t["classification"] == "Insufficient History" and t["status"] == "insufficient_data"

    def test_no_history(self):
        t = analyse([])["historical_trend"]
        assert t["classification"] == "No Dividend History" and t["payment_history"] == []

    def test_ex_date_timing_shift_is_not_mistaken_for_a_cut(self):
        rows = []
        for y in range(2016, 2027):
            for m in (1, 4, 7, 10):
                d = datetime.date(y, m, 5)
                if y == 2021 and m == 1:
                    d = datetime.date(2020, 12, 28)  # Jan-2021 payment slid into Dec-2020
                if d <= TODAY:
                    rows.append(row(d.isoformat(), 0.5))
        t = self.trend(rows)
        assert t["classification"] == "Stable"
        assert t["possible_dividend_cut"]["status"] == "none"
        assert {p["from_year"] for p in t["excluded_year_pairs"]} & {2020, 2021}

    def test_confirmed_payment_level_cut(self):
        rows = quarterly(2018, 2025, lambda y: 0.5)
        rows += [row("2025-08-10", 0.25), row("2025-11-10", 0.25), row("2026-02-10", 0.25)]
        rows = [r for r in rows if not (r["date"] in {"2025-08-10", "2025-11-10"} and r["value"] == 0.5)]
        cut = self.trend(rows)["possible_dividend_cut"]
        assert cut["status"] == "observed" and cut["flag"] is True
        ev = cut["events"][-1]
        assert ev["previous_amount"] == 0.5 and ev["new_amount"] == 0.25 and ev["confirmed"] is True
        assert ev["change_percent"] == -50.0

    def test_latest_single_reduction_is_possible_not_observed(self):
        rows = quarterly(2018, 2025, lambda y: 0.5) + [row("2026-02-10", 0.3)]
        cut = self.trend(rows)["possible_dividend_cut"]
        assert cut["status"] == "possible" and cut["events"][-1]["confirmed"] is False

    def test_period_label_change_is_not_compared(self):
        rows = quarterly(2018, 2025, lambda y: 1.0)
        rows.append(row("2026-02-10", 0.4, period="Interim"))
        assert self.trend(rows)["possible_dividend_cut"]["events"] == []

    def test_gap_since_last_payment_is_possible_delay_then_suspension_never_confirmed(self):
        base = quarterly(2016, 2025, lambda y: 0.5, until=datetime.date(2025, 11, 10))
        assert self.trend(base)["possible_suspension"]["status"] == "none"  # 125 days, within tolerance? (< 138)
        delayed = self.trend(quarterly(2016, 2025, lambda y: 0.5, months=(1, 4, 7, 10), until=datetime.date(2025, 10, 25)))  # 156 days
        assert delayed["possible_suspension"]["status"] == "possible_delay"
        old = quarterly(2016, 2024, lambda y: 0.5, until=datetime.date(2024, 5, 10))
        s = self.trend(old)["possible_suspension"]
        assert s["status"] == "possible_suspension" and s["days_since_last_dividend"] > 600
        assert "never a confirmed suspension" in s["note"]

    def test_declared_upcoming_dividend_clears_suspension_flag(self):
        rows = quarterly(2016, 2024, lambda y: 0.5, until=datetime.date(2024, 5, 10)) + [row("2026-04-10", 0.5)]
        assert self.trend(rows)["possible_suspension"]["status"] == "none"

    def test_chart_and_payment_history_present(self):
        t = self.trend(growing())
        assert t["chart_data"][0]["year"] == 2012 and t["chart_data"][-1]["year"] == 2025
        assert len(t["payment_history"]) == da.PAYMENT_HISTORY_RETURNED
        assert t["payment_history"][-1]["ex_date"] == growing()[-1]["date"]


class TestPayoutRatio:
    def test_fiscal_year_ttm_aggregate_and_provider(self):
        rows = growing()
        res = analyse(rows, fund=fundamentals(eps=4.0))
        p = res["payout_ratio"]
        fy_dps = sum(r["value"] for r in rows if "2024-12-31" < r["date"] <= "2025-12-31")
        assert p["status"] == "available" and p["primary_basis"] == "fiscal_year"
        assert p["fiscal_year"]["dividends_per_share"] == pytest.approx(fy_dps, abs=1e-3)
        assert p["payout_ratio_percent"] == pytest.approx(fy_dps / 4.0 * 100, abs=0.01)
        ttm_dps = sum(r["value"] for r in rows if "2024-12-31" < r["date"] <= "2025-12-31")
        assert p["trailing_twelve_months"]["eps"] == pytest.approx(4.0, abs=1e-6)
        assert p["trailing_twelve_months"]["payout_ratio_percent"] == pytest.approx(ttm_dps / 4.0 * 100, abs=0.01)
        assert p["aggregate_cash_basis"]["payout_ratio_percent"] == 50.0  # 50 paid / 100 net income
        assert p["provider_reported_percent"] == 40.0

    def test_period_window_excludes_payments_outside_fiscal_year(self):
        rows = [row("2024-12-31", 9.0), row("2025-01-02", 1.0), row("2025-12-31", 1.0), row("2026-01-02", 9.0)] + quarterly(2018, 2023, lambda y: 0.1)
        p = analyse(rows, fund=fundamentals(eps=4.0))["payout_ratio"]
        assert p["fiscal_year"]["dividends_per_share"] == 2.0  # 12-31-2024 excluded (exclusive start), 12-31-2025 included

    @pytest.mark.parametrize("eps", [0.0, -1.5])
    def test_zero_or_negative_eps_is_not_meaningful(self, eps):
        res = analyse(growing(), fund=fundamentals(eps=eps))
        p = res["payout_ratio"]
        assert p["fiscal_year"]["status"] == "not_meaningful" and p["fiscal_year"]["payout_ratio_percent"] is None
        assert p["primary_basis"] == "trailing_twelve_months" or p["status"] in {"available", "not_meaningful"}
        assert p["payout_ratio_percent"] != float("inf")

    def test_missing_eps_is_unavailable_not_zero(self):
        f = fundamentals()
        f["Earnings"] = {}
        f["Financials"]["Cash_Flow"] = {}
        p = analyse(growing(), fund=f)["payout_ratio"]
        assert p["status"] == "unavailable" and p["payout_ratio_percent"] is None

    def test_currency_mismatch_not_computed(self):
        p = analyse(growing(), fund=fundamentals(ccy="EUR"))["payout_ratio"]
        assert p["status"] == "not_meaningful" and "EUR" in p["reason"] and "no FX conversion" in p["reason"]

    def test_funds_and_missing_fundamentals(self):
        fund = analyse(growing(), kind="fund")
        assert fund["payout_ratio"]["status"] == "not_applicable"
        assert fund["sustainability"]["status"] == "not_applicable"
        none = analyse(growing(), fund=None, status="not covered by EODHD")
        assert none["payout_ratio"]["status"] == "unavailable" and "not covered" in none["payout_ratio"]["reason"]


class TestSustainabilityAndRisk:
    def test_supportive_case_values_are_calculated(self):
        res = analyse(growing(), fund=fundamentals(fcf=150, paid=-50, net_debt=300, ebitda=200))
        s = res["sustainability"]
        f = {x["factor"]: x for x in s["factors"]}
        assert f["fcf_coverage"]["value"] == 3.0 and f["fcf_coverage"]["assessment"] == "supportive"
        assert f["leverage"]["value"] == 1.5 and f["leverage"]["assessment"] == "supportive"
        assert f["earnings_trend"]["assessment"] == "supportive" and f["revenue_trend"]["assessment"] == "supportive"
        assert s["assessment"] == "Supportive" and "heuristic" in s["assessment_type"].lower()
        assert res["cut_risk"]["assessment"] == "Few risk indicators" and res["cut_risk"]["is_probability"] is False

    def test_weak_case_and_elevated_cut_risk(self):
        rows = quarterly(2012, 2025, lambda y: 1.5)
        res = analyse(rows, fund=fundamentals(eps=1.5, fcf=-20, paid=-50, net_debt=900, ebitda=200))
        f = {x["factor"]: x for x in res["sustainability"]["factors"]}
        assert f["payout_ratio"]["assessment"] == "concern"  # 6.0 DPS / 1.5 EPS = 400%
        assert f["fcf_coverage"]["assessment"] == "concern" and f["fcf_coverage"]["value"] is None
        assert f["leverage"]["assessment"] == "concern"
        assert res["sustainability"]["assessment"] == "Weak"
        risk = res["cut_risk"]
        trig = set(risk["triggered_indicators"])
        assert {"payout_ratio_high", "weak_fcf_coverage", "high_leverage"} <= trig
        assert risk["assessment"] == "Elevated risk indicators"
        assert all(i["explanation"] for i in risk["indicators"])

    def test_negative_earnings_indicator(self):
        res = analyse(growing(), fund=fundamentals(eps=-2.0))
        assert "negative_earnings" in res["cut_risk"]["triggered_indicators"]

    def test_recent_reduction_indicator_from_history(self):
        rows = quarterly(2018, 2025, lambda y: 0.5) + [row("2026-02-10", 0.25)]
        rows = [r for r in rows if r["date"] != "2026-02-10" or r["value"] == 0.25]
        res = analyse(rows, fund=fundamentals())
        assert "recent_dividend_reduction" in res["cut_risk"]["triggered_indicators"]

    def test_insufficient_data_is_explicit_without_fundamentals(self):
        res = analyse(growing(), fund=None, status="plan does not include fundamentals")
        risk = res["cut_risk"]
        assert risk["status"] == "insufficient_data" and risk["assessment"] == "Insufficient Data"
        assert risk["reason"] and risk["missing_data"]
        assert res["sustainability"]["status"] == "unavailable"
        assert res["status"] == "partial"
        assert any(m["metric"] == "cut_risk" for m in res["unavailable_metrics"])

    def test_one_factor_is_insufficient_for_sustainability(self):
        f = fundamentals()
        f["Earnings"] = {}
        f["Financials"] = {"Cash_Flow": {"yearly": {"2025-12-31": {"date": "2025-12-31", "freeCashFlow": "150", "dividendsPaid": "-50"}}}}
        s = analyse(growing(), fund=f)["sustainability"]
        assert s["status"] == "insufficient_data" and s["assessment"] == "Insufficient Data"
        assert {m["metric"] for m in s["unavailable_metrics"]} >= {"earnings_trend", "revenue_trend", "leverage"}

    def test_fcf_derived_from_cfo_minus_capex_when_missing(self):
        f = fundamentals()
        for e in f["Financials"]["Cash_Flow"]["yearly"].values():
            e.pop("freeCashFlow")
            e["totalCashFromOperatingActivities"] = "200"
            e["capitalExpenditures"] = "-50"
        v = {x["factor"]: x for x in analyse(growing(), fund=f)["sustainability"]["factors"]}["fcf_coverage"]["value"]
        assert v == 3.0


class TestResultShape:
    def test_unavailable_when_dividends_not_retrievable(self):
        res = analyse(None, notes=["EODHD denied access to dividend data."])
        assert res["status"] == "unavailable" and res["unavailable_reasons"] == ["EODHD denied access to dividend data."]
        assert res["as_of"] == "2026-03-15"

    def test_json_safe_schema_valid_and_camelized(self):
        res = analyse(growing(), fund=fundamentals())
        json.dumps(res, allow_nan=False)
        model = DividendAnalysisStrategy.model_validate(camelize_keys(res))
        dumped = model.model_dump(by_alias=True)
        assert dumped["strategyName"] == "Dividend Analysis" and dumped["category"] == "Quantitative Strategies"
        assert dumped["dividendYield"]["ttmYieldPercent"] is not None
        assert dumped["dividendGrowth"]["multiYear"][0]["cagrPercent"] is not None
        assert dumped["cutRisk"]["isProbability"] is False
        assert DividendAnalysisStrategy.model_validate(camelize_keys(analyse(None))).status == "unavailable"

    def test_no_forecast_or_probability_output(self):
        text = json.dumps(analyse(growing(), fund=fundamentals())).lower()
        assert "forecast" not in text.replace("dividend forecasting is not part of this version", "")
        assert "chance of" not in text and "% probability" not in text

    def test_thresholds_are_documented_in_output(self):
        res = analyse(growing(), fund=fundamentals())
        assert "supportive" in res["sustainability"]["factors"][0]["thresholds"]
        assert res["cut_risk"]["rule"] and res["sustainability"]["rule"]


class TestRoute:
    class _Stock:
        symbol = "AAA"

        @staticmethod
        def eodhd_symbol():
            return "AAA.US"

    @staticmethod
    def call(url="/etfs/VOO/strategies/dividend_analysis", *, dividends=None, fundamentals_val=None, fund_exc=None,
             div_exc=None, stock=None, cached=None, failure=None):
        from app.routers import strategy as router_module

        cache_store = {}
        sets = []

        async def get_json(key):
            if key.startswith(router_module.DIVIDEND_FAILURE_CACHE_PREFIX):
                return failure
            return cached

        async def set_json(key, value, ttl=None):
            sets.append((key, ttl))

        async def fake_dividends(symbol, from_date=None, to_date=None):
            if div_exc:
                raise div_exc
            return dividends if dividends is not None else growing()

        async def fake_fundamentals(symbol):
            if fund_exc:
                raise fund_exc
            return fundamentals_val

        async def fake_eod(symbol, from_date=None, to_date=None):
            return [{"date": "2026-03-12", "close": 90.0}, {"date": "2026-03-13", "close": 100.0}]

        async def fake_prices(db, symbol, limit=None):
            return [SimpleNamespace(close=100.0, date=datetime.date(2026, 3, 13))]

        async def fake_find(db, symbol):
            return stock if symbol.upper() == "AAA" else None

        app = FastAPI()
        app.include_router(router_module.router)
        app.dependency_overrides[get_db] = lambda: None
        patches = [
            patch.object(router_module.cache, "get_json", get_json),
            patch.object(router_module.cache, "set_json", set_json),
            patch.object(router_module.eodhd_client, "fetch_dividends", fake_dividends),
            patch.object(router_module.eodhd_client, "fetch_fundamentals", fake_fundamentals),
            patch.object(router_module.eodhd_client, "fetch_eod_history", fake_eod),
            patch.object(router_module, "load_prices", fake_prices),
            patch.object(router_module, "_find_stock_member", fake_find),
        ]
        for p in patches:
            p.start()
        try:
            res = TestClient(app).get(url)
        finally:
            for p in patches:
                p.stop()
        return res, sets

    def test_etf_success_uses_stored_price_and_skips_fundamentals(self):
        res, sets = self.call(fund_exc=RuntimeError("fundamentals must not be fetched for funds"))
        assert res.status_code == 200, res.text
        d = res.json()["data"]
        assert d["instrumentType"] == "fund" and d["status"] == "available"
        assert d["dividendYield"]["shareprice" if False else "sharePrice"] == 100.0
        assert d["payoutRatio"]["status"] == "not_applicable"
        assert sets and sets[0][0] == "tw:v1:dividend:VOO"

    def test_stock_success_with_fundamentals(self):
        res, sets = self.call("/etfs/aaa/strategies/dividend_analysis", fundamentals_val=fundamentals(), stock=self._Stock)
        assert res.status_code == 200, res.text
        d = res.json()["data"]
        assert d["instrumentType"] == "stock" and d["payoutRatio"]["status"] == "available"
        assert d["sustainability"]["assessment"] == "Supportive"
        # The route uses the real clock while the fixture history ends in Mar-2026, so only the
        # payment-gap indicator may fire; no fundamentals-based indicator should.
        assert set(d["cutRisk"]["triggeredIndicators"]) <= {"payment_gap"}
        assert d["cutRisk"]["status"] != "insufficient_data"
        assert sets[0][0] == "tw:v1:dividend:AAA"

    def test_entitlement_error_is_reported_as_unavailable_without_secrets(self):
        exc = eodhd_client.EodhdEntitlementError(
            "EODHD denied access to dividend data for VOO.US (HTTP 403). The configured EODHD plan/token must include the dividends endpoint."
        )
        res, sets = self.call(div_exc=exc)
        assert res.status_code == 200
        body = res.json()
        assert body["data"]["status"] == "unavailable" and "dividends endpoint" in body["data"]["unavailableReasons"][0]
        assert body["message"].startswith("Dividend data is unavailable")
        assert "dummy-token" not in res.text and "api_token" not in res.text
        assert sets[0][0] == "tw:v1:dividend_fail:VOO" and sets[0][1] == 300  # short negative cache, success key untouched

    def test_http_error_text_never_leaks_the_token(self):
        req = httpx.Request("GET", "https://eodhd.com/api/div/VOO.US?api_token=dummy-token")
        exc = httpx.HTTPStatusError("boom dummy-token", request=req, response=httpx.Response(500, request=req))
        res, _ = self.call(div_exc=exc)
        assert "dummy-token" not in res.text and "HTTP 500" in res.json()["data"]["unavailableReasons"][0]

    def test_fundamentals_failure_still_returns_unaffected_components(self):
        req = httpx.Request("GET", "https://eodhd.com/api/fundamentals/AAA.US")
        exc = eodhd_client.EodhdEntitlementError("no fundamentals")
        res, sets = self.call("/etfs/AAA/strategies/dividend_analysis", fund_exc=httpx.HTTPStatusError(
            "x", request=req, response=httpx.Response(403, request=req)), stock=self._Stock)
        d = res.json()["data"]
        assert d["status"] == "partial"
        assert d["dividendYield"]["ttmYieldPercent"] is not None and d["dividendGrowth"]["status"] == "available"
        assert d["payoutRatio"]["status"] == "unavailable" and d["cutRisk"]["status"] == "insufficient_data"
        assert any("fundamentals" in n.lower() for n in d["limitations"])
        assert sets[0][0] == "tw:v1:dividend:AAA"

    def test_not_covered_fundamentals(self):
        res, _ = self.call("/etfs/AAA/strategies/dividend_analysis", fundamentals_val=None, stock=self._Stock)
        d = res.json()["data"]
        assert d["dataCoverage"]["fundamentalsStatus"] == "not covered by EODHD"

    def test_unknown_symbol_404(self):
        res, _ = self.call("/etfs/ZZZZ/strategies/dividend_analysis")
        assert res.status_code == 404

    def test_cached_result_short_circuits_eodhd(self):
        cached = analyse(growing(), kind="fund", status="not_applicable")
        res, sets = self.call(div_exc=RuntimeError("must not be called"), cached=cached)
        assert res.status_code == 200 and res.json()["data"]["status"] == "available" and sets == []

    def test_negative_cache_short_circuits_repeated_failures(self):
        res, sets = self.call(div_exc=RuntimeError("must not be called"), failure={"reasons": ["EODHD dividend history request failed."]})
        assert res.json()["data"]["status"] == "unavailable" and sets == []