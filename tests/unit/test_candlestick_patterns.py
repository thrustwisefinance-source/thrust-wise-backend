"""
Tests for Candlestick Pattern Recognition (app.services.candlestick_patterns)
and its routes. Synthetic candles built so each pattern's documented rule is
met (or deliberately missed); no network, no database, no TA-Lib.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database import get_db
from app.schemas import CandlestickPatternRecognitionStrategy
from app.services import strategies as cp

D0 = datetime.date(2024, 1, 1)
NOW = datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc)  # long after every test date


def candle(o, h, l, c, v=1000):
    return (o, h, l, c, v)


def to_prices(candles, start=D0, skip=()):
    """Candle tuples -> DailyPrice-like objects on consecutive weekdays."""
    out, d = [], start
    for i, (o, h, l, c, v) in enumerate(candles):
        while d.weekday() >= 5:
            d += datetime.timedelta(days=1)
        out.append(SimpleNamespace(symbol="T", date=d, open=o, high=h, low=l, close=c, adjusted_close=c, volume=v))
        d += datetime.timedelta(days=1)
    return out


def down_prefix(n=40):
    """Steady decline: closes fall 2/bar, bearish 2-point bodies."""
    return [candle(150 - 2 * i + 2, 150 - 2 * i + 2.5, 150 - 2 * i - 0.5, 150 - 2 * i) for i in range(n)]


def up_prefix(n=40):
    return [candle(100 + 2 * i - 2, 100 + 2 * i + 0.5, 100 + 2 * i - 2.5, 100 + 2 * i) for i in range(n)]


def detect(candles, **kw):
    return cp.compute_candlestick_patterns(to_prices(candles), symbol="T", now=NOW, **kw)


def names(result, last_only=True):
    dets = result["latest_candle_detections"] if last_only else result["detections"]
    return {d["pattern"] for d in dets}


DOWN_END = 150 - 2 * 39  # 72
UP_END = 100 + 2 * 39  # 178


class TestSingleCandlePatterns:
    def test_hammer_after_downtrend_is_bullish(self):
        res = detect(down_prefix() + [candle(70.5, 71.7, 65, 71.5)])
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Hammer"]
        assert det and det[0]["classification"] == "Bullish" and det[0]["prior_trend"] == "Downtrend"
        assert "Hanging Man" not in names(res)

    def test_hanging_man_after_uptrend_is_bearish(self):
        res = detect(up_prefix() + [candle(UP_END - 1.5, UP_END - 0.3, UP_END - 8, UP_END - 0.5)])
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Hanging Man"]
        assert det and det[0]["classification"] == "Bearish" and det[0]["prior_trend"] == "Uptrend"
        assert "Hammer" not in names(res)

    def test_hammer_shape_without_trend_context_is_not_reported(self):
        flat = [candle(100, 100.5, 99.5, 100.1) for _ in range(40)]
        res = detect(flat + [candle(100.5, 101.7, 95, 101.5)])
        assert not names(res) & {"Hammer", "Hanging Man"}

    def test_shooting_star_after_uptrend(self):
        p = UP_END
        res = detect(up_prefix() + [candle(p + 0.5, p + 7, p + 0.4, p + 1.5)])
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Shooting Star"]
        assert det and det[0]["classification"] == "Bearish"

    def test_doji_is_neutral_in_any_context(self):
        res = detect(down_prefix() + [candle(70, 73, 67, 70.1)])
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Doji"]
        assert det and det[0]["classification"] == "Neutral" and det[0]["prior_trend"] == "Not required"

    def test_zero_range_candle_is_not_a_doji(self):
        res = detect(down_prefix() + [candle(70, 70, 70, 70)])
        assert "Doji" not in names(res)


class TestTwoCandlePatterns:
    def test_bullish_engulfing(self):
        res = detect(down_prefix() + [candle(72, 72.3, 69.8, 70), candle(69.5, 72.8, 69.3, 72.5)])
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Bullish Engulfing"]
        assert det and det[0]["classification"] == "Bullish" and det[0]["candle_count"] == 2
        assert [c["close"] for c in det[0]["candles"]] == [70, 72.5]
        assert det[0]["start_date"] < det[0]["date"]

    def test_bearish_engulfing(self):
        p = UP_END
        res = detect(up_prefix() + [candle(p, p + 2.3, p - 0.2, p + 2), candle(p + 2.5, p + 2.7, p - 0.5, p - 0.3)])
        assert "Bearish Engulfing" in names(res)

    def test_engulfing_requires_strictly_larger_body(self):
        res = detect(down_prefix() + [candle(72, 72.3, 69.8, 70), candle(70, 72.2, 69.9, 72)])
        assert "Bullish Engulfing" not in names(res)

    def test_bullish_harami(self):
        res = detect(down_prefix() + [candle(72, 72.3, 67.8, 68), candle(69, 71.3, 68.9, 71)])
        assert "Bullish Harami" in names(res)
        assert "Bullish Engulfing" not in names(res)

    def test_bearish_harami(self):
        p = UP_END
        res = detect(up_prefix() + [candle(p, p + 4.3, p - 0.2, p + 4), candle(p + 3, p + 3.2, p + 0.8, p + 1)])
        assert "Bearish Harami" in names(res)

    def test_harami_body_must_be_strictly_inside(self):
        res = detect(down_prefix() + [candle(72, 72.3, 67.8, 68), candle(68, 71.3, 67.9, 71)])
        assert "Bullish Harami" not in names(res)


class TestThreeCandlePatterns:
    def test_morning_star(self):
        res = detect(
            down_prefix()
            + [candle(72, 72.3, 66.8, 67), candle(66, 66.8, 65.7, 66.5), candle(67, 70.3, 66.9, 70)]
        )
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Morning Star"]
        assert det and det[0]["classification"] == "Bullish" and det[0]["candle_count"] == 3
        assert len(det[0]["candles"]) == 3

    def test_morning_star_fails_if_third_candle_does_not_reach_midpoint(self):
        res = detect(
            down_prefix()
            + [candle(72, 72.3, 66.8, 67), candle(66, 66.8, 65.7, 66.5), candle(67, 69.1, 66.9, 69)]
        )
        assert "Morning Star" not in names(res)

    def test_evening_star(self):
        p = UP_END
        res = detect(
            up_prefix()
            + [candle(p, p + 5.3, p - 0.2, p + 5), candle(p + 5.5, p + 6.2, p + 5.4, p + 6), candle(p + 5, p + 5.2, p + 1.8, p + 2)]
        )
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Evening Star"]
        assert det and det[0]["classification"] == "Bearish"

    def test_three_black_crows(self):
        p = UP_END
        crows = [
            candle(p + 1, p + 1.2, p - 2.2, p - 2),
            candle(p - 1, p - 0.8, p - 5.2, p - 5),
            candle(p - 4, p - 3.8, p - 8.2, p - 8),
        ]
        res = detect(up_prefix() + crows)
        det = [d for d in res["latest_candle_detections"] if d["pattern"] == "Three Black Crows"]
        assert det and det[0]["classification"] == "Bearish" and len(det[0]["candles"]) == 3

    def test_crows_require_opens_within_previous_body(self):
        p = UP_END
        crows = [
            candle(p + 1, p + 1.2, p - 2.2, p - 2),
            candle(p - 6, p - 5.8, p - 9.2, p - 9),  # opens below prior close: gap, not a crow
            candle(p - 8, p - 7.8, p - 12.2, p - 12),
        ]
        assert "Three Black Crows" not in names(detect(up_prefix() + crows))


class TestHistoryAndDataHandling:
    def test_detections_over_history_carry_their_dates(self):
        candles = down_prefix() + [candle(70.5, 71.7, 65, 71.5)] + down_prefix(20)[1:]
        res = detect(candles, lookback_bars=200)
        hammer = [d for d in res["detections"] if d["pattern"] == "Hammer"]
        assert hammer
        assert hammer[0]["date"] == to_prices(candles)[40].date.isoformat()
        assert res["detection_counts"]["Hammer"] >= 1
        assert set(res["detection_counts"]) == set(cp.PATTERN_ORDER) and len(cp.PATTERN_ORDER) == 11

    def test_window_limits_reported_detections(self):
        candles = down_prefix() + [candle(70.5, 71.7, 65, 71.5)] + down_prefix(30)[1:]
        res = detect(candles, lookback_bars=10)
        assert not [d for d in res["detections"] if d["pattern"] == "Hammer"]

    def test_no_pattern_returns_empty_list_not_fabricated(self):
        res = detect([candle(100 + i * 0.1, 100.6 + i * 0.1, 99.9 + i * 0.1, 100.5 + i * 0.1) for i in range(60)])
        assert res["total_detections"] == len(res["detections"])
        assert not names(res, last_only=True) & {"Hammer", "Morning Star", "Three Black Crows"}

    def test_insufficient_candles_returns_none(self):
        assert detect(down_prefix(cp.MIN_ROWS_CANDLESTICK - 1)) is None
        assert detect([]) is None

    def test_invalid_ohlc_rows_dropped_and_counted(self):
        bad = candle(100, 90, 95, 100)  # high below open/close
        candles = down_prefix(45)
        candles[10] = bad
        candles[11] = candle(float("nan"), 1, 1, 1)
        candles[12] = candle(0, 1, 0, 1)
        res = detect(candles)
        assert res["data_quality"]["invalid_candles_dropped"] == 3
        assert len(res["data_quality"]["invalid_candle_dates"]) == 3
        assert any("excluded" in w for w in res["warnings"])

    def test_pattern_is_not_formed_across_a_dropped_candle(self):
        # engulfing pair separated by an invalid candle must NOT be reported
        candles = down_prefix() + [candle(72, 72.3, 69.8, 70), candle(100, 90, 95, 100), candle(69.5, 72.8, 69.3, 72.5)]
        assert "Bullish Engulfing" not in names(detect(candles))

    def test_duplicates_collapsed_and_sorted_chronologically(self):
        prices = to_prices(down_prefix(45))
        shuffled = list(reversed(prices)) + [prices[3]]
        res = cp.compute_candlestick_patterns(shuffled, symbol="T", now=NOW)
        assert res["data_quality"]["duplicate_dates_dropped"] == 1
        dates = [c["date"] for c in res["chart_data"]]
        assert dates == sorted(dates)

    def test_unsupported_interval_and_bad_lookback_raise(self):
        with pytest.raises(ValueError, match="Unsupported interval"):
            cp.compute_candlestick_patterns(to_prices(down_prefix()), interval="1h", now=NOW)
        with pytest.raises(ValueError, match="lookback_bars"):
            cp.compute_candlestick_patterns(to_prices(down_prefix()), lookback_bars=2, now=NOW)

    def test_incomplete_latest_daily_candle_is_excluded(self):
        prices = to_prices(down_prefix(45))
        last = prices[-1].date
        during_session = datetime.datetime.combine(last, datetime.time(15, 0), tzinfo=datetime.timezone.utc)  # 10-11am ET
        res = cp.compute_candlestick_patterns(prices, symbol="T", now=during_session)
        assert res["analysis_end"] < last.isoformat()
        assert res["data_quality"]["incomplete_latest_candle_excluded"] is True
        after_close = datetime.datetime.combine(last, datetime.time(23, 0), tzinfo=datetime.timezone.utc)
        assert cp.compute_candlestick_patterns(prices, symbol="T", now=after_close)["analysis_end"] == last.isoformat()

    def test_no_confidence_or_probability_fields_and_schema_roundtrip(self):
        res = detect(down_prefix() + [candle(70.5, 71.7, 65, 71.5)])
        text = str(res).lower()
        assert "confidence" not in text.replace("no probability or confidence", "") and "probability" not in text.replace(
            "no probability or confidence", ""
        ).replace("not predictors", "")
        model = CandlestickPatternRecognitionStrategy.model_validate(res)
        dumped = model.model_dump(by_alias=True)
        assert dumped["strategyId"] == "candlestick_pattern_recognition" and dumped["category"] == "Technical Strategies"
        assert dumped["detections"][0]["startDate"]

    def test_existing_ta_candles_strategy_untouched(self):
        from app.services import strategies

        prices = to_prices([candle(100 + i * 0.1, 100.6 + i * 0.1, 99.9 + i * 0.1, 100.5 + i * 0.1) for i in range(120)])
        assert strategies.compute_ta_candles_trend_health(prices) is not None


class TestRoutes:
    @staticmethod
    def _client(prices):
        from app.routers import strategy as router_module

        app = FastAPI()
        app.include_router(router_module.router)
        app.dependency_overrides[get_db] = lambda: None
        patches = [
            patch.object(router_module, "load_prices", AsyncMock(return_value=prices)),
            patch.object(router_module.cache, "get_json", AsyncMock(return_value=None)),
            patch.object(router_module.cache, "set_json", AsyncMock(return_value=None)),
        ]
        for p in patches:
            p.start()
        return TestClient(app), patches

    def _get(self, prices, url):
        client, patches = self._client(prices)
        try:
            return client.get(url)
        finally:
            for p in patches:
                p.stop()

    def test_dedicated_route_returns_detections(self):
        prices = to_prices(down_prefix() + [candle(70.5, 71.7, 65, 71.5)] + down_prefix(5)[1:])
        res = self._get(prices, "/etfs/VOO/strategies/candlestick_patterns?lookback_bars=30")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["lookbackBars"] == 30 and data["interval"] == "1d"
        assert any(d["pattern"] == "Hammer" for d in data["detections"])

    def test_unsupported_interval_is_422_with_clear_message(self):
        res = self._get(to_prices(down_prefix()), "/etfs/VOO/strategies/candlestick_patterns?interval=1h")
        assert res.status_code == 422 and "Unsupported interval" in res.json()["detail"]

    def test_unknown_symbol_404_and_insufficient_history_422(self):
        assert self._get(to_prices(down_prefix()), "/etfs/NOPE/strategies/candlestick_patterns").status_code == 404
        res = self._get(to_prices(down_prefix(10)), "/etfs/VOO/strategies/candlestick_patterns")
        assert res.status_code == 422 and "Insufficient" in res.json()["detail"]

    def test_lookback_out_of_range_rejected(self):
        res = self._get(to_prices(down_prefix()), "/etfs/VOO/strategies/candlestick_patterns?lookback_bars=2")
        assert res.status_code == 422