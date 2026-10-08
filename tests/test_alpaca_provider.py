"""Alpaca provider against recorded-shape responses (no network)."""
import datetime as dt

import pytest

from apex_stocks.market_calendar import ET
from apex_stocks.providers.alpaca import AlpacaProvider
from apex_stocks.providers.base import DataUnavailable, MarketDataProvider

NOW = dt.datetime(2026, 10, 7, 8, 0, tzinfo=ET)


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body


class FakeHTTP:
    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for key, handler in self.routes.items():
            if url.endswith(key):
                r = handler(params or {}) if callable(handler) else handler
                return r if isinstance(r, Resp) else Resp(200, r)
        return Resp(404, {"message": "not found"})


def provider(routes, now=NOW):
    return AlpacaProvider("k", "s", now=lambda: now, session=FakeHTTP(routes), min_interval=0, max_retries=2)


def test_requires_credentials(monkeypatch):
    monkeypatch.delenv("ALPACA_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET", raising=False)
    with pytest.raises(DataUnavailable):
        AlpacaProvider()


def test_implements_interface():
    assert isinstance(provider({}), MarketDataProvider)


def test_universe_classifies_assets_and_falls_back_to_live_endpoint():
    assets = [
        {"symbol": "ABCD", "name": "Abcd Therapeutics Inc. Common Stock", "exchange": "NASDAQ", "tradable": True, "status": "active"},
        {"symbol": "ABCDW", "name": "Abcd Therapeutics Inc. Warrant", "exchange": "NASDAQ", "tradable": True, "status": "active"},
        {"symbol": "SOXL", "name": "Direxion Daily Semiconductor Bull 3X Shares", "exchange": "ARCA", "tradable": True, "status": "active"},
        {"symbol": "XYZ.U", "name": "XYZ Acquisition Corp Units", "exchange": "NYSE", "tradable": True, "status": "active"},
        {"symbol": "BIGA", "name": "Big Acquisition Corp Class A", "exchange": "NYSE", "tradable": True, "status": "active"},
        {"symbol": "OLD", "name": "Old Co", "exchange": "OTC", "tradable": False, "status": "active"},
    ]
    http_routes = {"paper-api.alpaca.markets/v2/assets": Resp(401, {}), "api.alpaca.markets/v2/assets": assets}
    p = provider(http_routes)
    got = {s.symbol: (s.asset_type, s.exchange, s.tradable) for s in p.list_universe()}
    assert got == {
        "ABCD": ("common", "NASDAQ", True),
        "ABCDW": ("warrant", "NASDAQ", True),
        "SOXL": ("etf", "ARCA", True),
        "XYZ.U": ("unit", "NYSE", True),
        "BIGA": ("spac", "NYSE", True),
        "OLD": ("common", "OTC", False),
    }


SNAP = {
    "ABCD": {
        "latestTrade": {"t": "2026-10-07T11:58:00Z", "p": 8.42},
        "latestQuote": {"t": "2026-10-07T11:59:00Z", "bp": 8.40, "ap": 8.44},
        "dailyBar": {"t": "2026-10-06T04:00:00Z", "o": 7.9, "h": 8.1, "l": 7.8, "c": 8.0, "v": 3e6, "vw": 7.95},
        "prevDailyBar": {"t": "2026-10-05T04:00:00Z", "o": 7.5, "h": 7.9, "l": 7.4, "c": 7.8, "v": 2e6},
    },
    "EMPTY": None,
}


def premarket_bars(params):
    assert params["feed"] == "sip" and params["timeframe"] == "5Min"
    end = dt.datetime.fromisoformat(params["end"].replace("Z", "+00:00"))
    assert end <= NOW - dt.timedelta(minutes=16)  # free-plan SIP delay respected
    return {"bars": {"ABCD": [
        {"t": "2026-10-07T11:00:00Z", "o": 8.1, "h": 8.3, "l": 8.1, "c": 8.3, "v": 100000},
        {"t": "2026-10-07T11:35:00Z", "o": 8.3, "h": 8.45, "l": 8.3, "c": 8.4, "v": 150000},
    ]}, "next_page_token": None}


def test_snapshot_premarket_uses_previous_session_close_and_drops_iex_quotes():
    p = provider({"/v2/stocks/snapshots": SNAP, "/v2/stocks/bars": premarket_bars})
    s = p.get_snapshots(["ABCD", "EMPTY"], include_premarket=True)
    assert list(s) == ["ABCD"]
    a = s["ABCD"]
    assert a.prev_close == 8.0            # dailyBar is yesterday's session pre-market
    assert a.last_price == a.premarket_price == 8.42
    assert a.premarket_volume == 250000
    assert a.bid is None and a.ask is None  # IEX quotes ignored before the open
    assert a.as_of == dt.datetime(2026, 10, 7, 11, 59, tzinfo=dt.timezone.utc)
    assert a.source == "alpaca:iex"


def test_snapshot_keeps_quotes_with_sip_feed():
    p = provider({"/v2/stocks/snapshots": SNAP})
    p.feed = "sip"
    a = p.get_snapshots(["ABCD"])["ABCD"]
    assert (a.bid, a.ask) == (8.40, 8.44)
    assert a.premarket_volume is None


def test_daily_bars_paginate_and_never_end_inside_sip_delay():
    pages = iter([
        {"bars": {"ABCD": [{"t": "2026-10-05T04:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}]}, "next_page_token": "p2"},
        {"bars": {"ABCD": [{"t": "2026-10-06T04:00:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 20}]}, "next_page_token": None},
    ])
    seen = []

    def bars(params):
        seen.append(params)
        return next(pages)

    p = provider({"/v2/stocks/bars": bars})
    out = p.get_daily_bars_many(["ABCD"], dt.date(2026, 10, 1), dt.date(2026, 10, 7))
    assert [b.close for b in out["ABCD"]] == [2, 3]
    assert seen[1]["page_token"] == "p2"
    assert seen[0]["adjustment"] == "all"
    end = dt.datetime.fromisoformat(seen[0]["end"].replace("Z", "+00:00"))
    assert end <= NOW - dt.timedelta(minutes=16)


def test_news_is_deduplicated_and_sorted():
    item = {"id": 1, "headline": "ABCD wins contract", "created_at": "2026-10-07T11:00:00Z", "symbols": ["ABCD"],
            "source": "benzinga", "url": "https://x", "summary": ""}
    later = dict(item, id=2, created_at="2026-10-07T11:30:00Z")
    p = provider({"/v1beta1/news": {"news": [item, later, item], "next_page_token": None}})
    news = p.get_news(["ABCD"], NOW - dt.timedelta(hours=20))
    assert [n.id for n in news] == ["alpaca-2", "alpaca-1"]
    assert news[0].provider == "alpaca"


def test_auth_errors_surface_as_data_unavailable():
    p = provider({"/v2/stocks/snapshots": Resp(403, {"message": "forbidden"})})
    with pytest.raises(DataUnavailable):
        p.get_snapshots(["ABCD"])


def test_server_errors_are_retried():
    responses = iter([Resp(500, {}), SNAP])
    p = provider({"/v2/stocks/snapshots": lambda params: next(responses)})
    assert "ABCD" in p.get_snapshots(["ABCD"])


def test_delisted_placeholder_symbols_are_skipped_and_invalid_ones_dropped_from_batches():
    assets = [{"symbol": "CCT_DELISTED", "name": "x", "exchange": "NYSE", "tradable": False, "status": "inactive"},
              {"symbol": "BRK.B", "name": "Berkshire Hathaway Class B", "exchange": "NYSE", "tradable": True, "status": "active"}]
    p = provider({"/v2/assets": assets})
    assert [s.symbol for s in p.list_universe()] == ["BRK.B"]

    def bars(params):
        if "OLD" in params["symbols"].split(","):
            return Resp(400, {"message": "invalid symbol: OLD"})
        return {"bars": {"ABCD": [{"t": "2026-10-05T04:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 10}]}}

    p = provider({"/v2/stocks/bars": bars})
    out = p.bars_raw(["ABCD", "OLD"], "1Day", NOW - dt.timedelta(days=5), NOW - dt.timedelta(days=1))
    assert list(out) == ["ABCD"]
