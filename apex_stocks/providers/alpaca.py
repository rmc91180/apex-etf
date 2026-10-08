"""Alpaca market data (https://docs.alpaca.markets/reference).

Needs ALPACA_KEY and ALPACA_SECRET (a free account is enough). Notes on the
free plan, which shape the choices below:

* Consolidated (SIP) data is free only when at least 15 minutes old, so
  history and pre-market volume use SIP with the end time held back 16
  minutes. That keeps pre-market volume on the same scale as the daily
  volume it is compared with.
* Real-time data on the free plan is IEX only (a few percent of volume). The
  latest IEX trade is used for the freshest price. IEX quotes before the open
  are too thin to judge spreads, so bid/ask are left empty pre-market unless
  ALPACA_DATA_FEED=sip (paid).
* Alpaca does not report market cap, float or trading halts. Halted stocks
  show up as stale quotes and are rejected for that.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import time
from typing import Any, Iterable, Sequence

import requests

from ..market_calendar import ET
from ..models import Bar, NewsItem, Security, Snapshot
from ..universe import classify_asset
from .base import DataUnavailable

DATA_URL = "https://data.alpaca.markets"
TRADING_URLS = ("https://paper-api.alpaca.markets", "https://api.alpaca.markets")
SIP_DELAY = dt.timedelta(minutes=16)
# Plain listed tickers, optionally with a share-class suffix (BRK.B). Delisted
# assets carry names like "CCT_DELISTED" that the data API rejects.
VALID_SYMBOL = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,2})?$")
_INVALID_SYMBOL = re.compile(r"invalid symbol: ([^\"'}\s,]+)")
EXCHANGE_MAP = {"NYSE": "NYSE", "NASDAQ": "NASDAQ", "AMEX": "AMEX", "ARCA": "ARCA", "BATS": "BATS", "OTC": "OTC"}


def _chunks(xs: Sequence[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield list(xs[i:i + n])


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def _rfc(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bar(b: dict) -> Bar:
    return Bar(_ts(b["t"]), b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw"))


class AlpacaProvider:
    name = "alpaca"
    is_live = True

    def __init__(self, key: str | None = None, secret: str | None = None, feed: str = "iex",
                 now=None, session: requests.Session | None = None, min_interval: float = 0.32,
                 max_retries: int = 4):
        key = key or os.environ.get("ALPACA_KEY", "")
        secret = secret or os.environ.get("ALPACA_SECRET", "")
        if not key or not secret:
            raise DataUnavailable("ALPACA_KEY and ALPACA_SECRET are not set")
        self.http = session or requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.feed = feed
        self._now = now or (lambda: dt.datetime.now(ET))
        self.min_interval, self.max_retries = min_interval, max_retries
        self._last_call = 0.0

    # ── HTTP ───────────────────────────────────────────────────
    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)  # free plan allows 200 requests a minute
            self._last_call = time.monotonic()
            try:
                r = self.http.get(url, params=params, timeout=30)
            except requests.RequestException as e:
                err = str(e)
            else:
                if r.status_code == 200:
                    return r.json()
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code in (400, 401, 403, 404, 422):
                    raise DataUnavailable(f"{url}: {err}")
            time.sleep(min(30, 2 ** attempt))
        raise DataUnavailable(f"{url}: {err}")

    def _paged(self, url: str, params: dict[str, Any], key: str, max_pages: int = 50) -> Iterable[dict]:
        params = dict(params)
        for _ in range(max_pages):
            data = self._get(url, params)
            yield data.get(key) or {}
            token = data.get("next_page_token")
            if not token:
                return
            params["page_token"] = token
        raise DataUnavailable(f"{url}: more than {max_pages} pages")

    # ── interface ──────────────────────────────────────────────
    def list_universe(self, status: str = "active") -> list[Security]:
        """Active assets. status="inactive" lists delisted ones, which backtests need to avoid survivorship bias."""
        last_err = None
        for base in TRADING_URLS:  # paper keys work on paper-api, live keys on api
            try:
                assets = self._get(f"{base}/v2/assets", {"status": status, "asset_class": "us_equity"})
                break
            except DataUnavailable as e:
                last_err = e
        else:
            raise DataUnavailable(f"asset list unavailable: {last_err}")
        out = []
        for a in assets:
            if not VALID_SYMBOL.match(a.get("symbol", "")):
                continue
            exch = EXCHANGE_MAP.get(a.get("exchange", ""), a.get("exchange", ""))
            out.append(Security(
                symbol=a["symbol"], name=a.get("name") or "", exchange=exch,
                asset_type=classify_asset(a["symbol"], a.get("name") or "", exch),
                tradable=status != "active" or (bool(a.get("tradable")) and a.get("status") == "active"),
            ))
        return out

    def get_snapshots(self, symbols: Sequence[str], include_premarket: bool = False) -> dict[str, Snapshot]:
        now = self._now().astimezone(ET)
        raw: dict[str, dict] = {}
        for batch in _chunks(list(symbols), 200):
            data = self._get(f"{DATA_URL}/v2/stocks/snapshots", {"symbols": ",".join(batch), "feed": self.feed})
            raw.update(data.get("snapshots", data))
        pm = self._premarket(list(raw), now) if include_premarket else {}
        regular_open = now.time() >= dt.time(9, 30)
        out = {}
        for sym, s in raw.items():
            if not s:
                continue
            trade, quote = s.get("latestTrade") or {}, s.get("latestQuote") or {}
            prev = s.get("prevDailyBar") or {}
            day = s.get("dailyBar") or {}
            times = [_ts(x["t"]) for x in (trade, quote) if x.get("t")]
            price = trade.get("p")
            pm_price, pm_vol, pm_time = pm.get(sym, (None, None, None))
            if pm_time:
                times.append(pm_time)
            if price is None:
                price = pm_price
            if price is None or not times:
                continue
            use_quote = (self.feed == "sip" or regular_open) and quote.get("bp", 0) > 0 and quote.get("ap", 0) > 0
            # Alpaca's dailyBar is the latest completed or current session; only today's counts as the open.
            day_is_today = day.get("t") and _ts(day["t"]).astimezone(ET).date() == now.date()
            prev_close = prev.get("c") if day_is_today else day.get("c") or prev.get("c")
            out[sym] = Snapshot(
                symbol=sym, as_of=max(times), source=f"alpaca:{self.feed}",
                last_price=price,
                bid=quote.get("bp") if use_quote else None,
                ask=quote.get("ap") if use_quote else None,
                prev_close=prev_close,
                day_open=day.get("o") if day_is_today and regular_open else None,
                day_volume=day.get("v") if day_is_today else None,
                vwap=day.get("vw") if day_is_today else None,
                premarket_price=price if not regular_open else pm_price,
                premarket_volume=pm_vol if include_premarket else None,
            )
        return out

    def _premarket(self, symbols: list[str], now: dt.datetime) -> dict[str, tuple[float, float, dt.datetime]]:
        """(last price, volume, last bar time) from 4:00 ET to now, consolidated, 16 minutes delayed."""
        start = now.replace(hour=4, minute=0, second=0, microsecond=0)
        end = min(now - SIP_DELAY, now.replace(hour=9, minute=30, second=0, microsecond=0))
        out: dict[str, tuple[float, float, dt.datetime]] = {}
        if end <= start:
            return {s: (None, 0.0, None) for s in symbols}
        for batch in _chunks(symbols, 100):
            acc: dict[str, list[dict]] = {}
            for page in self._paged(f"{DATA_URL}/v2/stocks/bars", {
                "symbols": ",".join(batch), "timeframe": "5Min", "start": _rfc(start), "end": _rfc(end),
                "feed": "sip", "limit": 10000, "adjustment": "raw"}, "bars"):
                for sym, bars in page.items():
                    acc.setdefault(sym, []).extend(bars)
            for sym in batch:
                bars = acc.get(sym, [])
                if bars:
                    out[sym] = (bars[-1]["c"], float(sum(b["v"] for b in bars)), _ts(bars[-1]["t"]) + dt.timedelta(minutes=5))
                else:
                    out[sym] = (None, 0.0, None)
        return out

    def bars_raw(self, symbols: Sequence[str], timeframe: str, start: dt.datetime, end: dt.datetime,
                 adjustment: str = "raw", batch_size: int = 100) -> dict[str, list[dict]]:
        """Consolidated bars as Alpaca returns them, never newer than the free-plan delay."""
        end = min(end, self._now() - SIP_DELAY)
        out: dict[str, list[dict]] = {}
        if end <= start:
            return out
        for batch in _chunks(list(symbols), batch_size):
            while batch:
                got: dict[str, list[dict]] = {}
                try:
                    for page in self._paged(f"{DATA_URL}/v2/stocks/bars", {
                        "symbols": ",".join(batch), "timeframe": timeframe, "start": _rfc(start), "end": _rfc(end),
                        "feed": "sip", "limit": 10000, "adjustment": adjustment}, "bars", max_pages=500):
                        for sym, bars in page.items():
                            got.setdefault(sym, []).extend(bars)
                except DataUnavailable as e:
                    bad = _INVALID_SYMBOL.search(str(e))
                    if not bad or bad.group(1) not in batch:
                        raise
                    batch = [s for s in batch if s != bad.group(1)]  # drop it and retry the rest
                    continue
                for sym, bars in got.items():
                    out.setdefault(sym, []).extend(bars)
                break
        return out

    def get_daily_bars_many(self, symbols: Sequence[str], start: dt.date, end: dt.date) -> dict[str, list[Bar]]:
        raw = self.bars_raw(symbols, "1Day", dt.datetime.combine(start, dt.time(0, 0), ET),
                            dt.datetime.combine(end, dt.time(23, 59), ET), adjustment="all")
        return {s: [_bar(b) for b in bars] for s, bars in raw.items()}

    def get_daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> list[Bar]:
        bars = self.get_daily_bars_many([symbol], start, end).get(symbol)
        if not bars:
            raise DataUnavailable(f"no daily bars for {symbol}")
        return bars

    def get_intraday_bars(self, symbol: str, start: dt.datetime, end: dt.datetime, minutes: int) -> list[Bar]:
        if minutes not in (1, 5, 15):
            raise ValueError("minutes must be 1, 5 or 15")
        end = min(end, self._now() - SIP_DELAY)
        bars: list[Bar] = []
        for page in self._paged(f"{DATA_URL}/v2/stocks/bars", {
            "symbols": symbol, "timeframe": f"{minutes}Min", "start": _rfc(start), "end": _rfc(end),
            "feed": "sip", "limit": 10000, "adjustment": "raw"}, "bars"):
            bars.extend(_bar(b) for b in page.get(symbol, []))
        return bars

    def get_news(self, symbols: Sequence[str], since: dt.datetime, until: dt.datetime | None = None) -> list[NewsItem]:
        items: dict[str, NewsItem] = {}
        params = {"start": _rfc(since), "limit": 50, "sort": "desc", "include_content": "false"}
        if until is not None:
            params["end"] = _rfc(until)
        for batch in _chunks(list(symbols), 50):
            for page in self._paged(f"{DATA_URL}/v1beta1/news", dict(params, symbols=",".join(batch)),
                                    "news", max_pages=10):
                for n in page:
                    nid = str(n["id"])
                    items[nid] = NewsItem(
                        id=f"alpaca-{nid}", symbols=tuple(n.get("symbols") or ()), headline=n.get("headline", ""),
                        source=n.get("source") or n.get("author") or "", url=n.get("url") or "",
                        published_at=_ts(n["created_at"]), summary=n.get("summary") or "", provider="alpaca")
        return sorted(items.values(), key=lambda n: n.published_at, reverse=True)
