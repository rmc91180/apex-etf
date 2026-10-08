"""Replays cached history as if it were a given pre-market moment.

It answers every provider call with only what was knowable at ``now``:
daily bars before the trade date, pre-market bars that had finished (16
minutes delayed, as on the live free plan), and news published by ``now``.
"""
from __future__ import annotations

import datetime as dt
from typing import Sequence

from ..history import DailyTable, HistoryCache
from ..market_calendar import ET, MarketCalendar
from ..models import Bar, NewsItem, Security, Snapshot
from .alpaca import SIP_DELAY
from .base import DataUnavailable


class ReplayProvider:
    name = "replay"
    is_live = False

    def __init__(self, cache: HistoryCache, universe: list[Security], daily: DailyTable, now: dt.datetime,
                 calendar: MarketCalendar | None = None):
        self.cache, self.daily, self.now = cache, daily, now.astimezone(ET)
        self.cal = calendar or MarketCalendar()
        self.trade_date = self.now.date()
        known = set(daily.symbols())
        self._universe = [s for s in universe if s.symbol in known]
        # the latest session strictly before the trade date
        d = self.trade_date - dt.timedelta(days=1)
        while not self.cal.is_trading_day(d):
            d -= dt.timedelta(days=1)
        self._prev = self.cal.session(d)

    def list_universe(self) -> list[Security]:
        return list(self._universe)

    def _prev_close(self, sym: str) -> float | None:
        row = self.daily.on(sym, self._prev.date)
        return row["c"] if row else None

    def get_snapshots(self, symbols: Sequence[str], include_premarket: bool = False) -> dict[str, Snapshot]:
        out = {}
        pm = {}
        if include_premarket:
            start = self.now.replace(hour=4, minute=0, second=0, microsecond=0)
            end = self.now - SIP_DELAY
            raw = self.cache.premarket(self.trade_date, list(symbols), start, end)
            for s, bars in raw.items():
                bars = [b for b in bars if dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                        + dt.timedelta(minutes=5) <= end]
                pm[s] = bars
        for s in symbols:
            pc = self._prev_close(s)
            if pc is None:
                continue
            if not include_premarket:
                out[s] = Snapshot(s, self.now, "replay", last_price=pc, prev_close=pc)
                continue
            bars = pm.get(s, [])
            if bars:
                last_t = dt.datetime.fromisoformat(bars[-1]["t"].replace("Z", "+00:00")) + dt.timedelta(minutes=5)
                price, vol = bars[-1]["c"], float(sum(b["v"] for b in bars))
                out[s] = Snapshot(s, last_t, "replay", last_price=price, prev_close=pc,
                                  premarket_price=price, premarket_volume=vol)
            else:
                out[s] = Snapshot(s, self._prev.close, "replay", last_price=pc, prev_close=pc,
                                  premarket_price=None, premarket_volume=0.0)
        return out

    def get_daily_bars_many(self, symbols: Sequence[str], start: dt.date, end: dt.date) -> dict[str, list[Bar]]:
        end = min(end, self.trade_date - dt.timedelta(days=1))  # never the trade date or later
        out = {}
        for s in symbols:
            w = self.daily.window(s, start, end)
            if w is None or len(w["d"]) == 0:
                continue
            out[s] = [Bar(dt.datetime.combine(dt.date.fromordinal(int(d)), dt.time(9, 30), ET),
                          float(o), float(h), float(l), float(c), float(v))
                      for d, o, h, l, c, v in zip(w["d"], w["o"], w["h"], w["l"], w["c"], w["v"])]
        return out

    def get_daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> list[Bar]:
        bars = self.get_daily_bars_many([symbol], start, end).get(symbol)
        if not bars:
            raise DataUnavailable(symbol)
        return bars

    def get_intraday_bars(self, symbol, start, end, minutes):
        raise DataUnavailable("intraday bars are not replayed before the open")

    def get_news(self, symbols: Sequence[str], since: dt.datetime) -> list[NewsItem]:
        items = self.cache.news(self.trade_date, list(symbols), since, self.now)
        return [n for n in items if since <= n.published_at <= self.now]
