"""Deterministic synthetic market data for tests and dry runs.

Every value here is made up. ``is_live`` is False and every record is
stamped ``source="mock"`` so nothing produced from it can pass as a real
signal.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import random
from typing import Callable, Sequence

from ..market_calendar import ET, MarketCalendar
from ..models import Bar, NewsItem, Security, Snapshot
from .base import DataUnavailable

# (security, base price, typical daily share volume)
_FIXTURES: list[tuple[Security, float, float]] = [
    (Security("MOCKA", "Mock Alpha Corp", "NASDAQ", "common", market_cap=850e6, float_shares=60e6), 8.40, 4_000_000),
    (Security("MOCKB", "Mock Beta Inc", "NYSE", "common", market_cap=2.1e9, float_shares=150e6), 14.10, 2_500_000),
    (Security("MOCKC", "Mock Gamma Therapeutics", "NASDAQ", "common", market_cap=310e6, float_shares=40e6), 4.25, 6_000_000),
    (Security("THIN", "Mock Illiquid Co", "AMEX", "common", market_cap=20e6, float_shares=3e6), 2.10, 40_000),
    (Security("PRICY", "Mock Expensive Corp", "NYSE", "common", market_cap=40e9), 85.00, 9_000_000),
    (Security("MKETF", "Mock Leveraged ETF", "ARCA", "etf"), 12.00, 20_000_000),
    (Security("MOCKW", "Mock Alpha Corp Warrant", "NASDAQ", "warrant"), 0.80, 300_000),
    (Security("HALTD", "Mock Halted Inc", "NASDAQ", "common", halted=True, market_cap=500e6), 6.00, 1_000_000),
]

_NEWS_TEMPLATES = {
    "MOCKA": ("Mock Alpha Corp wins multi-year supply contract", "contract"),
    "MOCKC": ("Mock Gamma Therapeutics reports topline trial data", "regulatory"),
}


def _rng(*parts: object) -> random.Random:
    seed = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(seed[:16], 16))


class MockProvider:
    name = "mock"
    is_live = False

    def __init__(self, now: Callable[[], dt.datetime] | None = None, calendar: MarketCalendar | None = None):
        self._now = now or (lambda: dt.datetime.now(ET))
        self._cal = calendar or MarketCalendar()
        self._fixtures = {s.symbol: (s, p, v) for s, p, v in _FIXTURES}

    def _fixture(self, symbol: str):
        try:
            return self._fixtures[symbol]
        except KeyError:
            raise DataUnavailable(f"mock provider has no data for {symbol}") from None

    def list_universe(self) -> list[Security]:
        return [s for s, _, _ in _FIXTURES]

    def get_daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> list[Bar]:
        _, base, vol = self._fixture(symbol)
        bars, d = [], start
        while d <= end:
            s = self._cal.session(d)
            if s:
                r = _rng(symbol, d)
                # Price drifts around the base level so any date is reproducible on its own.
                close = round(base * (1 + 0.15 * (r.random() - 0.5)), 2)
                open_ = round(close * (1 + 0.03 * (r.random() - 0.5)), 2)
                high = round(max(open_, close) * (1 + 0.03 * r.random()), 2)
                low = round(min(open_, close) * (1 - 0.03 * r.random()), 2)
                v = round(vol * (0.5 + r.random()))
                bars.append(Bar(s.open, open_, high, low, close, v, round((high + low + close) / 3, 4)))
            d += dt.timedelta(days=1)
        return bars

    def get_intraday_bars(self, symbol: str, start: dt.datetime, end: dt.datetime, minutes: int) -> list[Bar]:
        if minutes not in (1, 5, 15):
            raise ValueError("minutes must be 1, 5 or 15")
        _, base, vol = self._fixture(symbol)
        bars, t = [], start.astimezone(ET)
        step = dt.timedelta(minutes=minutes)
        while t < end:
            s = self._cal.session(t.date())
            if s and s.open <= t < s.close:
                r = _rng(symbol, t.isoformat())
                close = round(base * (1 + 0.1 * (r.random() - 0.5)), 2)
                open_ = round(close * (1 + 0.004 * (r.random() - 0.5)), 2)
                bars.append(Bar(t, open_, max(open_, close) + 0.01, min(open_, close) - 0.01, close,
                                round(vol / 390 * minutes * (0.5 + r.random()))))
            t += step
        return bars

    def get_snapshots(self, symbols: Sequence[str]) -> dict[str, Snapshot]:
        now = self._now().astimezone(ET)
        out = {}
        for sym in symbols:
            if sym not in self._fixtures:
                continue
            sec, base, vol = self._fixtures[sym]
            r = _rng(sym, now.date())
            prev = self._cal.next_session(now.date() - dt.timedelta(days=1), include_today=False)
            prev_bars = self.get_daily_bars(sym, prev.date, prev.date) if prev.date < now.date() else []
            prev_close = prev_bars[-1].close if prev_bars else base
            pre = round(prev_close * (1 + 0.08 * (r.random() - 0.3)), 2)
            spread = 0.01 if vol > 1_000_000 else 0.08
            out[sym] = Snapshot(
                symbol=sym, as_of=now, source="mock",
                last_price=pre, bid=round(pre - spread / 2, 2), ask=round(pre + spread / 2, 2),
                prev_close=prev_close, premarket_price=pre,
                premarket_volume=round(vol * 0.05 * (1 + 5 * r.random())),
            )
        return out

    def get_news(self, symbols: Sequence[str], since: dt.datetime) -> list[NewsItem]:
        now = self._now().astimezone(ET)
        items = []
        for sym in symbols:
            if sym in _NEWS_TEMPLATES:
                headline, kind = _NEWS_TEMPLATES[sym]
                published = now.replace(hour=7, minute=5, second=0, microsecond=0)
                if published > now:
                    published -= dt.timedelta(days=1)
                if published >= since:
                    items.append(NewsItem(
                        id=f"mock-{sym}-{published.date()}", symbols=(sym,), headline=headline,
                        source="Mock Newswire", url=f"https://example.invalid/{sym}/{kind}",
                        published_at=published, provider="mock",
                    ))
        return sorted(items, key=lambda n: n.published_at, reverse=True)
