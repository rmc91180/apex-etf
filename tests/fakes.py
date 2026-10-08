"""A scriptable provider for engine tests."""
import datetime as dt

from apex_stocks.market_calendar import ET, MarketCalendar
from apex_stocks.models import Bar, NewsItem, Security, Snapshot

CAL = MarketCalendar()


def trending_bars(end_date: dt.date, n: int, start_price: float, daily_pct: float, volume: float,
                  include_end: bool = False) -> list[Bar]:
    """n daily bars ending before end_date (or on it), rising daily_pct per day with mild wiggle."""
    days, d = [], end_date if include_end else end_date - dt.timedelta(days=1)
    while len(days) < n:
        s = CAL.session(d)
        if s:
            days.append(s)
        d -= dt.timedelta(days=1)
    days.reverse()
    bars, price = [], start_price
    for i, s in enumerate(days):
        wiggle = 0.004 if i % 2 else -0.002
        close = price * (1 + daily_pct / 100 + wiggle)
        bars.append(Bar(s.open, price, max(price, close) * 1.01, min(price, close) * 0.99, close, volume))
        price = close
    return bars


class FakeProvider:
    name = "fake"
    is_live = True

    def __init__(self, now: dt.datetime):
        self.now = now
        self.securities: dict[str, Security] = {}
        self.bars: dict[str, list[Bar]] = {}
        self.snaps: dict[str, Snapshot] = {}
        self.news: list[NewsItem] = []
        self.calls: list[str] = []

    def add(self, sec: Security, bars: list[Bar], premarket: float | None = None, pm_volume: float = 0,
            spread: float = 0.01, age_minutes: float = 1):
        self.securities[sec.symbol] = sec
        self.bars[sec.symbol] = bars
        last = bars[-1].close
        price = premarket if premarket is not None else last
        self.snaps[sec.symbol] = Snapshot(
            sec.symbol, self.now - dt.timedelta(minutes=age_minutes), "fake", last_price=price,
            bid=round(price - spread / 2, 4), ask=round(price + spread / 2, 4), prev_close=last,
            premarket_price=price, premarket_volume=pm_volume)

    def add_news(self, symbol: str, headline: str, hours_ago: float = 2):
        t = self.now - dt.timedelta(hours=hours_ago)
        self.news.append(NewsItem(f"{symbol}-{len(self.news)}", (symbol,), headline, "Wire", "https://x.test", t))

    def list_universe(self):
        self.calls.append("universe")
        return list(self.securities.values())

    def get_snapshots(self, symbols, include_premarket=False):
        self.calls.append(f"snapshots:{include_premarket}")
        return {s: self.snaps[s] for s in symbols if s in self.snaps}

    def get_daily_bars(self, symbol, start, end):
        return [b for b in self.bars[symbol] if start <= b.start.astimezone(ET).date() <= end]

    def get_intraday_bars(self, symbol, start, end, minutes):
        return []

    def get_news(self, symbols, since):
        return [n for n in self.news if n.published_at >= since and set(n.symbols) & set(symbols)]
