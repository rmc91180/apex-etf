"""Plain data types shared by providers, storage and the strategy."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str
    exchange: str            # NYSE, NASDAQ, AMEX (NYSE American), ARCA, OTC ...
    asset_type: str          # common, etf, etn, fund, preferred, warrant, right, unit, spac, other
    tradable: bool = True
    halted: bool = False
    market_cap: float | None = None
    float_shares: float | None = None
    shares_outstanding: float | None = None
    sector: str | None = None
    industry: str | None = None


@dataclass(frozen=True)
class Bar:
    start: dt.datetime       # tz-aware bar start
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float | None = None


@dataclass(frozen=True)
class Snapshot:
    """Latest market state for one symbol, as reported by a provider."""
    symbol: str
    as_of: dt.datetime       # tz-aware time the provider stamped the data
    source: str
    last_price: float | None = None
    bid: float | None = None
    ask: float | None = None
    prev_close: float | None = None
    day_open: float | None = None
    day_volume: float | None = None
    vwap: float | None = None
    premarket_price: float | None = None
    premarket_volume: float | None = None

    @property
    def spread_percent(self) -> float | None:
        if not self.bid or not self.ask or self.ask < self.bid:
            return None
        mid = (self.bid + self.ask) / 2
        return (self.ask - self.bid) / mid * 100


@dataclass(frozen=True)
class NewsItem:
    id: str
    symbols: tuple[str, ...]
    headline: str
    source: str              # publisher
    url: str
    published_at: dt.datetime  # tz-aware
    summary: str = ""
    provider: str = ""


@dataclass
class DataQuality:
    """Problems found while gathering data. Never filled with guesses."""
    issues: list[str] = field(default_factory=list)

    def add(self, issue: str) -> None:
        self.issues.append(issue)

    @property
    def ok(self) -> bool:
        return not self.issues
