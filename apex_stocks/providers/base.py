"""The market-data provider interface.

The strategy only talks to this interface, so a provider (Alpaca, Polygon,
...) can be swapped without touching scoring code. Providers must raise
``DataUnavailable`` rather than return invented or default values.
"""
from __future__ import annotations

import datetime as dt
from typing import Protocol, Sequence, runtime_checkable

from ..models import Bar, NewsItem, Security, Snapshot


class DataUnavailable(RuntimeError):
    pass


@runtime_checkable
class MarketDataProvider(Protocol):
    name: str
    # False for synthetic data. Live notifications must refuse a non-live provider.
    is_live: bool

    def list_universe(self) -> list[Security]: ...

    def get_snapshots(self, symbols: Sequence[str], include_premarket: bool = False) -> dict[str, Snapshot]:
        """Latest snapshot per symbol. Symbols with no data are omitted.

        With include_premarket, also fill premarket_price and premarket_volume
        (this can be expensive, so the engine asks only for a short list).
        """
        ...

    def get_daily_bars(self, symbol: str, start: dt.date, end: dt.date) -> list[Bar]:
        """Daily bars from start to end inclusive, oldest first."""
        ...

    def get_intraday_bars(
        self, symbol: str, start: dt.datetime, end: dt.datetime, minutes: int
    ) -> list[Bar]:
        """1-, 5- or 15-minute bars in [start, end), oldest first."""
        ...

    def get_news(self, symbols: Sequence[str], since: dt.datetime) -> list[NewsItem]:
        """Company news published at or after ``since``, newest first."""
        ...
