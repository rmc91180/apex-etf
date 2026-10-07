import datetime as dt

import pytest

from apex_stocks.market_calendar import ET
from apex_stocks.providers import DataUnavailable, MarketDataProvider, MockProvider, get_provider

NOW = dt.datetime(2026, 10, 7, 8, 0, tzinfo=ET)


@pytest.fixture
def mock():
    return MockProvider(now=lambda: NOW)


def test_implements_interface_and_is_not_live(mock):
    assert isinstance(mock, MarketDataProvider)
    assert mock.is_live is False
    assert isinstance(get_provider("mock"), MockProvider)


def test_universe_includes_cases_the_filters_must_reject(mock):
    types = {s.symbol: s for s in mock.list_universe()}
    assert types["MKETF"].asset_type == "etf"
    assert types["MOCKW"].asset_type == "warrant"
    assert types["HALTD"].halted


def test_daily_bars_only_on_sessions_and_deterministic(mock):
    bars = mock.get_daily_bars("MOCKA", dt.date(2026, 11, 23), dt.date(2026, 11, 30))
    dates = [b.start.date() for b in bars]
    assert dt.date(2026, 11, 26) not in dates  # Thanksgiving
    assert dt.date(2026, 11, 28) not in dates  # Saturday
    assert len(bars) == 5
    assert bars == mock.get_daily_bars("MOCKA", dt.date(2026, 11, 23), dt.date(2026, 11, 30))
    for b in bars:
        assert b.low <= min(b.open, b.close) <= max(b.open, b.close) <= b.high


def test_intraday_bars_stay_inside_session(mock):
    start = dt.datetime(2026, 10, 7, 9, 0, tzinfo=ET)
    end = dt.datetime(2026, 10, 7, 10, 0, tzinfo=ET)
    bars = mock.get_intraday_bars("MOCKB", start, end, 5)
    assert len(bars) == 6 and bars[0].start.time() == dt.time(9, 30)


def test_snapshots_are_stamped_as_mock_and_skip_unknown(mock):
    snaps = mock.get_snapshots(["MOCKA", "NOPE"])
    assert list(snaps) == ["MOCKA"]
    s = snaps["MOCKA"]
    assert s.source == "mock" and s.as_of == NOW
    assert s.spread_percent is not None and s.spread_percent > 0


def test_unknown_symbol_raises(mock):
    with pytest.raises(DataUnavailable):
        mock.get_daily_bars("NOPE", dt.date(2026, 10, 1), dt.date(2026, 10, 7))


def test_news_respects_since(mock):
    items = mock.get_news(["MOCKA", "MOCKB"], since=NOW - dt.timedelta(hours=12))
    assert [n.symbols for n in items] == [("MOCKA",)]
    assert mock.get_news(["MOCKA"], since=NOW) == []
