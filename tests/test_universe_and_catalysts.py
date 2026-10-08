import datetime as dt

import pytest

from apex_stocks.catalysts import assess, classify
from apex_stocks.config import Config
from apex_stocks.market_calendar import ET
from apex_stocks.models import NewsItem, Security
from apex_stocks.universe import classify_asset, static_rejections

NOW = dt.datetime(2026, 10, 7, 8, 0, tzinfo=ET)


@pytest.mark.parametrize("sym,name,exch,kind", [
    ("ACME", "Acme Corp Common Stock", "NYSE", "common"),
    ("ACMEW", "Acme Corp", "NASDAQ", "warrant"),
    ("ACME.WS", "Acme Corp", "NYSE", "warrant"),
    ("ACMEU", "Acme Corp", "NASDAQ", "unit"),
    ("ACME.PRA", "Acme Corp 6.5% Series A Preferred", "NYSE", "preferred"),
    ("TQQQ", "ProShares UltraPro QQQ", "NASDAQ", "etf"),
    ("SPAK", "Spak Acquisition Corp", "NASDAQ", "spac"),
    ("BDCX", "Example Income Fund", "NYSE", "fund"),
])
def test_classify_asset(sym, name, exch, kind):
    assert classify_asset(sym, name, exch) == kind


def test_spacs_only_when_enabled():
    spac = Security("SPAK", "Spak Acquisition Corp", "NASDAQ", "spac")
    assert static_rejections(spac, Config())
    assert not static_rejections(spac, Config(ALLOW_SPACS=True))


def news(headline, hours_ago=2):
    return NewsItem("n", ("X",), headline, "Wire", "u", NOW - dt.timedelta(hours=hours_ago))


@pytest.mark.parametrize("headline,category,direction", [
    ("X receives FDA approval for lead drug", "fda_positive", 1),
    ("X receives complete response letter from FDA", "fda_negative", -1),
    ("X to be acquired by Y for $12 per share", "acquisition", 1),
    ("X announces $15 million registered direct offering", "offering", -1),
    ("X reports Q3 revenue that beats estimates", "beat", 1),
    ("X raises full-year guidance", "guidance_raise", 1),
    ("X lowers full-year guidance", "guidance_cut", -1),
    ("X awarded Navy contract", "contract", 1),
    ("X announces 1-for-20 reverse stock split", "reverse_split", -1),
    ("Analyst upgrades X to buy", "upgrade", 1),
    ("X to present at investor conference", None, None),
])
def test_classify_headlines(headline, category, direction):
    c = classify(news(headline), NOW)
    assert (c.category if c else None) == category
    if c:
        assert c.direction == direction


def test_catalyst_strength_decays_with_age():
    fresh = assess([news("X awarded Navy contract", 1)], NOW).score
    stale = assess([news("X awarded Navy contract", 30)], NOW).score
    assert fresh > stale > 0


def test_bearish_news_dampens_bullish():
    good = assess([news("X awarded Navy contract")], NOW).score
    mixed = assess([news("X awarded Navy contract"), news("X announces public offering")], NOW)
    assert mixed.score < good and mixed.negatives
