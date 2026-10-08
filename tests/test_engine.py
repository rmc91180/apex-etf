import datetime as dt

import pytest

from apex_stocks import messages
from apex_stocks.config import Config
from apex_stocks.engine import select
from apex_stocks.market_calendar import ET
from apex_stocks.models import Security
from tests.fakes import FakeProvider, trending_bars

NOW = dt.datetime(2026, 10, 7, 8, 0, tzinfo=ET)  # Wednesday pre-market
DAY = NOW.date()
CFG = Config()


def sec(sym, **kw):
    base = dict(name=f"{sym} Inc", exchange="NASDAQ", asset_type="common", market_cap=900e6, float_shares=80e6)
    base.update(kw)
    return Security(sym, **base)


def strong(p: FakeProvider, sym="GOOD", headline="GOOD Inc wins $40 million government contract"):
    bars = trending_bars(DAY, 120, 6.0, 0.4, 3_000_000)
    last = bars[-1].close
    p.add(sec(sym), bars, premarket=round(last * 1.06, 2), pm_volume=900_000)
    if headline:
        p.add_news(sym, headline)
    return last


@pytest.fixture
def p():
    return FakeProvider(NOW)


def test_strong_catalyst_setup_is_recommended_with_a_full_plan(p):
    last = strong(p)
    r = select(p, CFG, NOW)
    assert r.status == "recommended", r.reason
    assert r.pick.symbol == "GOOD"
    plan = r.plan
    assert plan.reference_price == round(last * 1.06, 2)
    assert plan.entry_low < plan.reference_price < plan.entry_high < plan.max_entry
    assert plan.target_pct == pytest.approx(7, abs=0.2) and plan.stop_pct == pytest.approx(-4, abs=0.2)
    assert plan.planned_entry_at == dt.datetime(2026, 10, 7, 9, 30, tzinfo=ET)
    assert plan.planned_exit_at == dt.datetime(2026, 10, 8, 15, 30, tzinfo=ET)
    assert r.pick.features.catalyst_category == "contract"


def test_message_never_promises_a_rise(p):
    strong(p)
    text = messages.morning_signal(select(p, CFG, NOW), "America/New_York")
    assert "GOOD" in text and "Planned exit" in text and "not a guarantee" in text
    for banned in ("will rise", "will go up", "guaranteed", "sure thing"):
        assert banned not in text.lower()


def test_non_common_and_untradable_securities_are_excluded(p):
    bars = trending_bars(DAY, 120, 6.0, 0.4, 3_000_000)
    p.add(sec("ETFX", asset_type="etf"), bars, pm_volume=900_000)
    p.add(sec("OTCX", exchange="OTC"), bars, pm_volume=900_000)
    p.add(sec("HALT", halted=True), bars, pm_volume=900_000)
    r = select(p, CFG, NOW)
    assert r.status == "no_trade"
    assert {c.symbol for c in r.candidates} == set()


def test_price_at_or_above_20_is_rejected(p):
    bars = trending_bars(DAY, 120, 15.0, 0.4, 3_000_000)
    p.add(sec("PRICY"), bars, premarket=20.05, pm_volume=900_000)
    p.add_news("PRICY", "PRICY Inc wins contract")
    r = select(p, CFG, NOW)
    c = next(c for c in r.candidates if c.symbol == "PRICY")
    assert any("outside" in x for x in c.rejections)
    assert r.status == "no_trade"


def test_illiquid_stock_never_scored(p):
    bars = trending_bars(DAY, 120, 6.0, 0.4, 50_000)
    p.add(sec("THIN"), bars, premarket=bars[-1].close * 1.1, pm_volume=500_000)
    assert select(p, CFG, NOW).status == "no_trade"


@pytest.mark.parametrize("kind,setup,expect", [
    ("wide spread", dict(spread=0.5), "spread"),
    ("stale quote", dict(age_minutes=90), "stale quote"),
])
def test_quality_rejections(p, kind, setup, expect):
    bars = trending_bars(DAY, 120, 6.0, 0.4, 3_000_000)
    p.add(sec("BAD"), bars, premarket=bars[-1].close * 1.05, pm_volume=900_000, **setup)
    p.add_news("BAD", "BAD Inc wins contract")
    c = select(p, CFG, NOW).candidates[0]
    assert any(expect in x for x in c.rejections), c.rejections


def test_offering_news_rejects_even_with_good_news(p):
    strong(p)
    p.add_news("GOOD", "GOOD Inc announces $25 million registered direct offering", hours_ago=1)
    r = select(p, CFG, NOW)
    assert r.status == "no_trade"
    assert any("negative news (offering)" in x for x in r.candidates[0].rejections)


def test_runup_without_catalyst_is_rejected(p):
    bars = trending_bars(DAY, 120, 3.0, 0.2, 5_000_000)
    for i in range(5):  # a sharp 5-day run, each day under the split-suspect ratio
        b = bars[-5 + i]
        bars[-5 + i] = type(b)(b.start, b.open, b.close * 1.16, b.low, b.close * (1.13 ** (i + 1)), b.volume)
    p.add(sec("RUN"), bars, pm_volume=900_000)
    c = select(p, CFG, NOW).candidates[0]
    assert any("no new catalyst" in x for x in c.rejections), c.rejections


def test_split_like_jump_is_rejected(p):
    bars = trending_bars(DAY, 120, 6.0, 0.1, 3_000_000)
    b = bars[60]
    bars[60] = type(b)(b.start, b.open, b.close * 2.2, b.low, b.close * 2.1, b.volume)
    p.add(sec("SPLT"), bars, pm_volume=900_000)
    c = select(p, CFG, NOW).candidates[0]
    assert any("split" in x for x in c.rejections)


def test_no_catalyst_caps_the_score(p):
    strong(p, headline=None)
    cfg = Config(MIN_SIGNAL_SCORE=0, NO_CATALYST_SCORE_CAP=30)
    r = select(p, cfg, NOW)
    assert r.pick.score.signal <= 30 and r.pick.score.capped_by_no_catalyst


def test_no_trade_when_threshold_not_met(p):
    strong(p)
    r = select(p, Config(MIN_SIGNAL_SCORE=99), NOW)
    assert r.status == "no_trade" and r.pick is None and "GOOD" in r.reason
    assert "NO TRADE TODAY" in messages.no_trade(r)


def test_low_confidence_blocks_high_signal(p):
    bars = trending_bars(DAY, 120, 6.0, 0.4, 3_000_000)
    p.add(sec("THINPM", market_cap=None, float_shares=None), bars, premarket=bars[-1].close * 1.06,
          pm_volume=2_000, age_minutes=20)
    p.add_news("THINPM", "THINPM Inc wins contract")
    scored = select(p, Config(MIN_SIGNAL_SCORE=0, MIN_CONFIDENCE_SCORE=0), NOW).pick
    assert scored.score.confidence < 50
    assert "thin pre-market trading" in " ".join(scored.score.quality_issues)
    r = select(p, Config(MIN_SIGNAL_SCORE=0), NOW)
    assert r.status == "no_trade"


def test_skips_closed_days_and_after_open(p):
    strong(p)
    assert select(p, CFG, dt.datetime(2026, 10, 10, 8, 0, tzinfo=ET)).status == "skipped"
    assert select(p, CFG, dt.datetime(2026, 10, 7, 9, 45, tzinfo=ET)).status == "skipped"
    assert p.calls == []


def test_trade_date_bar_is_never_used(p):
    """Look-ahead guard: a bar dated on the trade date must be ignored."""
    bars = trending_bars(DAY, 120, 6.0, 0.4, 3_000_000, include_end=True)
    p.add(sec("PEEK"), bars, pm_volume=900_000)
    p.add_news("PEEK", "PEEK Inc wins contract")
    r = select(p, Config(MIN_SIGNAL_SCORE=0), NOW)
    assert r.pick.features.prev_close == bars[-2].close


def test_position_size_respects_risk_and_liquidity(p):
    strong(p)
    r = select(p, Config(ACCOUNT_SIZE=10_000, MAX_POSITION_RISK_PERCENT=1), NOW)
    plan = r.plan
    per_share = plan.reference_price - plan.stop
    assert plan.max_dollar_risk == 100
    assert plan.max_shares * per_share <= 100 + per_share
    assert plan.max_shares * plan.reference_price <= 10_000


def test_friday_pick_exits_same_day(p):
    fri = dt.datetime(2026, 10, 9, 8, 0, tzinfo=ET)
    p.now = fri
    bars = trending_bars(fri.date(), 120, 6.0, 0.4, 3_000_000)
    p.add(sec("FRI"), bars, premarket=bars[-1].close * 1.05, pm_volume=900_000)
    p.add_news("FRI", "FRI Inc wins contract")
    r = select(p, Config(MIN_SIGNAL_SCORE=0), fri)
    assert r.plan.planned_exit_at == dt.datetime(2026, 10, 9, 15, 30, tzinfo=ET)
