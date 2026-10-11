import datetime as dt

import numpy as np
import pytest

from apex_stocks import research
from apex_stocks.config import Config
from apex_stocks.history import DailyTable
from apex_stocks.market_calendar import MarketCalendar

CAL = MarketCalendar()
CFG = Config()


def table(sym_prices: dict[str, list[tuple]], start=dt.date(2025, 1, 2)) -> DailyTable:
    """sym -> list of (o, h, l, c, v), one per session starting at ``start``."""
    days = []
    d = start
    while len(days) < max(len(v) for v in sym_prices.values()):
        if CAL.session(d):
            days.append(d.toordinal())
        d += dt.timedelta(days=1)
    data = {}
    for s, rows in sym_prices.items():
        arr = np.array(rows, dtype=float)
        data[s] = {"d": np.array(days[:len(rows)], dtype=np.int32), "o": arr[:, 0], "h": arr[:, 1],
                   "l": arr[:, 2], "c": arr[:, 3], "v": arr[:, 4]}
    return DailyTable(data)


def flat(n, p=10.0, v=2_000_000):
    return [(p, p * 1.01, p * 0.99, p, v)] * n


def test_rsi_series_matches_scalar_indicator():
    from apex_stocks.indicators import rsi
    c = np.array([10, 10.5, 10.2, 9.8, 9.9, 10.4, 10.1, 9.7, 9.5, 9.9])
    s = research.rsi_series(c, 2)
    for i in range(3, len(c)):
        assert s[i] == pytest.approx(rsi(list(c[:i + 1]), 2))


def test_simulate_daily_time_exit_and_stops():
    a = {"o": np.array([10.0, 9.0]), "h": np.array([10.5, 9.5]), "l": np.array([9.8, 8.5]), "c": np.array([10.2, 9.4])}
    assert research.simulate_daily(a, 0, 1, 0.04, 0) == (pytest.approx(-0.06), "time")
    # 1.5 ATR stop at 9.40: day two gaps below it, so the exit is that open.
    assert research.simulate_daily(a, 0, 1, 0.04, 1.5) == (pytest.approx(-0.10), "stop")
    # Wide stop at 8.0 is never touched.
    assert research.simulate_daily(a, 0, 1, 0.04, 5) == (pytest.approx(-0.06), "time")


def drop_then_bounce():
    """80 quiet days, then a 12% drop, then a rebound the next session."""
    rows = flat(80)
    rows.append((10.0, 10.0, 8.7, 8.8, 4_000_000))          # drop day (feature day)
    rows.append((8.8, 9.6, 8.7, 9.5, 3_000_000))            # trade day
    rows += [(9.5, 9.7, 9.4, 9.6, 2_000_000)] * 5
    return rows


def test_reversal_candidate_uses_only_prior_close_and_buys_next_open():
    t = table({"DROP": drop_then_bounce(), "SPY": flat(87, 500)})
    sessions = [CAL.session(dt.date.fromordinal(int(x))) for x in t.data["DROP"]["d"]]
    trade_day = sessions[81].date
    days = [x for x in research.sessions(CAL, CFG, trade_day, trade_day)]
    cands = research.candidates(t, CFG, days, ["DROP"], progress=lambda *_: None)
    rev = [c for c in cands if c.strategy == "reversal"]
    assert len(rev) == 1 and rev[0].strength == pytest.approx(12.0)
    exit_day = days[0][2]
    exit_close = t.on("DROP", exit_day)["c"]
    assert rev[0].gross["open|0"] == pytest.approx(exit_close / 8.8 - 1)
    # Overnight plans buy the 8.8 drop-day close and sell the trade day's open or close.
    if (trade_day - days[0][1]).days <= 2:
        assert rev[0].gross["close>open"] == pytest.approx(0.0)
        assert rev[0].gross["close>close"] == pytest.approx(9.5 / 8.8 - 1)


def test_chase_rule_skips_gap_up_open_for_morning_plans_only():
    rows = drop_then_bounce()
    rows[81] = (9.2, 9.6, 9.1, 9.5, 3_000_000)   # opens 4.5% above the 8.8 close
    t = table({"DROP": rows})
    d = dt.date.fromordinal(int(t.data["DROP"]["d"][81]))
    days = research.sessions(CAL, CFG, d, d)
    cands = research.candidates(t, CFG, days, ["DROP"], progress=lambda *_: None)
    if (d - days[0][1]).days <= 2:
        assert cands and all(c.gross["open|0"] is None and c.gross["close>open"] is not None for c in cands)
    else:
        assert cands == []


def test_overnight_plans_skip_entries_more_than_48h_before_the_next_session():
    fri, mon = dt.date(2026, 3, 6), dt.date(2026, 3, 9)
    rows = flat(80)
    rows.append((10.0, 10.0, 8.7, 8.8, 4_000_000))
    rows.append((8.8, 9.6, 8.7, 9.5, 3_000_000))
    rows += [(9.5, 9.7, 9.4, 9.6, 2_000_000)] * 3
    # start so the drop lands on Friday 2026-03-06
    start = fri
    n = 80
    while n:
        start -= dt.timedelta(days=1)
        if CAL.session(start):
            n -= 1
    t = table({"DROP": rows}, start=start)
    assert dt.date.fromordinal(int(t.data["DROP"]["d"][80])) == fri
    days = research.sessions(CAL, CFG, mon, mon)
    cands = research.candidates(t, CFG, days, ["DROP"], progress=lambda *_: None)
    rev = [c for c in cands if c.strategy == "reversal"]
    assert rev and rev[0].gross["close>open"] is None and rev[0].gross["open|0"] is not None


def test_illiquid_and_expensive_stocks_are_not_candidates():
    thin = drop_then_bounce()
    thin = [(o, h, l, c, 1000) for o, h, l, c, _ in thin]
    pricey = [(o * 5, h * 5, l * 5, c * 5, v) for o, h, l, c, v in drop_then_bounce()]
    t = table({"THIN": thin, "PRICY": pricey})
    d = dt.date.fromordinal(int(t.data["THIN"]["d"][81]))
    assert research.candidates(t, CFG, research.sessions(CAL, CFG, d, d), None, progress=lambda *_: None) == []


def C(day, strat, sym, strength, ret):
    return research.Candidate(day, strat, sym, strength, {p: ret for p in research.PLANS})


def test_picks_take_strongest_and_respect_threshold_and_market_filter():
    d1, d2 = dt.date(2026, 3, 2), dt.date(2026, 3, 3)
    prev = {d1: dt.date(2026, 2, 27), d2: d1}
    cands = [C(d1, "reversal", "A", 7, 0.01), C(d1, "reversal", "B", 12, -0.02), C(d2, "reversal", "C", 8, 0.03)]
    k = research.ConfigKey("reversal", 6, "open|0", "any")
    assert [c.symbol for c in research.picks(cands, k, {}, prev)] == ["B", "C"]
    k10 = research.ConfigKey("reversal", 10, "open|0", "any")
    assert [c.symbol for c in research.picks(cands, k10, {}, prev)] == ["B"]
    up = research.ConfigKey("reversal", 6, "open|0", "spy_up")
    regime = {dt.date(2026, 2, 27).toordinal(): False, d1.toordinal(): True}
    assert [c.symbol for c in research.picks(cands, up, regime, prev)] == ["C"]


def test_walk_forward_is_out_of_sample_and_can_choose_no_trading():
    good = research.ConfigKey("reversal", 6, "open|0", "any")
    bad = research.ConfigKey("breakout", 1.5, "open|0", "any")
    days = []
    d = dt.date(2025, 1, 2)
    while d < dt.date(2025, 11, 1):
        if CAL.session(d):
            days.append(d)
        d += dt.timedelta(days=1)
    by = {good: [C(x, "reversal", "G", 7, 0.02 if i % 4 else -0.01) for i, x in enumerate(days)],
          bad: [C(x, "breakout", "B", 2, -0.01 if i % 4 else 0.005) for i, x in enumerate(days)]}
    months = sorted({(x.year, x.month) for x in days})
    wf = research.walk_forward(by, months, [good, bad])
    assert all(f["chosen"] == good.label() for f in wf["folds"])
    first_test = wf["folds"][0]["test_period"]
    assert first_test.startswith("2025-07")
    assert research.walk_forward({bad: by[bad]}, months, [bad])["out_of_sample"]["0.25%"] == {"trades": 0}


def test_evaluate_and_report_render():
    t = table({"DROP": drop_then_bounce(), "SPY": flat(87, 500)})
    first = dt.date.fromordinal(int(t.data["DROP"]["d"][70]))
    last = dt.date.fromordinal(int(t.data["DROP"]["d"][85]))
    days = research.sessions(CAL, CFG, first, last)
    cands = research.candidates(t, CFG, days, ["DROP"], progress=lambda *_: None)
    s = research.evaluate(cands, days, research.spy_regime(t), research.baseline(t, CFG, days, ["DROP"]))
    md = research.report_markdown(s)
    assert "Verdict" in md and "reversal" in md


def test_catalyst_candidates_use_only_news_before_the_scan():
    rows = flat(80) + [(10.0, 10.1, 9.9, 10.0, 2_000_000), (11.0, 11.8, 10.9, 11.5, 6_000_000)] + [(11.5, 11.7, 11.3, 11.6, 2e6)] * 4
    t = table({"NEWS": rows})
    d = dt.date.fromordinal(int(t.data["NEWS"]["d"][81]))
    days = research.sessions(CAL, CFG, d, d)
    scan = dt.datetime.combine(d, CFG.scan_time, research.ET)

    def item(when, headline):
        return {"id": headline, "t": when.astimezone(dt.timezone.utc).isoformat(), "s": ["NEWS"], "h": headline, "m": ""}

    def news(month, syms):
        assert syms == ["NEWS"]
        return [item(scan - dt.timedelta(hours=2), "NEWS receives FDA approval for lead drug"),
                item(scan + dt.timedelta(hours=1), "NEWS announces public offering")]   # after the scan: ignored

    cands = research.catalyst_candidates(t, CFG, days, ["NEWS"], news, progress=lambda *_: None)
    assert len(cands) == 1 and cands[0].strategy == "catalyst" and cands[0].strength == 100
    assert cands[0].gross["open|0"] == pytest.approx(t.on("NEWS", days[0][2])["c"] / 11.0 - 1)
    if CAL.session(d + dt.timedelta(days=1)):
        assert cands[0].gross["day>open"] == pytest.approx(11.5 / 11.5 - 1)


def test_catalyst_configs_use_their_own_plans():
    ks = [k for k in research.all_configs() if k.strategy == "catalyst"]
    assert {k.plan for k in ks} == {"open|0", "open|2.5", "day>open", "day>close"}
