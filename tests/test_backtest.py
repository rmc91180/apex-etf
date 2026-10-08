import dataclasses
import datetime as dt

import pytest

from apex_stocks import backtest
from apex_stocks.backtest import DayRecord, Trade, metrics, simulate, walk_forward
from apex_stocks.config import Config
from apex_stocks.history import HistoryCache
from apex_stocks.market_calendar import ET, MarketCalendar
from apex_stocks.models import NewsItem, Security
from apex_stocks.providers.replay import ReplayProvider

CAL = MarketCalendar()
D = dt.date(2026, 10, 7)


@dataclasses.dataclass
class P:
    symbol: str = "X"
    max_entry: float = 10.2
    stop: float = 9.6
    target: float = 10.7
    planned_entry_at: dt.datetime = dt.datetime(2026, 10, 7, 9, 30, tzinfo=ET)
    planned_exit_at: dt.datetime = dt.datetime(2026, 10, 8, 15, 30, tzinfo=ET)


def bar(t: dt.datetime, o, h, l, c, v=1000):
    return {"t": t.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"), "o": o, "h": h, "l": l, "c": c, "v": v}


def path(start: dt.datetime, prices: list[tuple]):
    return [bar(start + dt.timedelta(minutes=5 * i), *p) for i, p in enumerate(prices)]


close = lambda d: CAL.session(d).close
OPEN = dt.datetime(2026, 10, 7, 9, 30, tzinfo=ET)


def test_target_hit():
    bars = path(OPEN, [(10, 10.2, 9.9, 10.1), (10.1, 10.8, 10.1, 10.7)])
    t, why = simulate(P(), 10.0, bars, close)
    assert why is None and t.exit_reason == "target" and t.exit == 10.7
    assert t.net_return(0) == pytest.approx(0.07)


def test_stop_wins_when_both_hit_in_one_bar():
    t, _ = simulate(P(), 10.0, path(OPEN, [(10, 10.9, 9.5, 10.0)]), close)
    assert t.exit_reason == "stop" and t.exit == 9.6


def test_overnight_gap_through_stop_exits_at_open():
    day2 = dt.datetime(2026, 10, 8, 9, 30, tzinfo=ET)
    bars = path(OPEN, [(10, 10.1, 9.9, 10.0)]) + path(day2, [(9.2, 9.3, 9.1, 9.2)])
    t, _ = simulate(P(), 10.0, bars, close)
    assert t.exit_reason == "stop" and t.exit == 9.2


def test_time_exit_uses_bar_ending_at_exit():
    day2 = dt.datetime(2026, 10, 8, 15, 20, tzinfo=ET)
    bars = path(OPEN, [(10, 10.1, 9.9, 10.0)]) + path(day2, [(10.1, 10.2, 10.0, 10.15), (10.15, 10.3, 10.1, 10.25),
                                                            (10.25, 10.3, 10.2, 10.3)])
    t, _ = simulate(P(), 10.0, bars, close)
    assert t.exit_reason == "time" and t.exit == 10.25
    assert t.exit_at == dt.datetime(2026, 10, 8, 15, 30, tzinfo=ET)


def test_untradable_opens_are_skipped():
    assert simulate(P(), 10.5, [], close) == (None, "opened at $10.50, above the $10.20 max entry")
    assert simulate(P(), 9.5, [], close)[0] is None


def test_slippage_reduces_return():
    t = Trade("X", 10, 10.7, "target", OPEN, 7, 0, 10)
    assert t.net_return(0.0025) < t.net_return(0) and t.net_return(0.0025) == pytest.approx(10.7 * 0.9975 / 10.025 - 1)


def test_metrics():
    m = metrics([0.05, -0.02, 0.03, -0.04])
    assert m["trades"] == 4 and m["win_rate"] == 50.0
    assert m["avg_return_pct"] == pytest.approx(0.5)
    assert m["profit_factor"] == pytest.approx(0.08 / 0.06, abs=0.01)
    assert m["max_drawdown_pct"] == pytest.approx(-4.0)
    assert metrics([]) == {"trades": 0}


def records(win_when_signal_at_least: float | None):
    out, d = [], dt.date(2025, 1, 2)
    i = 0
    while d < dt.date(2025, 11, 1):
        if CAL.is_trading_day(d):
            sig = 50 + (i * 7) % 45
            win = win_when_signal_at_least is not None and sig >= win_when_signal_at_least
            ret = 1.04 + (i % 3) * 0.005 if win else 0.97 - (i % 3) * 0.004
            t = Trade("X", 10.0, 10.0 * ret, "time", OPEN, 0, 0, 100)
            out.append(DayRecord(d, "recommended", "", 5, "X", sig, 80, None, "uptrend", t))
            i += 1
        d += dt.timedelta(days=1)
    return out


def test_walk_forward_learns_threshold_and_scores_out_of_sample():
    wf = walk_forward(records(70))
    assert wf["folds"] and all(f["chosen"] != "no trading" for f in wf["folds"])
    assert all(f["chosen"]["min_signal"] >= 70 for f in wf["folds"])
    assert wf["out_of_sample"]["0.25%"]["avg_return_pct"] > 0


def test_walk_forward_stays_out_when_nothing_works():
    wf = walk_forward(records(None))
    assert all(f["chosen"] == "no trading" for f in wf["folds"])
    assert wf["out_of_sample"]["0.25%"] == {"trades": 0}
    assert "no out-of-sample trades" in backtest.verdict(wf["out_of_sample"])


# ── replay and an end-to-end run on synthetic history ──────────

class FakeSource:
    """Generates deterministic Alpaca-shaped history: GOOD trends up and has news; FLAT does nothing."""

    def __init__(self):
        self.news_requests = []

    def list_universe(self, status="active"):
        if status != "active":
            return [Security("GONE", "Gone Corp", "NASDAQ", "common")]
        return [Security("GOOD", "Good Corp", "NASDAQ", "common"),
                Security("FLAT", "Flat Corp", "NYSE", "common"),
                Security("SPY", "SPDR S&P 500 ETF Trust", "ARCA", "etf")]

    def _price(self, sym, d):
        n = (d - dt.date(2025, 1, 1)).days
        return {"GOOD": 5 + n * 0.01, "FLAT": 8.0, "SPY": 500 + n * 0.1, "GONE": 3.0}[sym]

    def bars_raw(self, symbols, timeframe, start, end, adjustment="raw"):
        out = {}
        for s in symbols:
            bars = []
            if timeframe == "1Day":
                d = start.date()
                while d <= end.date():
                    if CAL.is_trading_day(d):
                        p = self._price(s, d)
                        bars.append(bar(dt.datetime.combine(d, dt.time(0), ET), p, p * 1.02, p * 0.98,
                                        p * 1.005, 4_000_000))
                    d += dt.timedelta(days=1)
            else:
                t = start
                while t < end:
                    sess = CAL.session(t.date())
                    if sess and t.time() >= dt.time(4) and t < sess.close:
                        p = self._price(s, t.date())
                        pre = t < sess.open
                        if s == "GOOD":
                            p = p * (1.04 if pre else 1.04 + 0.002 * ((t - sess.open).seconds // 300))
                        bars.append(bar(t, p, p * 1.001, p * 0.999, p, 60_000 if pre else 50_000))
                    t += dt.timedelta(minutes=5)
            out[s] = bars
        return out

    def get_news(self, symbols, since, until):
        self.news_requests.append((tuple(symbols), since, until))
        items = []
        if "GOOD" in symbols:
            day = until.date()
            items.append(NewsItem(f"n-{day}", ("GOOD",), "Good Corp wins multi-year contract", "Wire", "u",
                                  dt.datetime.combine(day, dt.time(7, 0), ET)))
            items.append(NewsItem(f"late-{day}", ("GOOD",), "Good Corp announces public offering", "Wire", "u",
                                  dt.datetime.combine(day, dt.time(9, 15), ET)))  # after the scan
        return items


@pytest.fixture
def cache(tmp_path):
    return HistoryCache(tmp_path / "cache", FakeSource())


def test_replay_only_sees_the_past(cache):
    cfg = Config()
    now = dt.datetime(2025, 10, 7, 8, 45, tzinfo=ET)
    daily = cache.daily(["GOOD", "FLAT"], dt.date(2024, 11, 1), dt.date(2025, 10, 20), 19.99, 10e6, ("SPY",))
    p = ReplayProvider(cache, cache.universe(), daily, now)
    bars = p.get_daily_bars_many(["GOOD"], dt.date(2025, 9, 1), dt.date(2025, 10, 20))["GOOD"]
    assert bars[-1].start.date() == dt.date(2025, 10, 6)
    snap = p.get_snapshots(["GOOD"], include_premarket=True)["GOOD"]
    assert snap.as_of <= now - dt.timedelta(minutes=16)
    assert snap.prev_close == pytest.approx(FakeSource()._price("GOOD", dt.date(2025, 10, 6)) * 1.005)
    news = p.get_news(["GOOD"], now - dt.timedelta(hours=20))
    assert [n.headline for n in news] == ["Good Corp wins multi-year contract"]


def test_cache_is_reused_without_a_source(tmp_path):
    src_cache = HistoryCache(tmp_path / "c", FakeSource())
    src_cache.universe()
    assert {s.symbol for s in HistoryCache(tmp_path / "c").universe()} >= {"GOOD", "GONE"}
    with pytest.raises(RuntimeError):
        HistoryCache(tmp_path / "c").news(D, ["GOOD"], OPEN, OPEN)


def test_end_to_end_run_and_report(cache):
    cfg = Config()
    recs = backtest.run(cache, cfg, dt.date(2025, 9, 1), dt.date(2025, 9, 30), progress=lambda _: None)
    assert len(recs) == 21
    picked = [r for r in recs if r.symbol]
    assert picked and all(r.symbol == "GOOD" for r in picked)
    assert any(r.trade and r.trade.exit_reason == "target" for r in picked)
    assert all(r.regime == "uptrend" for r in recs)
    s = backtest.summarize(recs, cfg)
    md = backtest.report_markdown(s)
    assert "Verdict" in md and "| Every top pick" in md
