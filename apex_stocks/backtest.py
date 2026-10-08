"""Walk-forward backtest of the daily pick, with slippage sensitivity.

For every trading day in the period the real selection engine runs against a
replay of what was knowable at the scan time. The top-ranked candidate's
trade is simulated on 5-minute bars: buy at the open (skipped if it opens
above the max entry or below the stop), exit at the target, the stop or the
planned exit time, whichever comes first. If the target and the stop fall in
the same 5-minute bar, the stop is assumed (the conservative choice).

Score thresholds are then chosen on each training window and applied to the
following unseen test window, so the headline result is out-of-sample.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .config import Config
from .engine import select
from .history import DailyTable, HistoryCache, eligible_symbols
from .market_calendar import ET, MarketCalendar
from .providers.replay import ReplayProvider

SLIPPAGES = (0.0, 0.001, 0.0025, 0.005, 0.01)   # per side
DEFAULT_SLIPPAGE = 0.0025
SIGNAL_GRID = (50, 55, 60, 65, 70, 75, 80)
CONFIDENCE_GRID = (40, 50, 60, 70)


@dataclass
class Trade:
    symbol: str
    entry: float
    exit: float
    exit_reason: str          # target, stop, time
    exit_at: dt.datetime
    mfe_pct: float
    mae_pct: float
    holding_minutes: int

    def net_return(self, slip: float) -> float:
        return (self.exit * (1 - slip)) / (self.entry * (1 + slip)) - 1


@dataclass
class DayRecord:
    day: dt.date
    status: str
    reason: str
    scored: int = 0
    symbol: str | None = None
    signal: float | None = None
    confidence: float | None = None
    catalyst: str | None = None
    regime: str | None = None
    trade: Trade | None = None
    skip_reason: str | None = None         # why the top pick could not be traded
    baseline_returns: list[float] = field(default_factory=list)  # all scored candidates, time exit


def _ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ET)


def simulate(plan, open_price: float, bars: list[dict], session_close: Callable[[dt.date], dt.datetime]) -> tuple[Trade | None, str | None]:
    """Simulate one plan. ``bars`` are raw 5-minute bars from the entry open onward."""
    if open_price > plan.max_entry:
        return None, f"opened at ${open_price:.2f}, above the ${plan.max_entry:.2f} max entry"
    if open_price <= plan.stop:
        return None, f"opened at ${open_price:.2f}, at or below the ${plan.stop:.2f} stop"
    entry = open_price
    hi, lo = entry, entry
    exit_price, reason, exit_at = None, None, None
    last = None
    for b in bars:
        t = _ts(b["t"])
        if t >= plan.planned_exit_at:
            break
        s_close = session_close(t.date())
        if t + dt.timedelta(minutes=5) > s_close or t.time() < dt.time(9, 30):
            continue  # regular session only
        last = (b, t)
        if b["l"] <= plan.stop:
            exit_price, reason, exit_at = min(plan.stop, b["o"]), "stop", t
            lo = min(lo, b["l"])
            break
        if b["h"] >= plan.target:
            exit_price, reason, exit_at = max(plan.target, b["o"]), "target", t
            hi = max(hi, b["h"])
            break
        hi, lo = max(hi, b["h"]), min(lo, b["l"])
    if exit_price is None:
        if last is None:
            return None, "no intraday data"
        b, t = last
        exit_price, reason, exit_at = b["c"], "time", t + dt.timedelta(minutes=5)
    return Trade(plan.symbol, entry, exit_price, reason, exit_at,
                 round((hi / entry - 1) * 100, 3), round((lo / entry - 1) * 100, 3),
                 int((exit_at - plan.planned_entry_at).total_seconds() // 60)), None


# ── metrics ────────────────────────────────────────────────────

def metrics(returns: Sequence[float]) -> dict:
    n = len(returns)
    if n == 0:
        return {"trades": 0}
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    curve, peak, dd = 0.0, 0.0, 0.0
    for r in returns:
        curve += r
        peak = max(peak, curve)
        dd = min(dd, curve - peak)
    sd = statistics.pstdev(returns) if n > 1 else 0.0
    mean = sum(returns) / n
    return {
        "trades": n,
        "win_rate": round(len(wins) / n * 100, 1),
        "avg_return_pct": round(mean * 100, 3),
        "median_return_pct": round(statistics.median(returns) * 100, 3),
        "avg_win_pct": round(sum(wins) / len(wins) * 100, 3) if wins else None,
        "avg_loss_pct": round(sum(losses) / len(losses) * 100, 3) if losses else None,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "total_return_pct": round(sum(returns) * 100, 2),
        "max_drawdown_pct": round(dd * 100, 2),
        "sharpe_approx": round(mean / sd * math.sqrt(252), 2) if sd > 0 else None,
        "t_stat": round(mean / (sd / math.sqrt(n)), 2) if sd > 0 and n > 1 else None,
        "best_pct": round(max(returns) * 100, 2),
        "worst_pct": round(min(returns) * 100, 2),
    }


def traded(records: Sequence[DayRecord], min_signal: float, min_conf: float) -> list[DayRecord]:
    return [r for r in records if r.trade and r.signal >= min_signal and r.confidence >= min_conf]


def _objective(records, s, c, slip) -> float:
    rs = [r.trade.net_return(slip) for r in traded(records, s, c)]
    if len(rs) < 10:
        return -math.inf
    sd = statistics.pstdev(rs)
    return (sum(rs) / len(rs)) / (sd / math.sqrt(len(rs))) if sd > 0 else -math.inf


def walk_forward(records: Sequence[DayRecord], train_months: int = 6, test_months: int = 2,
                 slip: float = DEFAULT_SLIPPAGE) -> dict:
    """Pick thresholds on each training window, apply them to the next test window."""
    months = sorted({(r.day.year, r.day.month) for r in records})
    folds, oos = [], []
    i = 0
    while i + train_months < len(months):
        train_m = set(months[i:i + train_months])
        test_m = set(months[i + train_months:i + train_months + test_months])
        train = [r for r in records if (r.day.year, r.day.month) in train_m]
        test = [r for r in records if (r.day.year, r.day.month) in test_m]
        best, best_obj = None, 0.0  # must beat zero: otherwise the fold stays out of the market
        for s in SIGNAL_GRID:
            for c in CONFIDENCE_GRID:
                obj = _objective(train, s, c, slip)
                if obj > best_obj:
                    best, best_obj = (s, c), obj
        test_trades = traded(test, *best) if best else []
        rs = [r.trade.net_return(slip) for r in test_trades]
        oos.extend(test_trades)
        folds.append({
            "train": f"{min(train_m)[0]}-{min(train_m)[1]:02d}..{max(train_m)[0]}-{max(train_m)[1]:02d}",
            "test_period": f"{min(test_m)[0]}-{min(test_m)[1]:02d}..{max(test_m)[0]}-{max(test_m)[1]:02d}",
            "chosen": {"min_signal": best[0], "min_confidence": best[1]} if best else "no trading",
            "train_t_stat": round(best_obj, 2) if best else None,
            "test": metrics(rs),
        })
        i += test_months
    return {"folds": folds, "out_of_sample": {f"{s * 100:g}%": metrics([r.trade.net_return(s) for r in oos])
                                              for s in SLIPPAGES}}


# ── the run ────────────────────────────────────────────────────

def regime(daily: DailyTable, day: dt.date) -> str | None:
    w = daily.window("SPY", day - dt.timedelta(days=90), day - dt.timedelta(days=1))
    if w is None or len(w["c"]) < 50:
        return None
    return "uptrend" if w["c"][-1] > w["c"][-50:].mean() else "downtrend"


def run(cache: HistoryCache, cfg: Config, start: dt.date, end: dt.date,
        progress: Callable[[str], None] = print) -> list[DayRecord]:
    cal = MarketCalendar()
    scan_cfg = dataclasses.replace(cfg, MIN_SIGNAL_SCORE=0.0, MIN_CONFIDENCE_SCORE=0.0)
    universe = cache.universe()
    elig = eligible_symbols(universe, cfg)
    progress(f"universe {len(universe)}, eligible {len(elig)}; loading daily bars")
    daily = cache.daily(elig, start - dt.timedelta(days=330), end + dt.timedelta(days=7),
                        cfg.MAX_STOCK_PRICE, cfg.MIN_AVG_DOLLAR_VOLUME, always_keep=("SPY",))
    progress(f"daily bars kept for {len(daily.symbols())} symbols")

    records: list[DayRecord] = []
    d = start
    while d <= end:
        session = cal.session(d)
        if session is None:
            d += dt.timedelta(days=1)
            continue
        now = dt.datetime.combine(d, cfg.scan_time, ET)
        prov = ReplayProvider(cache, universe, daily, now, cal)
        r = select(prov, scan_cfg, now, cal)
        rec = DayRecord(d, r.status, r.reason, len(r.ranked), regime=regime(daily, d))
        if r.pick:
            p = r.pick
            rec.symbol, rec.signal, rec.confidence = p.symbol, p.score.signal, p.score.confidence
            rec.catalyst = p.features.catalyst_category
            today = daily.on(p.symbol, d)
            if today is None:
                rec.skip_reason = "no bar on the trade date (halted or delisted)"
            else:
                bars = cache.intraday(p.symbol, session.open, r.plan.planned_exit_at)
                rec.trade, rec.skip_reason = simulate(r.plan, today["o"], bars,
                                                      lambda day: cal.session(day).close)
        # Baseline: every scored candidate held open-to-close of the exit day (time exit only).
        exit_day = cal.plan_exit(session.open, cfg).date()
        for c in r.ranked:
            o, x = daily.on(c.symbol, d), daily.on(c.symbol, exit_day)
            if o and x and o["o"] > 0:
                rec.baseline_returns.append(x["c"] / o["o"] - 1)
        records.append(rec)
        t = rec.trade
        progress(f"{d} {rec.status:11} {rec.symbol or '-':6} sig {rec.signal or 0:5.1f} conf {rec.confidence or 0:5.1f} "
                 + (f"{t.exit_reason:6} {t.net_return(0) * 100:+6.2f}%" if t else (rec.skip_reason or rec.reason)[:60]))
        d += dt.timedelta(days=1)
    return records


def summarize(records: Sequence[DayRecord], cfg: Config) -> dict:
    slip = DEFAULT_SLIPPAGE
    at_cfg = traded(records, cfg.MIN_SIGNAL_SCORE, cfg.MIN_CONFIDENCE_SCORE)
    all_top = traded(records, 0, 0)
    base = [x for r in records for x in r.baseline_returns]
    buckets = {}
    for lo in range(40, 100, 10):
        rs = [r.trade.net_return(slip) for r in all_top if lo <= r.signal < lo + 10]
        if rs:
            buckets[f"{lo}-{lo + 9}"] = metrics(rs)
    by_regime = {g: metrics([r.trade.net_return(slip) for r in at_cfg if r.regime == g])
                 for g in ("uptrend", "downtrend")}
    monthly = {}
    for r in at_cfg:
        monthly.setdefault(f"{r.day:%Y-%m}", []).append(r.trade.net_return(slip))
    exits = {}
    for r in at_cfg:
        exits[r.trade.exit_reason] = exits.get(r.trade.exit_reason, 0) + 1
    wf = walk_forward(records)
    oos = wf["out_of_sample"]
    return {
        "period": f"{records[0].day}..{records[-1].day}" if records else None,
        "trading_days": len(records),
        "days_with_a_top_candidate": sum(1 for r in records if r.symbol),
        "untradable_top_picks": sum(1 for r in records if r.symbol and not r.trade),
        "config_thresholds": {"min_signal": cfg.MIN_SIGNAL_SCORE, "min_confidence": cfg.MIN_CONFIDENCE_SCORE},
        "at_config_thresholds": {f"{s * 100:g}%": metrics([r.trade.net_return(s) for r in at_cfg]) for s in SLIPPAGES},
        "every_top_pick": metrics([r.trade.net_return(slip) for r in all_top]),
        "baseline_all_scored_candidates_time_exit": metrics(base),
        "exit_reasons": exits,
        "by_signal_score": buckets,
        "by_market_regime": by_regime,
        "monthly": {m: metrics(v) for m, v in sorted(monthly.items())},
        "walk_forward": wf,
        "verdict": verdict(oos),
    }


def verdict(oos: dict) -> str:
    """Plain-language read of the out-of-sample result at each slippage level."""
    lines = []
    for label, m in oos.items():
        if not m.get("trades"):
            lines.append(f"{label} slippage: no out-of-sample trades")
            continue
        t = m.get("t_stat") or 0
        status = ("positive and statistically meaningful" if m["avg_return_pct"] > 0 and t >= 2
                  else "positive but could be luck" if m["avg_return_pct"] > 0
                  else "no edge (average trade loses)")
        lines.append(f"{label} slippage: {status} (avg {m['avg_return_pct']:+.2f}%/trade, "
                     f"t={t}, {m['trades']} trades)")
    return "\n".join(lines)


def report_markdown(s: dict) -> str:
    def row(name, m):
        if not m.get("trades"):
            return f"| {name} | 0 | | | | | | |"
        return (f"| {name} | {m['trades']} | {m['win_rate']}% | {m['avg_return_pct']:+.2f}% | "
                f"{m['median_return_pct']:+.2f}% | {m['profit_factor']} | {m['max_drawdown_pct']}% | {m['t_stat']} |")

    head = "| | Trades | Win rate | Avg | Median | Profit factor | Max DD (pts) | t-stat |\n|---|---|---|---|---|---|---|---|"
    out = [f"# APEX Stocks backtest {s['period']}", "",
           f"{s['trading_days']} trading days; a top candidate existed on {s['days_with_a_top_candidate']}; "
           f"{s['untradable_top_picks']} of those could not be entered (gap past max entry or below stop, or no data).", "",
           "## Verdict (walk-forward, out-of-sample)", "", "```", s["verdict"], "```", "",
           "## Out-of-sample by slippage (per side)", "", head]
    out += [row(k, v) for k, v in s["walk_forward"]["out_of_sample"].items()]
    out += ["", f"## Current thresholds (signal ≥ {s['config_thresholds']['min_signal']}, confidence ≥ "
                f"{s['config_thresholds']['min_confidence']}), in-sample", "", head]
    out += [row(k, v) for k, v in s["at_config_thresholds"].items()]
    out += ["", "## Comparisons at 0.25% slippage", "", head,
            row("Every top pick (no thresholds)", s["every_top_pick"]),
            row("Baseline: all scored candidates, time exit, no slippage", s["baseline_all_scored_candidates_time_exit"])]
    out += ["", "## By signal score (every top pick, 0.25% slippage)", "", head]
    out += [row(k, v) for k, v in s["by_signal_score"].items()]
    out += ["", "## By market regime (SPY vs its 50-day average)", "", head]
    out += [row(k, v) for k, v in s["by_market_regime"].items()]
    out += ["", "## Monthly (current thresholds, 0.25% slippage)", "", head]
    out += [row(k, v) for k, v in s["monthly"].items()]
    out += ["", "## Walk-forward folds", "", "| Train | Test | Chosen | Test trades | Test avg | Test t |", "|---|---|---|---|---|---|"]
    for f in s["walk_forward"]["folds"]:
        m = f["test"]
        out.append(f"| {f['train']} | {f['test_period']} | {f['chosen']} | {m.get('trades', 0)} | "
                   f"{m.get('avg_return_pct', '')} | {m.get('t_stat', '')} |")
    out += ["", f"Exit reasons at current thresholds: {s['exit_reasons']}", "",
            "Notes: returns are per trade as a percent of position size; drawdown is in cumulative percentage points. "
            "Sharpe and t-stat treat trades as independent daily observations. Daily bars are unadjusted, so stocks with "
            "a split in their lookback are rejected, as live. Delisted symbols are included where Alpaca still lists them."]
    return "\n".join(out)
