"""Signal research: test alternative daily-pick ideas on cached daily bars.

The first scoring model (engine.py) showed no edge in the walk-forward
backtest, so this module screens new ideas quickly and honestly before any of
them is wired into the live engine.

Each idea is a rule on data known at the previous close. On trade day D the
single strongest candidate is bought at D's open and sold at the planned exit
(the close of the exit session, a stand-in for 3:30 PM ET), unless a
volatility stop is hit first. Stops are checked on daily lows; a gap through
the stop exits at that day's open. The intraday simulator in backtest.py
validates whichever idea survives here.

Every idea × threshold × stop × market filter is a "config". Configs are
chosen on 6 months and tested on the next 2, so the headline numbers are
out-of-sample, including the choice of idea itself.
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from .backtest import DEFAULT_SLIPPAGE, SLIPPAGES, metrics
from .config import Config
from .history import DailyTable, HistoryCache, eligible_symbols
from .market_calendar import ET, MarketCalendar

# Strategy thresholds: a candidate qualifies when its "strength" is at least the threshold.
STRATEGIES = {
    # Short-term pullback inside an uptrend: close above its 100-day average, 2-day RSI washed out.
    "pullback": {"label": "Pullback in an uptrend (2-day RSI)", "thresholds": (80, 90, 95)},
    # Sharp one-day drop: bet on a partial rebound.
    "reversal": {"label": "Sharp one-day drop (rebound)", "thresholds": (6, 10, 15)},
    # Breakout: closed at a 20-day high, near the top of its range, on heavy volume.
    "breakout": {"label": "20-day breakout on volume", "thresholds": (1.5, 2.0, 3.0)},
}
STOPS = (0.0, 1.5, 2.5)              # stop distance in ATRs below entry; 0 = no stop, time exit only
MARKET_FILTERS = ("any", "spy_up")   # spy_up: only trade when SPY closed above its 50-day average
MIN_TRAIN_TRADES = 10
SKIPPED_SPLIT = [0]   # eligible stock-days dropped for a possible split during the hold


@dataclass
class Candidate:
    day: dt.date
    strategy: str
    symbol: str
    strength: float
    gross: dict[float, float]        # stop (ATRs) -> gross return
    exit_reason: dict[float, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigKey:
    strategy: str
    threshold: float
    stop: float
    market: str

    def label(self) -> str:
        stop = "no stop" if not self.stop else f"{self.stop:g} ATR stop"
        mkt = "" if self.market == "any" else ", SPY uptrend only"
        return f"{self.strategy} ≥{self.threshold:g}, {stop}{mkt}"


def net(gross: float, slip: float) -> float:
    return (1 + gross) * (1 - slip) / (1 + slip) - 1


# ── feature arrays ─────────────────────────────────────────────

def _rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def _rolling_max(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = np.lib.stride_tricks.sliding_window_view(x, n)
        out[n - 1:] = w.max(axis=1)
    return out


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = x[:n].mean()
    out[n - 1] = a
    for i in range(n, len(x)):
        a = (a * (n - 1) + x[i]) / n
        out[i] = a
    return out


def rsi_series(c: np.ndarray, n: int = 2) -> np.ndarray:
    """Wilder's RSI for every bar (NaN until there is enough history)."""
    out = np.full(len(c), np.nan)
    if len(c) < n + 1:
        return out
    d = np.diff(c)
    g, l = _wilder(np.maximum(d, 0), n), _wilder(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(l == 0, np.where(g > 0, 100.0, 50.0), 100 - 100 / (1 + g / l))
    out[1:] = r
    return out


def features(a: dict[str, np.ndarray], split_ratio: float) -> dict[str, np.ndarray]:
    o, h, l, c, v = a["o"], a["h"], a["l"], a["c"], a["v"]
    prev_c = np.insert(c[:-1], 0, np.nan)
    tr = np.maximum(h - l, np.maximum(abs(h - prev_c), abs(l - prev_c)))
    tr[0] = h[0] - l[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        jump = np.maximum(c / prev_c, prev_c / c)
        open_jump = np.maximum(o / prev_c, prev_c / o)
        rng = h - l
        close_pos = np.where(rng > 0, (c - l) / rng, 0.5)
        f = {
            "sma100": _rolling_mean(c, 100),
            "rsi2": rsi_series(c, 2),
            "ret1": c / prev_c - 1,
            "vol_ratio": v / np.insert(_rolling_mean(v, 20)[:-1], 0, np.nan),
            "atr_pct": _wilder(tr, 14) / c,
            "adv_dollar": _rolling_mean(c * v, 20),
            "adv_shares": _rolling_mean(v, 20),
            "hi20_prev": np.insert(_rolling_max(c, 20)[:-1], 0, np.nan),
            "close_pos": close_pos,
            # A raw (unadjusted) series jumps on a split; skip anything near one.
            "split_recent": _rolling_max(np.nan_to_num(np.maximum(jump, open_jump), nan=1.0), 60) >= split_ratio,
            "jump": np.maximum(np.nan_to_num(jump, nan=1.0), np.nan_to_num(open_jump, nan=1.0)),
        }
    return f


def strength(strategy: str, f: dict[str, np.ndarray], i: int) -> float | None:
    """How strongly the idea fires at the close of bar i (higher = stronger), or None."""
    c_above = lambda k: not math.isnan(f[k][i])
    if strategy == "pullback":
        if not (c_above("sma100") and c_above("rsi2")) or f["close"][i] <= f["sma100"][i]:
            return None
        return 100 - f["rsi2"][i]
    if strategy == "reversal":
        if not c_above("ret1") or f["ret1"][i] >= 0:
            return None
        return -f["ret1"][i] * 100
    if strategy == "breakout":
        if not (c_above("hi20_prev") and c_above("vol_ratio")):
            return None
        if f["close"][i] <= f["hi20_prev"][i] or f["close_pos"][i] < 0.75:
            return None
        return float(f["vol_ratio"][i])
    raise ValueError(strategy)


def simulate_daily(a, j: int, k: int, atr_pct: float, stop_atrs: float) -> tuple[float, str]:
    """Buy the open of bar j, sell the close of bar k unless a stop is hit first."""
    entry = a["o"][j]
    stop = entry * (1 - stop_atrs * atr_pct) if stop_atrs else None
    for t in range(j, k + 1):
        if stop is not None:
            if t > j and a["o"][t] <= stop:
                return a["o"][t] / entry - 1, "stop"
            if a["l"][t] <= stop:
                return stop / entry - 1, "stop"
    return a["c"][k] / entry - 1, "time"


# ── candidate generation ───────────────────────────────────────

def spy_regime(daily: DailyTable) -> dict[int, bool]:
    """Ordinal of a session -> SPY closed above its 50-day average that day."""
    s = daily.data.get("SPY")
    if s is None:
        return {}
    m = _rolling_mean(s["c"], 50)
    return {int(d): bool(c > mm) for d, c, mm in zip(s["d"], s["c"], m) if not math.isnan(mm)}


def candidates(daily: DailyTable, cfg: Config, days: Sequence[tuple[dt.date, dt.date, dt.date]],
               symbols: Iterable[str] | None = None, progress: Callable[[str], None] = print) -> list[Candidate]:
    """All qualifying candidates. ``days`` holds (trade day, previous session, exit day)."""
    out: list[Candidate] = []
    skipped_split = SKIPPED_SPLIT
    syms = list(symbols) if symbols is not None else [s for s in daily.symbols() if s != "SPY"]
    plan = [(d.toordinal(), p.toordinal(), e.toordinal(), d) for d, p, e in days]
    for n, sym in enumerate(syms):
        a = daily.data[sym]
        if len(a["d"]) < 61:
            continue
        f = features(a, cfg.SPLIT_SUSPECT_RATIO)
        f["close"] = a["c"]
        pos = {int(x): i for i, x in enumerate(a["d"])}
        for d_ord, p_ord, e_ord, day in plan:
            i, j, k = pos.get(p_ord), pos.get(d_ord), pos.get(e_ord)
            if i is None or j is None or k is None or i < 60:
                continue
            # Eligibility, all as of the previous close.
            c = a["c"][i]
            if not (cfg.MIN_STOCK_PRICE <= c <= cfg.MAX_STOCK_PRICE):
                continue
            if f["adv_dollar"][i] < cfg.MIN_AVG_DOLLAR_VOLUME or f["adv_shares"][i] < cfg.MIN_AVG_SHARE_VOLUME:
                continue
            if f["split_recent"][i] or math.isnan(f["atr_pct"][i]):
                continue
            # A split inside the holding period would fake the return. This also drops genuine
            # moves of 80%+ in a day or two, so the count is reported to keep the bias visible.
            if f["jump"][j:k + 1].max() >= cfg.SPLIT_SUSPECT_RATIO:
                skipped_split[0] += 1
                continue
            # Don't chase: skip when it opens too far above the prior close (same rule as live).
            if a["o"][j] > c * (1 + cfg.MAX_CHASE_PERCENT / 100) or a["o"][j] <= 0:
                continue
            for strat in STRATEGIES:
                s = strength(strat, f, i)
                if s is None or s < min(STRATEGIES[strat]["thresholds"]):
                    continue
                gross, why = {}, {}
                for st in STOPS:
                    gross[st], why[st] = simulate_daily(a, j, k, f["atr_pct"][i], st)
                out.append(Candidate(day, strat, sym, round(float(s), 4), gross, why))
        if n % 500 == 0:
            progress(f"  scanned {n}/{len(syms)} symbols, {len(out)} candidates so far")
    return out


def baseline(daily: DailyTable, cfg: Config, days, symbols=None) -> list[float]:
    """Every eligible stock, every day, open-to-exit with no stop: the drift to beat."""
    out = []
    syms = list(symbols) if symbols is not None else [s for s in daily.symbols() if s != "SPY"]
    for sym in syms:
        a = daily.data[sym]
        if len(a["d"]) < 61:
            continue
        adv = _rolling_mean(a["c"] * a["v"], 20)
        pos = {int(x): i for i, x in enumerate(a["d"])}
        for d, p, e in days:
            i, j, k = pos.get(p.toordinal()), pos.get(d.toordinal()), pos.get(e.toordinal())
            if i is None or j is None or k is None or i < 60:
                continue
            if cfg.MIN_STOCK_PRICE <= a["c"][i] <= cfg.MAX_STOCK_PRICE and adv[i] >= cfg.MIN_AVG_DOLLAR_VOLUME:
                r = a["c"][k] / a["o"][j] - 1
                if abs(r) < 0.8:   # a split, not a trade
                    out.append(r)
    return out


# ── selection and evaluation ───────────────────────────────────

def all_configs() -> list[ConfigKey]:
    return [ConfigKey(s, t, st, m) for s, spec in STRATEGIES.items() for t in spec["thresholds"]
            for st in STOPS for m in MARKET_FILTERS]


def picks(cands: Sequence[Candidate], key: ConfigKey, regime: dict[int, bool],
          prev_session: dict[dt.date, dt.date]) -> list[Candidate]:
    """The single strongest qualifying candidate per day under a config."""
    best: dict[dt.date, Candidate] = {}
    for c in cands:
        if c.strategy != key.strategy or c.strength < key.threshold:
            continue
        if key.market == "spy_up" and not regime.get(prev_session[c.day].toordinal(), False):
            continue
        b = best.get(c.day)
        if b is None or (c.strength, c.symbol) > (b.strength, b.symbol):
            best[c.day] = c
    return [best[d] for d in sorted(best)]


def returns(ps: Sequence[Candidate], key: ConfigKey, slip: float) -> list[float]:
    return [net(c.gross[key.stop], slip) for c in ps]


def basket(cands: Sequence[Candidate], key: ConfigKey, regime, prev_session, slip: float) -> list[float]:
    """Average of all qualifying candidates per day: is there signal at all, with less noise?"""
    days: dict[dt.date, list[float]] = {}
    for c in cands:
        if c.strategy != key.strategy or c.strength < key.threshold:
            continue
        if key.market == "spy_up" and not regime.get(prev_session[c.day].toordinal(), False):
            continue
        days.setdefault(c.day, []).append(net(c.gross[key.stop], slip))
    return [sum(v) / len(v) for _, v in sorted(days.items())]


def _t(rs: Sequence[float]) -> float:
    if len(rs) < MIN_TRAIN_TRADES:
        return -math.inf
    sd = statistics.pstdev(rs)
    return (sum(rs) / len(rs)) / (sd / math.sqrt(len(rs))) if sd > 0 else -math.inf


def _month(d: dt.date) -> tuple[int, int]:
    return d.year, d.month


def walk_forward(by_config: dict[ConfigKey, list[Candidate]], months: list[tuple[int, int]],
                 configs: Sequence[ConfigKey], train_months: int = 6, test_months: int = 2,
                 slip: float = DEFAULT_SLIPPAGE) -> dict:
    """Choose the best config on each training window, trade it on the next test window."""
    folds, oos = [], []
    i = 0
    while i + train_months < len(months):
        train_m = set(months[i:i + train_months])
        test_m = set(months[i + train_months:i + train_months + test_months])
        best, best_t = None, 0.0     # must beat zero, otherwise stay out of the market
        for k in configs:
            t = _t(returns([c for c in by_config[k] if _month(c.day) in train_m], k, slip))
            if t > best_t:
                best, best_t = k, t
        test = [c for c in by_config[best] if _month(c.day) in test_m] if best else []
        oos.extend((best, c) for c in test)
        folds.append({
            "train": f"{min(train_m)[0]}-{min(train_m)[1]:02d}..{max(train_m)[0]}-{max(train_m)[1]:02d}",
            "test_period": f"{min(test_m)[0]}-{min(test_m)[1]:02d}..{max(test_m)[0]}-{max(test_m)[1]:02d}",
            "chosen": best.label() if best else "no trading",
            "train_t_stat": round(best_t, 2) if best else None,
            "test": metrics(returns(test, best, slip)) if best else {"trades": 0},
        })
        i += test_months
    return {"folds": folds,
            "out_of_sample": {f"{s * 100:g}%": metrics([net(c.gross[k.stop], s) for k, c in oos]) for s in SLIPPAGES}}


def sessions(cal: MarketCalendar, cfg: Config, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date, dt.date]]:
    out = []
    d = start
    while d <= end:
        s = cal.session(d)
        if s is not None:
            prev = d - dt.timedelta(days=1)
            while cal.session(prev) is None:
                prev -= dt.timedelta(days=1)
            out.append((d, prev, cal.plan_exit(s.open, cfg).date()))
        d += dt.timedelta(days=1)
    return out


def evaluate(cands: list[Candidate], days, regime: dict[int, bool], base: list[float]) -> dict:
    prev_session = {d: p for d, p, _ in days}
    months = sorted({_month(d) for d, _, _ in days})
    configs = all_configs()
    by_config = {k: picks(cands, k, regime, prev_session) for k in configs}
    slip = DEFAULT_SLIPPAGE

    table = []
    for k in configs:
        rs = returns(by_config[k], k, slip)
        m = metrics(rs)
        m["basket_avg_pct"] = metrics(basket(cands, k, regime, prev_session, slip)).get("avg_return_pct")
        exits: dict[str, int] = {}
        for c in by_config[k]:
            exits[c.exit_reason[k.stop]] = exits.get(c.exit_reason[k.stop], 0) + 1
        m["exits"] = exits
        table.append({"config": k.label(), "key": k.__dict__, **m})

    per_strategy = {s: walk_forward(by_config, months, [k for k in configs if k.strategy == s])
                    for s in STRATEGIES}
    combined = walk_forward(by_config, months, configs)
    return {
        "period": f"{days[0][0]}..{days[-1][0]}" if days else None,
        "trading_days": len(days),
        "candidates": len(cands),
        "stock_days_skipped_for_possible_split": SKIPPED_SPLIT[0],
        "baseline_every_eligible_stock_no_slippage": metrics(base),
        "configs_full_period_in_sample": table,
        "walk_forward_per_idea": per_strategy,
        "walk_forward_choosing_among_all_ideas": combined,
    }


def run(cache: HistoryCache, cfg: Config, start: dt.date, end: dt.date,
        progress: Callable[[str], None] = print) -> dict:
    cal = MarketCalendar()
    universe = cache.universe()
    elig = eligible_symbols(universe, cfg)
    daily = cache.daily(elig, start - dt.timedelta(days=330), end + dt.timedelta(days=7),
                        cfg.MAX_STOCK_PRICE, cfg.MIN_AVG_DOLLAR_VOLUME, always_keep=("SPY",))
    allowed = set(elig)
    syms = [s for s in daily.symbols() if s in allowed and s != "SPY"]
    progress(f"daily bars for {len(syms)} eligible symbols")
    days = sessions(cal, cfg, start, end)
    cands = candidates(daily, cfg, days, syms, progress)
    progress(f"{len(cands)} candidates; computing baseline")
    return evaluate(cands, days, spy_regime(daily), baseline(daily, cfg, days, syms))


# ── report ─────────────────────────────────────────────────────

def _row(name: str, m: dict, extra: str = "") -> str:
    if not m.get("trades"):
        return f"| {name} | 0 | | | | | |{extra}"
    return (f"| {name} | {m['trades']} | {m['win_rate']}% | {m['avg_return_pct']:+.2f}% | "
            f"{m['profit_factor']} | {m['max_drawdown_pct']}% | {m['t_stat']} |{extra}")


def verdict(wf: dict) -> str:
    lines = []
    for label, m in wf["out_of_sample"].items():
        if not m.get("trades"):
            lines.append(f"{label} slippage: no out-of-sample trades")
            continue
        t = m.get("t_stat") or 0
        status = ("positive and statistically meaningful" if m["avg_return_pct"] > 0 and t >= 2
                  else "positive but could be luck" if m["avg_return_pct"] > 0
                  else "no edge (average trade loses)")
        lines.append(f"{label} slippage: {status} (avg {m['avg_return_pct']:+.2f}%/trade, t={t}, {m['trades']} trades)")
    return "\n".join(lines)


def report_markdown(s: dict) -> str:
    head = "| | Trades | Win rate | Avg | Profit factor | Max DD (pts) | t-stat |\n|---|---|---|---|---|---|---|"
    out = [f"# APEX Stocks signal research {s['period']}", "",
           f"{s['trading_days']} trading days, {s['candidates']} qualifying candidates across all ideas "
           f"({s.get('stock_days_skipped_for_possible_split', 0)} stock-days skipped for a possible split). "
           "Daily-bar simulation: buy the open, sell the exit-day close, stops checked on daily lows.", "",
           "## Verdict: choosing among all ideas, walk-forward, out-of-sample", "", "```",
           verdict(s["walk_forward_choosing_among_all_ideas"]), "```", "", "| Train | Test | Chosen | Test trades | Test avg | Test t |",
           "|---|---|---|---|---|---|"]
    for f in s["walk_forward_choosing_among_all_ideas"]["folds"]:
        m = f["test"]
        out.append(f"| {f['train']} | {f['test_period']} | {f['chosen']} | {m.get('trades', 0)} | "
                   f"{m.get('avg_return_pct', '')} | {m.get('t_stat', '')} |")
    out += ["", "## Each idea on its own, walk-forward, out-of-sample at 0.25% slippage", "", head]
    for strat, wf in s["walk_forward_per_idea"].items():
        out.append(_row(STRATEGIES[strat]["label"], wf["out_of_sample"]["0.25%"]))
    out += ["", "Chosen per fold:", ""]
    for strat, wf in s["walk_forward_per_idea"].items():
        out.append(f"- {strat}: " + "; ".join(f"{f['test_period']} → {f['chosen']} "
                                              f"({f['test'].get('avg_return_pct', 'n/a')}%)" for f in wf["folds"]))
    out += ["", "## Every config over the full period (in-sample, 0.25% slippage; for diagnosis only)", "",
            "| Config | Trades | Win rate | Avg | Profit factor | Max DD (pts) | t-stat | Basket avg | Exits |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(s["configs_full_period_in_sample"], key=lambda r: -(r.get("t_stat") or -99)):
        out.append(_row(r["config"], r, f" {r.get('basket_avg_pct', '')} | {r.get('exits', '')} |"))
    out += ["", "## Baseline", "", head, _row("Every eligible stock, every day, no stop, no slippage",
                                             s["baseline_every_eligible_stock_no_slippage"]), "",
            "Notes: returns are per trade. The exit is the exit day's 4:00 PM close, a stand-in for the 3:30 PM "
            "exit. The max-chase rule uses the prior close as the reference. Stops are ATR-based and checked on "
            "daily lows; if a stop and a big gain happen the same day, the stop is assumed. Trades near a split "
            "are skipped. Delisted stocks are included where Alpaca still lists them."]
    return "\n".join(out)
