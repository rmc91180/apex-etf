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
    # Bullish news before the open (same keyword classifier and 20h window as the live engine).
    # Strength is the catalyst score x 100. Only tested when a news archive is loaded.
    "catalyst": {"label": "Bullish news catalyst", "thresholds": (40, 60, 75),
                 "plans": ("open|0", "open|2.5", "day>open", "day>close")},
}
PRICE_STRATEGIES = ("pullback", "reversal", "breakout")
STOPS = (0.0, 1.5, 2.5)              # stop distance in ATRs below entry; 0 = no stop, time exit only
# How the trade is held. "open" plans buy the morning open and sell at the planned exit, with a stop
# in ATRs (0 = none). The two overnight plans buy the signal day's close and sell the next session's
# open or close; they need an afternoon alert instead of a morning one, and skip Fridays (48h cap).
PLANS = tuple(f"open|{st:g}" for st in STOPS) + ("close>open", "close>close")
PLAN_LABELS = {"open|0": "buy open, no stop", "open|1.5": "buy open, 1.5 ATR stop",
               "open|2.5": "buy open, 2.5 ATR stop", "close>open": "buy close, sell next open",
               "close>close": "buy close, sell next close",
               "day>open": "buy news-day close, sell next open", "day>close": "buy news-day close, sell next close"}
MARKET_FILTERS = ("any", "spy_up")   # spy_up: only trade when SPY closed above its 50-day average
MIN_TRAIN_TRADES = 10
SKIPPED_SPLIT = [0]   # eligible stock-days dropped for a possible split during the hold


@dataclass
class Candidate:
    day: dt.date
    strategy: str
    symbol: str
    strength: float
    gross: dict[str, float | None]   # plan -> gross return (None: not tradable under that plan)
    exit_reason: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ConfigKey:
    strategy: str
    threshold: float
    plan: str
    market: str

    def label(self) -> str:
        mkt = "" if self.market == "any" else ", SPY uptrend only"
        return f"{self.strategy} ≥{self.threshold:g}, {PLAN_LABELS[self.plan]}{mkt}"


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
            if f["jump"][j] >= cfg.SPLIT_SUSPECT_RATIO or a["o"][j] <= 0:
                skipped_split[0] += 1
                continue
            # Morning plans: don't chase a stock that opens too far above the prior close (as live).
            open_ok = (f["jump"][j:k + 1].max() < cfg.SPLIT_SUSPECT_RATIO
                       and a["o"][j] <= c * (1 + cfg.MAX_CHASE_PERCENT / 100))
            # Overnight plans: the exit must come within 48 hours, so no Friday or pre-holiday entries.
            overnight_ok = (day - dt.date.fromordinal(p_ord)).days <= 2
            if not open_ok and not overnight_ok:
                continue
            for strat in PRICE_STRATEGIES:
                s = strength(strat, f, i)
                if s is None or s < min(STRATEGIES[strat]["thresholds"]):
                    continue
                gross: dict[str, float | None] = {}
                why: dict[str, str] = {}
                for st in STOPS:
                    key = f"open|{st:g}"
                    if open_ok:
                        gross[key], why[key] = simulate_daily(a, j, k, f["atr_pct"][i], st)
                    else:
                        gross[key] = None
                gross["close>open"] = a["o"][j] / c - 1 if overnight_ok else None
                gross["close>close"] = a["c"][j] / c - 1 if overnight_ok else None
                why["close>open"] = why["close>close"] = "time"
                out.append(Candidate(day, strat, sym, round(float(s), 4), gross, why))
        if n % 500 == 0:
            progress(f"  scanned {n}/{len(syms)} symbols, {len(out)} candidates so far")
    return out


def baseline(daily: DailyTable, cfg: Config, days, symbols=None) -> dict[str, list[float]]:
    """Every eligible stock, every day, no stop: the drift each plan has to beat."""
    out: dict[str, list[float]] = {"open|0": [], "close>open": [], "close>close": []}
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
                    out["open|0"].append(r)
                if (d - p).days <= 2 and abs(a["c"][j] / a["c"][i] - 1) < 0.8:
                    out["close>open"].append(a["o"][j] / a["c"][i] - 1)
                    out["close>close"].append(a["c"][j] / a["c"][i] - 1)
    return out


# ── selection and evaluation ───────────────────────────────────

def all_configs() -> list[ConfigKey]:
    return [ConfigKey(s, t, pl, m) for s, spec in STRATEGIES.items() for t in spec["thresholds"]
            for pl in spec.get("plans", PLANS) for m in MARKET_FILTERS]


def picks(cands: Sequence[Candidate], key: ConfigKey, regime: dict[int, bool],
          prev_session: dict[dt.date, dt.date]) -> list[Candidate]:
    """The single strongest qualifying candidate per day under a config."""
    best: dict[dt.date, Candidate] = {}
    for c in cands:
        if c.strategy != key.strategy or c.strength < key.threshold or c.gross[key.plan] is None:
            continue
        if key.market == "spy_up" and not regime.get(prev_session[c.day].toordinal(), False):
            continue
        b = best.get(c.day)
        if b is None or (c.strength, c.symbol) > (b.strength, b.symbol):
            best[c.day] = c
    return [best[d] for d in sorted(best)]


def returns(ps: Sequence[Candidate], key: ConfigKey, slip: float) -> list[float]:
    return [net(c.gross[key.plan], slip) for c in ps]


def basket(cands: Sequence[Candidate], key: ConfigKey, regime, prev_session, slip: float) -> list[float]:
    """Average of all qualifying candidates per day: is there signal at all, with less noise?"""
    days: dict[dt.date, list[float]] = {}
    for c in cands:
        if c.strategy != key.strategy or c.strength < key.threshold or c.gross[key.plan] is None:
            continue
        if key.market == "spy_up" and not regime.get(prev_session[c.day].toordinal(), False):
            continue
        days.setdefault(c.day, []).append(net(c.gross[key.plan], slip))
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
            "out_of_sample": {f"{s * 100:g}%": metrics([net(c.gross[k.plan], s) for k, c in oos]) for s in SLIPPAGES}}


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


def evaluate(cands: list[Candidate], days, regime: dict[int, bool], base: dict[str, list[float]]) -> dict:
    prev_session = {d: p for d, p, _ in days}
    months = sorted({_month(d) for d, _, _ in days})
    present = {c.strategy for c in cands}
    configs = [k for k in all_configs() if k.strategy in present]
    by_config = {k: picks(cands, k, regime, prev_session) for k in configs}
    slip = DEFAULT_SLIPPAGE

    table = []
    for k in configs:
        rs = returns(by_config[k], k, slip)
        m = metrics(rs)
        m["basket_avg_pct"] = metrics(basket(cands, k, regime, prev_session, slip)).get("avg_return_pct")
        exits: dict[str, int] = {}
        for c in by_config[k]:
            exits[c.exit_reason[k.plan]] = exits.get(c.exit_reason[k.plan], 0) + 1
        m["exits"] = exits
        table.append({"config": k.label(), "key": k.__dict__, **m})

    per_strategy = {s: walk_forward(by_config, months, [k for k in configs if k.strategy == s])
                    for s in STRATEGIES if s in present}
    combined = walk_forward(by_config, months, configs)
    by_timing = {
        "Morning alert (buy the open)": walk_forward(by_config, months, [k for k in configs if k.plan.startswith("open")]),
        "Afternoon alert (buy the close, hold overnight)":
            walk_forward(by_config, months, [k for k in configs if not k.plan.startswith("open")]),
    }
    return {
        "period": f"{days[0][0]}..{days[-1][0]}" if days else None,
        "trading_days": len(days),
        "candidates": len(cands),
        "stock_days_skipped_for_possible_split": SKIPPED_SPLIT[0],
        "baseline_every_eligible_stock_no_slippage": {PLAN_LABELS[k]: metrics(v) for k, v in base.items()},
        "configs_full_period_in_sample": table,
        "walk_forward_per_idea": per_strategy,
        "walk_forward_choosing_among_all_ideas": combined,
        "walk_forward_by_alert_timing": by_timing,
    }


def _eligible_by_month(daily: DailyTable, cfg: Config, syms, days) -> dict[tuple[int, int], set[str]]:
    out: dict[tuple[int, int], set[str]] = {}
    prev = sorted({p for _, p, _ in days})
    for sym in syms:
        a = daily.data[sym]
        adv = _rolling_mean(a["c"] * a["v"], 20)
        pos = {int(x): i for i, x in enumerate(a["d"])}
        for p in prev:
            i = pos.get(p.toordinal())
            if i is not None and cfg.MIN_STOCK_PRICE <= a["c"][i] <= cfg.MAX_STOCK_PRICE \
                    and adv[i] >= cfg.MIN_AVG_DOLLAR_VOLUME:
                out.setdefault(_month(p), set()).add(sym)
    return out


def catalyst_candidates(daily: DailyTable, cfg: Config, days, syms, news_for_month: Callable,
                        progress: Callable[[str], None] = print) -> list[Candidate]:
    """Bullish-news candidates, scored exactly as the live engine scores catalysts."""
    from .catalysts import assess
    from .models import NewsItem

    by_month = _eligible_by_month(daily, cfg, syms, days)
    news: dict[str, list[NewsItem]] = {}
    for (y, m), month_syms in sorted(by_month.items()):
        items = news_for_month(dt.date(y, m, 1), sorted(month_syms))
        for it in items:
            n = NewsItem(id=it["id"], symbols=tuple(it["s"]), headline=it["h"], source=it.get("src", ""), url="",
                         published_at=dt.datetime.fromisoformat(it["t"].replace("Z", "+00:00")), summary=it["m"])
            for s in n.symbols:
                news.setdefault(s, []).append(n)
        progress(f"  news {y}-{m:02d}: {len(items)} items for {len(month_syms)} symbols")
    for v in news.values():
        v.sort(key=lambda n: n.published_at)
    times = {s: [n.published_at for n in v] for s, v in news.items()}

    import bisect
    out: list[Candidate] = []
    for sym in syms:
        if sym not in news:
            continue
        a = daily.data[sym]
        f = features(a, cfg.SPLIT_SUSPECT_RATIO)
        pos = {int(x): i for i, x in enumerate(a["d"])}
        for day, prev, exit_day in days:
            i, j, k = pos.get(prev.toordinal()), pos.get(day.toordinal()), pos.get(exit_day.toordinal())
            if i is None or j is None or k is None or i < 60:
                continue
            c = a["c"][i]
            if not (cfg.MIN_STOCK_PRICE <= c <= cfg.MAX_STOCK_PRICE):
                continue
            if f["adv_dollar"][i] < cfg.MIN_AVG_DOLLAR_VOLUME or f["adv_shares"][i] < cfg.MIN_AVG_SHARE_VOLUME:
                continue
            if f["split_recent"][i] or math.isnan(f["atr_pct"][i]) or a["o"][j] <= 0:
                continue
            now = dt.datetime.combine(day, cfg.scan_time, ET)
            lo = bisect.bisect_left(times[sym], now - dt.timedelta(hours=cfg.NEWS_LOOKBACK_HOURS))
            hi = bisect.bisect_right(times[sym], now)
            if lo >= hi:
                continue
            ca = assess(news[sym][lo:hi], now)
            if ca.score * 100 < min(STRATEGIES["catalyst"]["thresholds"]):
                continue
            gross: dict[str, float | None] = {}
            why: dict[str, str] = {}
            hold_ok = f["jump"][j + 1:k + 1].max() < cfg.SPLIT_SUSPECT_RATIO if k > j else True
            for st, key in ((0.0, "open|0"), (2.5, "open|2.5")):
                if hold_ok:
                    gross[key], why[key] = simulate_daily(a, j, k, f["atr_pct"][i], st)
                else:
                    gross[key] = None
            nxt = j + 1
            ok = (nxt < len(a["d"]) and a["d"][nxt] - a["d"][j] <= 2 and f["jump"][nxt] < cfg.SPLIT_SUSPECT_RATIO)
            gross["day>open"] = a["o"][nxt] / a["c"][j] - 1 if ok else None
            gross["day>close"] = a["c"][nxt] / a["c"][j] - 1 if ok else None
            why["day>open"] = why["day>close"] = "time"
            out.append(Candidate(day, "catalyst", sym, round(ca.score * 100, 2), gross, why))
    progress(f"{len(out)} catalyst candidates")
    return out


def run(cache: HistoryCache, cfg: Config, start: dt.date, end: dt.date,
        progress: Callable[[str], None] = print, with_news: bool = False) -> dict:
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
    if with_news:
        cands += catalyst_candidates(daily, cfg, days, syms, cache.news_archive, progress)
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
           "Daily-bar simulation: morning plans buy the open and sell the exit-day close with stops checked on "
           "daily lows; overnight plans buy the signal day's close and sell the next open or close.", "",
           "## Verdict: choosing among all ideas, walk-forward, out-of-sample", "", "```",
           verdict(s["walk_forward_choosing_among_all_ideas"]), "```", "", "| Train | Test | Chosen | Test trades | Test avg | Test t |",
           "|---|---|---|---|---|---|"]
    for f in s["walk_forward_choosing_among_all_ideas"]["folds"]:
        m = f["test"]
        out.append(f"| {f['train']} | {f['test_period']} | {f['chosen']} | {m.get('trades', 0)} | "
                   f"{m.get('avg_return_pct', '')} | {m.get('t_stat', '')} |")
    out += ["", "## By alert timing, walk-forward, out-of-sample at 0.25% slippage", "", head]
    for name, wf in s["walk_forward_by_alert_timing"].items():
        out.append(_row(name, wf["out_of_sample"]["0.25%"]))
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
    out += ["", "## Baseline: every eligible stock, every day, no slippage", "", head]
    out += [_row(k, v) for k, v in s["baseline_every_eligible_stock_no_slippage"].items()]
    out += ["",
            "Notes: returns are per trade. Overnight plans use the signal day's close as both the signal and the "
            "entry, so live they need the scan near 3:45 PM and an entry at the close. The morning exit is the exit day's 4:00 PM close, a stand-in for the 3:30 PM "
            "exit. The max-chase rule uses the prior close as the reference. Stops are ATR-based and checked on "
            "daily lows; if a stop and a big gain happen the same day, the stop is assumed. Trades near a split "
            "are skipped. Delisted stocks are included where Alpaca still lists them."]
    return "\n".join(out)
