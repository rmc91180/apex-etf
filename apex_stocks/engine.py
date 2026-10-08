"""The daily selection pipeline: universe -> filters -> features -> scores -> one pick or NO TRADE.

``select`` reads only from the provider and never writes or sends anything,
so the same code runs live and in backtests.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Sequence

from . import catalysts, features as feat, scoring
from .config import Config
from .market_calendar import ET, MarketCalendar
from .models import Bar, Security, Snapshot
from .plan import TradePlan, build_plan
from .providers.base import DataUnavailable, MarketDataProvider
from .universe import static_rejections

BEARISH_REJECT_MATERIALITY = 0.6


@dataclass
class Candidate:
    symbol: str
    name: str
    rejections: list[str] = field(default_factory=list)
    features: feat.Features | None = None
    score: scoring.Score | None = None

    @property
    def passed(self) -> bool:
        return not self.rejections and self.score is not None


@dataclass
class SelectionResult:
    trade_date: dt.date
    as_of: dt.datetime
    status: str                           # recommended, no_trade, skipped, failed
    reason: str
    provider: str
    live_data: bool
    universe_size: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    pick: Candidate | None = None
    plan: TradePlan | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def ranked(self) -> list[Candidate]:
        ok = [c for c in self.candidates if c.passed]
        return sorted(ok, key=lambda c: (c.score.signal, c.score.confidence), reverse=True)


def daily_bars_many(provider: MarketDataProvider, symbols: Sequence[str], start: dt.date,
                    end: dt.date) -> dict[str, list[Bar]]:
    many = getattr(provider, "get_daily_bars_many", None)
    if many:
        return many(symbols, start, end)
    out = {}
    for s in symbols:
        try:
            out[s] = provider.get_daily_bars(s, start, end)
        except DataUnavailable:
            pass
    return out


def hard_rejections(f: feat.Features, snap: Snapshot, cat: catalysts.CatalystAssessment, cfg: Config) -> list[str]:
    r = []
    if not cfg.MIN_STOCK_PRICE <= f.reference_price <= cfg.MAX_STOCK_PRICE:
        r.append(f"price ${f.reference_price:.2f} outside ${cfg.MIN_STOCK_PRICE:.2f}-${cfg.MAX_STOCK_PRICE:.2f}")
    if f.history_days < cfg.MIN_HISTORY_DAYS:
        r.append(f"only {f.history_days} days of history")
    if f.avg_dollar_volume_20d < cfg.MIN_AVG_DOLLAR_VOLUME:
        r.append(f"avg dollar volume ${f.avg_dollar_volume_20d / 1e6:.1f}M below minimum")
    if f.avg_volume_20d < cfg.MIN_AVG_SHARE_VOLUME:
        r.append(f"avg volume {f.avg_volume_20d:,.0f} shares below minimum")
    if snap.bid and snap.ask and snap.bid > snap.ask:
        r.append("crossed quote (bid above ask)")
    if f.spread_pct is not None and f.spread_pct > cfg.MAX_SPREAD_PERCENT:
        r.append(f"spread {f.spread_pct:.2f}% too wide")
    if f.quote_age_minutes > cfg.MAX_QUOTE_AGE_MINUTES:
        r.append(f"stale quote ({f.quote_age_minutes:.0f} min old)")
    if f.max_daily_jump_ratio >= cfg.SPLIT_SUSPECT_RATIO:
        r.append(f"possible split or bad data (a {f.max_daily_jump_ratio:.1f}x one-day move in history)")
    if abs(f.gap_pct) > cfg.MAX_ABS_GAP_PERCENT:
        r.append(f"abnormal pre-market gap {f.gap_pct:+.0f}%")
    if f.ret_5d is not None and f.ret_5d > cfg.MAX_5D_RUNUP_NO_CATALYST_PERCENT and not cat.has_bullish:
        r.append(f"already up {f.ret_5d:.0f}% in 5 days with no new catalyst")
    for c in cat.negatives:
        if c.materiality >= BEARISH_REJECT_MATERIALITY:
            r.append(f"negative news ({c.category}): {c.headline[:80]}")
            break
    return r


def select(provider: MarketDataProvider, cfg: Config, now: dt.datetime,
           calendar: MarketCalendar | None = None) -> SelectionResult:
    cal = calendar or MarketCalendar()
    now = now.astimezone(ET)
    trade_date = now.date()
    result = SelectionResult(trade_date, now, "failed", "", provider.name, provider.is_live)

    session = cal.session(trade_date)
    if session is None:
        result.status, result.reason = "skipped", "market closed today"
        return result
    if now >= session.open:
        result.status, result.reason = "skipped", "scan must run before the open"
        return result

    try:
        universe: list[Security] = provider.list_universe()
    except DataUnavailable as e:
        result.reason = f"universe unavailable: {e}"
        return result
    result.universe_size = len(universe)
    by_symbol = {s.symbol: s for s in universe}

    # Stage 1: static eligibility.
    eligible = []
    for sec in universe:
        rej = static_rejections(sec, cfg)
        if rej:
            continue  # not stored: thousands of ETFs/warrants add nothing to the audit
        eligible.append(sec.symbol)

    # Stage 2: price screen on the latest snapshot.
    try:
        snaps = provider.get_snapshots(eligible)
    except DataUnavailable as e:
        result.reason = f"quotes unavailable: {e}"
        return result
    priced = [s for s in eligible if s in snaps and
              cfg.MIN_STOCK_PRICE * 0.8 <= (snaps[s].premarket_price or snaps[s].last_price or 0) <= cfg.MAX_STOCK_PRICE * 1.2]
    if not priced:
        result.status, result.reason = "no_trade", "no eligible stocks with usable quotes in the price range"
        return result

    # Stage 3: history and liquidity.
    start = trade_date - dt.timedelta(days=320)  # ~220 sessions, enough for the 200-day average
    end = trade_date - dt.timedelta(days=1)  # never include the trade date itself
    bars = daily_bars_many(provider, priced, start, end)

    liquid = []
    for sym in priced:
        b = [x for x in bars.get(sym, []) if x.start.astimezone(ET).date() < trade_date]
        if len(b) < 21:
            result.candidates.append(Candidate(sym, by_symbol[sym].name, [f"only {len(b)} days of history"]))
            continue
        avg_dollar = sum(x.close * x.volume for x in b[-20:]) / 20
        if avg_dollar < cfg.MIN_AVG_DOLLAR_VOLUME:
            continue  # the long illiquid tail is not worth storing per day
        liquid.append((sym, b))

    # Stage 4: detailed quotes (pre-market volume) and news for the liquid set.
    syms = [s for s, _ in liquid]
    try:
        detail = provider.get_snapshots(syms, include_premarket=True) if syms else {}
    except DataUnavailable as e:
        result.issues.append(f"pre-market detail unavailable: {e}")
        detail = {s: snaps[s] for s in syms}
    try:
        news = provider.get_news(syms, now - dt.timedelta(hours=cfg.NEWS_LOOKBACK_HOURS)) if syms else []
    except DataUnavailable as e:
        result.issues.append(f"news unavailable, catalysts not assessed: {e}")
        news = []
    news_by = {}
    for n in news:
        for s in n.symbols:
            news_by.setdefault(s, []).append(n)

    for sym, b in liquid:
        sec = by_symbol[sym]
        snap = detail.get(sym) or snaps[sym]
        cat = catalysts.assess(news_by.get(sym, []), now)
        f = feat.compute(sec, b, snap, cat, now, cfg.PREMARKET_VOLUME_FRACTION)
        cand = Candidate(sym, sec.name, hard_rejections(f, snap, cat, cfg), f)
        if not cand.rejections:
            cand.score = scoring.score(f, cfg)
        result.candidates.append(cand)

    ranked = result.ranked
    qualified = [c for c in ranked
                 if c.score.signal >= cfg.MIN_SIGNAL_SCORE and c.score.confidence >= cfg.MIN_CONFIDENCE_SCORE]
    if not qualified:
        best = ranked[0] if ranked else None
        result.status = "no_trade"
        result.reason = ("no candidate met the minimum signal and confidence scores"
                         + (f" (best: {best.symbol} signal {best.score.signal}, confidence {best.score.confidence})"
                            if best else ""))
        return result

    pick = qualified[0]
    result.pick = pick
    result.plan = build_plan(pick, cfg, session, cal)
    result.status, result.reason = "recommended", f"{pick.symbol} ranked first of {len(ranked)} scored candidates"
    return result
