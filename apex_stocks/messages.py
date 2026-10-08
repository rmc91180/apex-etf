"""Telegram message text. Plain statements of what the data shows; never a promise."""
from __future__ import annotations

from html import escape

from .engine import Candidate, SelectionResult
from .market_calendar import format_dual

DISCLAIMER = "Quantitative signal, not a guarantee. You decide whether to trade."


def reasons(c: Candidate, limit: int = 4) -> list[str]:
    f, d = c.features, c.score.details
    out = []
    if f.catalyst_headline:
        out.append(f"Catalyst ({f.catalyst_category.replace('_', ' ')}): {escape(f.catalyst_headline[:120])}")
    if f.premarket_rel_volume and f.premarket_rel_volume >= 2:
        out.append(f"Pre-market volume {f.premarket_rel_volume:.1f}x normal")
    if f.gap_pct >= 2:
        out.append(f"Gapping {f.gap_pct:+.1f}% pre-market")
    if d["technical"].get("breakout_20d", 0) >= 0.8:
        out.append("Trading at or above its 20-day high")
    if f.ret_5d is not None and f.ret_5d > 3:
        out.append(f"Up {f.ret_5d:.1f}% over 5 days")
    if f.rsi14 is not None and 50 <= f.rsi14 <= 70:
        out.append(f"RSI {f.rsi14:.0f}: strong but not overbought")
    if f.avg_dollar_volume_20d >= 20e6:
        out.append(f"Liquid: ${f.avg_dollar_volume_20d / 1e6:.0f}M average daily dollar volume")
    return out[:limit]


def morning_signal(r: SelectionResult, user_tz: str) -> str:
    c, p = r.pick, r.plan
    lines = [
        f"📈 <b>DAILY STOCK SIGNAL: {escape(c.symbol)}</b>",
        escape(c.name),
        "",
        f"Pre-market: ${p.reference_price:.2f}",
        f"Signal score: {c.score.signal:.0f}/100 · Confidence: {c.score.confidence:.0f}/100",
        "",
        f"<b>Entry</b> at the open: ${p.entry_low:.2f}–${p.entry_high:.2f} (skip above ${p.max_entry:.2f})",
        f"<b>Target</b>: ${p.target:.2f} ({p.target_pct:+.1f}%)",
        f"<b>Stop</b>: ${p.stop:.2f} ({p.stop_pct:+.1f}%)",
        f"<b>Planned exit</b>: {format_dual(p.planned_exit_at, user_tz)}",
    ]
    if p.max_shares is not None:
        lines.append(f"Max size: {p.max_shares:,} shares (risking about ${p.max_dollar_risk:,.0f} to the stop)")
    why = reasons(c)
    if why:
        lines += ["", "<b>Why</b>"] + [f"{i}. {w}" for i, w in enumerate(why, 1)]
    if p.risks:
        lines += ["", "<b>Risks</b>"] + [f"• {escape(x)}" for x in p.risks[:4]]
    lines += ["", f"<i>{DISCLAIMER}</i>"]
    return "\n".join(lines)


def no_trade(r: SelectionResult) -> str:
    lines = [f"⏸ <b>NO TRADE TODAY</b> ({r.trade_date:%a %b %-d})", escape(r.reason)]
    runner_up = r.ranked[:3]
    if runner_up:
        lines += ["", "Closest candidates:"] + [
            f"• {escape(c.symbol)}: signal {c.score.signal:.0f}, confidence {c.score.confidence:.0f}" for c in runner_up]
    if r.issues:
        lines += ["", "Data issues: " + escape("; ".join(r.issues))]
    return "\n".join(lines)
