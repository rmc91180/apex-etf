"""Turn a chosen candidate into concrete numbers: entry, target, stop, exit time, size."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from .config import Config
from .market_calendar import MarketCalendar, Session

if TYPE_CHECKING:
    from .engine import Candidate


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    reference_price: float
    entry_low: float
    entry_high: float
    max_entry: float
    target: float
    stop: float
    target_pct: float
    stop_pct: float
    planned_entry_at: object   # datetime, session open
    planned_exit_at: object    # datetime
    holding_hours: float
    max_shares: int | None = None
    max_dollar_risk: float | None = None
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _r(x: float) -> float:
    # Sub-$1 stocks quote in 4 decimals; everything we allow trades in cents.
    return round(x, 2)


def risk_factors(c: "Candidate") -> list[str]:
    f, out = c.features, []
    if f.market_cap is not None and f.market_cap < 300e6:
        out.append(f"small cap (${f.market_cap / 1e6:.0f}M)")
    if f.float_shares is not None and f.float_shares < 20e6:
        out.append(f"low float ({f.float_shares / 1e6:.1f}M shares)")
    if f.atr_pct is not None and f.atr_pct > 8:
        out.append(f"high volatility (average daily range {f.atr_pct:.1f}%)")
    if f.gap_pct > 10:
        out.append(f"large pre-market gap ({f.gap_pct:+.1f}%) can fade at the open")
    if f.catalyst_category == "earnings":
        out.append("earnings reaction can reverse quickly")
    if f.catalyst_category is None:
        out.append("no identifiable news catalyst")
    if f.rsi14 is not None and f.rsi14 > 75:
        out.append(f"overbought (RSI {f.rsi14:.0f})")
    out.extend(c.score.quality_issues)
    return out


def build_plan(c: "Candidate", cfg: Config, session: Session, cal: MarketCalendar) -> TradePlan:
    ref = c.features.reference_price
    target = _r(ref * (1 + cfg.PROFIT_TARGET_PERCENT / 100))
    stop = _r(ref * (1 - cfg.STOP_LOSS_PERCENT / 100))
    entry_at = session.open
    exit_at = cal.plan_exit(entry_at, cfg)

    max_shares = max_risk = None
    if cfg.ACCOUNT_SIZE > 0:
        max_risk = round(cfg.ACCOUNT_SIZE * cfg.MAX_POSITION_RISK_PERCENT / 100, 2)
        per_share = ref - stop
        by_risk = math.floor(max_risk / per_share) if per_share > 0 else 0
        by_liquidity = math.floor(c.features.avg_volume_20d * cfg.MAX_POSITION_ADV_PERCENT / 100)
        by_cash = math.floor(cfg.ACCOUNT_SIZE / ref)
        max_shares = max(0, min(by_risk, by_liquidity, by_cash))

    return TradePlan(
        symbol=c.symbol,
        reference_price=_r(ref),
        entry_low=_r(ref * (1 - cfg.ENTRY_BAND_PERCENT / 100)),
        entry_high=_r(ref * (1 + cfg.ENTRY_BAND_PERCENT / 100)),
        max_entry=_r(ref * (1 + cfg.MAX_CHASE_PERCENT / 100)),
        target=target,
        stop=stop,
        target_pct=round((target / ref - 1) * 100, 2),
        stop_pct=round((stop / ref - 1) * 100, 2),
        planned_entry_at=entry_at,
        planned_exit_at=exit_at,
        holding_hours=round((exit_at - entry_at).total_seconds() / 3600, 2),
        max_shares=max_shares,
        max_dollar_risk=max_risk,
        risks=risk_factors(c),
    )
