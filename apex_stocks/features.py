"""Turn raw bars, a snapshot and news into the numbers the scorer uses.

Only data available before the scan time is used: daily bars must end before
the trade date, so a backtest cannot see the future.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass

from . import indicators as ind
from .catalysts import CatalystAssessment
from .models import Bar, Security, Snapshot


@dataclass(frozen=True)
class Features:
    symbol: str
    reference_price: float          # pre-market price, else last trade
    prev_close: float
    gap_pct: float                  # reference vs previous close
    ret_1d: float | None
    ret_5d: float | None
    ret_20d: float | None
    sma20: float | None
    sma50: float | None
    sma200: float | None
    sma20_slope_pct: float | None   # 5-day change of the 20-day average
    rsi14: float | None
    macd_hist: float | None
    macd_hist_prev: float | None
    atr_pct: float | None
    pct_b: float | None
    high_20d: float | None          # highest high of the 20 sessions before yesterday
    avg_volume_20d: float
    avg_dollar_volume_20d: float
    prev_day_rel_volume: float | None
    premarket_volume: float | None
    premarket_rel_volume: float | None
    spread_pct: float | None
    quote_age_minutes: float
    history_days: int
    max_daily_jump_ratio: float     # largest close-to-close ratio either way, for split detection
    market_cap: float | None
    float_shares: float | None
    catalyst_score: float
    catalyst_category: str | None
    catalyst_headline: str | None
    bearish_news: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def compute(sec: Security, bars: list[Bar], snap: Snapshot, cat: CatalystAssessment, now: dt.datetime,
            premarket_fraction: float) -> Features:
    if not bars:
        raise ValueError(f"{sec.symbol}: no daily bars")
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    vols = [b.volume for b in bars]
    prev_close = closes[-1]
    ref = snap.premarket_price or snap.last_price or prev_close

    v20 = vols[-20:]
    avg_vol = sum(v20) / len(v20)
    dollar = [b.close * b.volume for b in bars[-20:]]
    avg_dollar = sum(dollar) / len(dollar)
    sma20_now = ind.sma(closes, 20)
    sma20_then = ind.sma(closes[:-5], 20) if len(closes) >= 25 else None
    macd = ind.macd_hist(closes)
    a = ind.atr(highs, lows, closes)
    prior_avg = sum(vols[-21:-1]) / len(vols[-21:-1]) if len(vols) > 1 else None
    pm_rel = None
    if snap.premarket_volume is not None and avg_vol > 0:
        pm_rel = snap.premarket_volume / (avg_vol * premarket_fraction)
    jumps = [max(b / a_, a_ / b) for a_, b in zip(closes[:-1], closes[1:]) if a_ > 0 and b > 0]

    return Features(
        symbol=sec.symbol,
        reference_price=round(ref, 4),
        prev_close=prev_close,
        gap_pct=(ref / prev_close - 1) * 100 if prev_close > 0 else 0.0,
        ret_1d=ind.pct_return(closes, 1),
        ret_5d=ind.pct_return(closes, 5),
        ret_20d=ind.pct_return(closes, 20),
        sma20=sma20_now,
        sma50=ind.sma(closes, 50),
        sma200=ind.sma(closes, 200),
        sma20_slope_pct=(sma20_now / sma20_then - 1) * 100 if sma20_now and sma20_then else None,
        rsi14=ind.rsi(closes),
        macd_hist=macd[0] if macd else None,
        macd_hist_prev=macd[1] if macd else None,
        atr_pct=a / prev_close * 100 if a and prev_close else None,
        pct_b=ind.bollinger_pct_b(closes),
        high_20d=max(highs[-21:-1]) if len(highs) > 21 else None,
        avg_volume_20d=avg_vol,
        avg_dollar_volume_20d=avg_dollar,
        prev_day_rel_volume=vols[-1] / prior_avg if prior_avg else None,
        premarket_volume=snap.premarket_volume,
        premarket_rel_volume=pm_rel,
        spread_pct=snap.spread_percent,
        quote_age_minutes=max(0.0, (now - snap.as_of).total_seconds() / 60),
        history_days=len(bars),
        max_daily_jump_ratio=max(jumps) if jumps else 1.0,
        market_cap=sec.market_cap,
        float_shares=sec.float_shares,
        catalyst_score=cat.score,
        catalyst_category=cat.primary.category if cat.primary else None,
        catalyst_headline=cat.primary.headline if cat.primary else None,
        bearish_news=tuple(f"{c.category}: {c.headline}" for c in cat.negatives),
    )
