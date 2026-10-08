"""Transparent 0-100 scoring. Every number traces back to a named factor.

Signal score: how attractive the setup is (weighted sub-scores).
Confidence score: how far the data behind it can be trusted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import Config
from .features import Features


def lin(x: float | None, lo: float, hi: float) -> float:
    """0 at lo, 1 at hi, clipped. Works for hi < lo (decreasing)."""
    if x is None:
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return min(1.0, max(0.0, (x - lo) / (hi - lo)))


def band(x: float | None, ramp_lo: float, full_lo: float, full_hi: float, ramp_hi: float, tail: float = 0.0) -> float:
    """1 inside [full_lo, full_hi], ramping to 0 below and down to ``tail`` above."""
    if x is None:
        return 0.0
    if x < full_lo:
        return lin(x, ramp_lo, full_lo)
    if x <= full_hi:
        return 1.0
    return tail + (1 - tail) * lin(x, ramp_hi, full_hi)


@dataclass
class Score:
    signal: float
    confidence: float
    components: dict[str, float]          # category -> 0-1
    details: dict[str, dict[str, float]]  # category -> factor -> 0-1
    capped_by_no_catalyst: bool = False
    quality_issues: list[str] = field(default_factory=list)


def _weighted(parts: dict[str, tuple[float, float]]) -> tuple[float, dict[str, float]]:
    total = sum(w for _, w in parts.values())
    return sum(v * w for v, w in parts.values()) / total, {k: round(v, 4) for k, (v, _) in parts.items()}


def momentum(f: Features) -> tuple[float, dict]:
    return _weighted({
        "premarket_gap": (band(f.gap_pct, 0, 3, 15, 40, tail=0.2), 0.35),
        "return_5d": (band(f.ret_5d, -5, 3, 20, 50, tail=0.2), 0.20),
        "return_20d": (band(f.ret_20d, -10, 5, 40, 100, tail=0.3), 0.15),
        "above_averages": ((f.sma20 is not None and f.prev_close > f.sma20) * 0.5
                           + (f.sma50 is not None and f.prev_close > f.sma50) * 0.5, 0.15),
        "trend_slope": (lin(f.sma20_slope_pct, 0, 5), 0.15),
    })


def volume(f: Features) -> tuple[float, dict]:
    return _weighted({
        "premarket_rel_volume": (lin(f.premarket_rel_volume, 1, 6), 0.50),
        "prev_day_rel_volume": (lin(f.prev_day_rel_volume, 1, 3), 0.25),
        "dollar_volume": (lin(math.log10(max(f.avg_dollar_volume_20d, 1)), 7, 8.5), 0.25),
    })


def technical(f: Features) -> tuple[float, dict]:
    macd_ok = 0.0
    if f.macd_hist is not None:
        macd_ok = (f.macd_hist > 0) * 0.6 + (f.macd_hist_prev is not None and f.macd_hist > f.macd_hist_prev) * 0.4
    breakout = 0.0
    if f.high_20d:
        breakout = lin(f.reference_price / f.high_20d, 0.95, 1.02)
    return _weighted({
        "rsi": (band(f.rsi14, 35, 50, 70, 85, tail=0.2), 0.30),
        "macd": (macd_ok, 0.25),
        "breakout_20d": (breakout, 0.25),
        "bollinger": (band(f.pct_b, 0.2, 0.6, 1.0, 1.3, tail=0.3), 0.20),
    })


def risk(f: Features) -> tuple[float, dict]:
    return _weighted({
        "spread": (lin(f.spread_pct, 1.0, 0.1), 0.35),
        "atr_range": (band(f.atr_pct, 1.5, 3, 8, 15, tail=0.1), 0.25),
        "market_cap": (lin(math.log10(f.market_cap), 7.7, 9) if f.market_cap else 0.5, 0.20),
        "float": (lin(math.log10(f.float_shares), 6.7, 7.7) if f.float_shares else 0.5, 0.20),
    })


def confidence(f: Features, cfg: Config) -> tuple[float, list[str]]:
    """Trust in the data, 0-100.

    Inputs a provider structurally cannot supply (no pre-market spreads or
    fundamentals on the free Alpaca plan) are left out and the remaining
    weights rescaled, rather than counted as bad data. They are still listed
    as issues so the message says what was not checked.
    """
    issues: list[str] = []
    fresh_full = cfg.EXPECTED_DATA_DELAY_MINUTES + 5
    parts = {
        "fresh_quote": (lin(f.quote_age_minutes, cfg.MAX_QUOTE_AGE_MINUTES, fresh_full), 0.25),
        "premarket_liquidity": (lin((f.premarket_volume or 0) * f.reference_price, 50_000, 1_000_000), 0.30),
        "history": (lin(f.history_days, cfg.MIN_HISTORY_DAYS, 200), 0.15),
    }
    if f.spread_pct is not None:
        parts["tight_spread"] = (lin(f.spread_pct, cfg.MAX_SPREAD_PERCENT * 1.5, 0.2), 0.20)
    else:
        issues.append("no pre-market bid/ask quote; spread not checked")
    if f.market_cap is not None or f.float_shares is not None:
        parts["fundamentals_known"] = ((f.market_cap is not None) * 0.5 + (f.float_shares is not None) * 0.5, 0.10)
    else:
        issues.append("market cap and float unknown")
    if f.premarket_volume is None:
        issues.append("pre-market volume unavailable")
    elif f.premarket_volume * f.reference_price < 50_000:
        issues.append("thin pre-market trading; pre-market price may not hold at the open")
    if f.quote_age_minutes > fresh_full + (cfg.MAX_QUOTE_AGE_MINUTES - fresh_full) / 2:
        issues.append(f"latest pre-market trade is {f.quote_age_minutes:.0f} minutes old")
    score, _ = _weighted(parts)
    return round(score * 100, 1), issues


def score(f: Features, cfg: Config) -> Score:
    comps, details = {}, {}
    for name, fn in (("momentum", momentum), ("volume", volume), ("technical", technical), ("risk", risk)):
        comps[name], details[name] = fn(f)
    comps["catalyst"], details["catalyst"] = f.catalyst_score, {"strength": f.catalyst_score}
    weights = {"momentum": cfg.WEIGHT_MOMENTUM, "volume": cfg.WEIGHT_VOLUME, "catalyst": cfg.WEIGHT_CATALYST,
               "technical": cfg.WEIGHT_TECHNICAL, "risk": cfg.WEIGHT_RISK}
    signal = sum(comps[k] * w for k, w in weights.items())
    capped = False
    if f.catalyst_category is None and signal > cfg.NO_CATALYST_SCORE_CAP:
        signal, capped = cfg.NO_CATALYST_SCORE_CAP, True
    conf, issues = confidence(f, cfg)
    return Score(round(signal, 1), conf, {k: round(v, 4) for k, v in comps.items()}, details, capped, issues)
