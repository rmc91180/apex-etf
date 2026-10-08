"""Technical indicators on plain float lists (oldest first).

Each function returns None when there is not enough history, rather than a
value computed from too little data.
"""
from __future__ import annotations

import math
from typing import Sequence


def sma(xs: Sequence[float], n: int) -> float | None:
    if len(xs) < n or n <= 0:
        return None
    return sum(xs[-n:]) / n


def ema_series(xs: Sequence[float], n: int) -> list[float]:
    if len(xs) < n:
        return []
    k = 2 / (n + 1)
    out = [sum(xs[:n]) / n]
    for x in xs[n:]:
        out.append(x * k + out[-1] * (1 - k))
    return out


def pct_return(xs: Sequence[float], n: int) -> float | None:
    """Percent change over the last n periods."""
    if len(xs) < n + 1 or xs[-n - 1] <= 0:
        return None
    return (xs[-1] / xs[-n - 1] - 1) * 100


def rsi(closes: Sequence[float], n: int = 14) -> float | None:
    """Wilder's RSI."""
    if len(closes) < n + 1:
        return None
    gains, losses = [], []
    for a, b in zip(closes[:-1], closes[1:]):
        d = b - a
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    return 100 - 100 / (1 + avg_g / avg_l)


def macd_hist(closes: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float] | None:
    """(latest histogram, previous histogram), both as a percent of price."""
    if len(closes) < slow + signal + 1:
        return None
    ef, es = ema_series(closes, fast), ema_series(closes, slow)
    macd = [f - s for f, s in zip(ef[slow - fast:], es)]
    sig = ema_series(macd, signal)
    hist = [m - s for m, s in zip(macd[signal - 1:], sig)]
    if len(hist) < 2:
        return None
    price = closes[-1]
    return hist[-1] / price * 100, hist[-2] / closes[-2] * 100


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], n: int = 14) -> float | None:
    """Wilder's average true range."""
    if len(closes) < n + 1:
        return None
    trs = [max(h - l, abs(h - pc), abs(l - pc)) for h, l, pc in zip(highs[1:], lows[1:], closes[:-1])]
    a = sum(trs[:n]) / n
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def bollinger_pct_b(closes: Sequence[float], n: int = 20, k: float = 2.0) -> float | None:
    """Position within the bands: 0 = lower band, 1 = upper band."""
    if len(closes) < n:
        return None
    window = closes[-n:]
    mid = sum(window) / n
    sd = math.sqrt(sum((x - mid) ** 2 for x in window) / n)
    if sd == 0:
        return 0.5
    return (closes[-1] - (mid - k * sd)) / (2 * k * sd)
