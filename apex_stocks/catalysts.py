"""Deterministic catalyst detection from news headlines and summaries.

Keyword rules, not an LLM, so every classification can be audited. An LLM
may later add a second opinion, but never replaces this.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Sequence

from .models import NewsItem

# (category, materiality 0-1, direction +1 bullish / -1 bearish / 0 unclear, pattern)
_RULES: list[tuple[str, float, int, str]] = [
    ("offering", 0.9, -1, r"\b(public offering|registered direct|private placement|at-the-market|atm program|"
                          r"prices? (an? )?offering|shelf registration|dilut)"),
    ("bankruptcy", 1.0, -1, r"\b(bankruptcy|chapter 11|going concern|delist)"),
    ("reverse_split", 0.8, -1, r"\breverse (stock )?split\b"),
    ("legal", 0.6, -1, r"\b(lawsuit|class action|subpoena|sec investigation|fraud|indict)"),
    ("downgrade", 0.6, -1, r"\bdowngrade"),
    ("guidance_cut", 0.8, -1, r"\b(lowers|cuts|reduces|withdraws) (its )?(full[- ]year |fy\d* )?(guidance|outlook|forecast)"),
    ("miss", 0.7, -1, r"\b(misses|missed|below) (estimates|expectations|consensus)"),
    ("fda_negative", 1.0, -1, r"\b(complete response letter|crl\b|clinical hold|fails? to meet|did not meet)"),
    ("fda_positive", 1.0, 1, r"\b(fda (approval|approves|clears|grants)|breakthrough therapy|fast track|"
                             r"met (its )?primary endpoint|positive (topline|phase|trial|data))"),
    ("acquisition", 0.9, 1, r"\b(to be acquired|definitive (merger )?agreement|acquisition of|agrees to acquire|"
                            r"buyout|takeover|tender offer)"),
    ("guidance_raise", 0.8, 1, r"\b(raises|boosts|increases|reaffirms) (its )?(full[- ]year |fy\d* )?(guidance|outlook|forecast)"),
    ("beat", 0.75, 1, r"\b(beats?|tops?|exceeds?|above) (analyst |wall street )?(estimates|expectations|consensus)|record (quarterly )?revenue"),
    ("earnings", 0.5, 0, r"\b(earnings|quarterly results|q[1-4] (results|revenue)|fiscal (year|quarter))"),
    ("contract", 0.75, 1, r"\b(contract|award(ed)?|purchase order|supply agreement|government order)\b"),
    ("partnership", 0.6, 1, r"\b(partnership|collaboration|strategic alliance|licens(e|ing) agreement|joint venture)"),
    ("upgrade", 0.6, 1, r"\b(upgrade[sd]?|initiates? coverage .*buy|price target raised|raises price target)"),
    ("product", 0.45, 1, r"\b(launch(es)?|unveils|introduces|commercial availability)"),
    ("insider_buy", 0.5, 1, r"\b(insider (buy|purchase)|director (buys|purchases)|ceo (buys|purchases))"),
]
_COMPILED = [(c, m, d, re.compile(p, re.I)) for c, m, d, p in _RULES]


@dataclass(frozen=True)
class Catalyst:
    category: str
    materiality: float      # 0-1
    direction: int          # +1, 0, -1
    news_id: str
    headline: str
    source: str
    url: str
    published_at: dt.datetime
    age_hours: float

    @property
    def strength(self) -> float:
        """Materiality decayed by age: full within 6h, half by 24h."""
        decay = 1.0 if self.age_hours <= 6 else max(0.0, 1 - (self.age_hours - 6) / 36)
        return self.materiality * decay


def classify(item: NewsItem, now: dt.datetime) -> Catalyst | None:
    text = f"{item.headline} {item.summary}"
    for category, materiality, direction, rx in _COMPILED:
        if rx.search(text):
            age = max(0.0, (now - item.published_at).total_seconds() / 3600)
            return Catalyst(category, materiality, direction, item.id, item.headline, item.source, item.url,
                            item.published_at, round(age, 2))
    return None


@dataclass(frozen=True)
class CatalystAssessment:
    score: float                     # 0-1, bullish strength
    primary: Catalyst | None         # strongest bullish catalyst
    negatives: tuple[Catalyst, ...]  # bearish items found
    all: tuple[Catalyst, ...]

    @property
    def has_bullish(self) -> bool:
        return self.primary is not None


def assess(items: Sequence[NewsItem], now: dt.datetime) -> CatalystAssessment:
    found = [c for c in (classify(n, now) for n in items) if c]
    bullish = sorted((c for c in found if c.direction > 0), key=lambda c: c.strength, reverse=True)
    bearish = tuple(c for c in found if c.direction < 0)
    score = bullish[0].strength if bullish else 0.0
    # Several independent bullish items add a little conviction.
    score = min(1.0, score + 0.1 * max(0, len(bullish) - 1))
    # A material bearish item in the same window outweighs the good news.
    if bearish:
        score *= max(0.0, 1 - max(c.strength for c in bearish))
    return CatalystAssessment(round(score, 4), bullish[0] if bullish else None, bearish, tuple(found))
