"""Which securities are eligible at all, before any scoring."""
from __future__ import annotations

import re

from .config import Config
from .models import Security

ALLOWED_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}  # AMEX = NYSE American

_NAME_RULES: list[tuple[str, re.Pattern]] = [
    ("warrant", re.compile(r"\bwarrants?\b|\bwts?\b", re.I)),
    ("right", re.compile(r"\brights?\b", re.I)),
    ("unit", re.compile(r"\bunits?\b", re.I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpfd\b|\bdepositary shares?\b|\b\d+(\.\d+)?% .*(notes?|series)", re.I)),
    ("etn", re.compile(r"\betns?\b|exchange traded notes?", re.I)),
    ("etf", re.compile(r"\betfs?\b|\bproshares\b|\bdirexion\b|\bishares\b|\bspdr\b|\bvaneck\b|\binvesco\b.*\btrust\b"
                       r"|\b(2|3)x\b|\bleveraged\b|\binverse\b|\bindex fund\b", re.I)),
    ("fund", re.compile(r"\bfund\b|\btrust\b.*\b(income|municipal|bond)\b|\bclosed[- ]end\b", re.I)),
    ("spac", re.compile(r"\bacquisition corp|\bacquisition co\b|\bacquisition (limited|ltd|inc)|\bcapital acquisition\b"
                        r"|\bblank check\b", re.I)),
    ("notes", re.compile(r"\bnotes? due\b|\bdebentures?\b|\bsenior notes?\b", re.I)),
]


_NASDAQ_FIFTH_LETTER = {"W": "warrant", "U": "unit", "R": "right"}


def classify_asset(symbol: str, name: str, exchange: str = "") -> str:
    """Best-effort asset type from a symbol, company name and exchange.

    Used for providers that don't label ETFs, warrants, units and so on.
    Anything not recognised is treated as common stock.
    """
    if exchange == "NASDAQ" and len(symbol) == 5 and symbol[-1] in _NASDAQ_FIFTH_LETTER:
        return _NASDAQ_FIFTH_LETTER[symbol[-1]]
    if re.search(r"[./](WS|W|U|R|RT|WT)$", symbol) or re.search(r"\.P[A-Z]?$|-P[A-Z]?$", symbol):
        return {"WS": "warrant", "W": "warrant", "WT": "warrant", "U": "unit", "R": "right", "RT": "right"}.get(
            re.split(r"[./]", symbol)[-1], "preferred")
    for kind, rx in _NAME_RULES:
        if rx.search(name):
            return kind
    return "common"


def static_rejections(sec: Security, cfg: Config) -> list[str]:
    """Reasons a security is ineligible regardless of today's data."""
    reasons = []
    allowed_types = {"common", "spac"} if cfg.ALLOW_SPACS else {"common"}
    if sec.asset_type not in allowed_types:
        reasons.append(f"asset type {sec.asset_type}")
    if sec.exchange not in ALLOWED_EXCHANGES:
        reasons.append(f"exchange {sec.exchange}")
    if not sec.tradable:
        reasons.append("not tradable")
    if sec.halted:
        reasons.append("trading halted")
    return reasons
