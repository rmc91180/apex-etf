"""Strategy and runtime configuration.

Every tunable lives here and can be overridden with an environment variable
of the same name (e.g. ``MAX_STOCK_PRICE=15``). Nothing else in the package
should hard-code a threshold.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import get_type_hints


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    # Universe
    MAX_STOCK_PRICE: float = 19.99
    MIN_STOCK_PRICE: float = 1.00
    MIN_AVG_DOLLAR_VOLUME: float = 10_000_000
    MIN_AVG_SHARE_VOLUME: float = 500_000
    MAX_SPREAD_PERCENT: float = 1.0
    MIN_HISTORY_DAYS: int = 60
    ALLOW_SPACS: bool = False
    # Snapshots older than this are treated as stale and rejected.
    MAX_QUOTE_AGE_MINUTES: int = 30
    # Reject a stock that has already run this far over 5 days without fresh news.
    MAX_5D_RUNUP_NO_CATALYST_PERCENT: float = 40.0
    # Reject pre-market gaps beyond this size (possible bad print, split, or binary event).
    MAX_ABS_GAP_PERCENT: float = 60.0
    # A one-day close-to-close move beyond this ratio in history suggests a split or bad data.
    SPLIT_SUSPECT_RATIO: float = 1.8
    # Typical share of a day's volume that trades before 9:30, used to normalise pre-market volume.
    PREMARKET_VOLUME_FRACTION: float = 0.05
    # How far back to look for catalyst news, in hours before the scan.
    NEWS_LOOKBACK_HOURS: float = 20.0

    # Scoring weights (percent, must sum to 100)
    WEIGHT_MOMENTUM: float = 25.0
    WEIGHT_VOLUME: float = 20.0
    WEIGHT_CATALYST: float = 20.0
    WEIGHT_TECHNICAL: float = 20.0
    WEIGHT_RISK: float = 15.0
    # Without a qualifying catalyst the signal score is capped here (catalyst-first rule).
    NO_CATALYST_SCORE_CAP: float = 65.0

    # Trade plan
    PROFIT_TARGET_PERCENT: float = 7.0
    STOP_LOSS_PERCENT: float = 4.0
    MAX_HOLDING_HOURS: float = 48.0
    # Which session to exit in: 0 = same day as entry, 1 = next trading day.
    EXIT_SESSION_OFFSET: int = 1
    EXIT_MINUTES_BEFORE_CLOSE: int = 30
    # Entry range is reference price +/- this percent; max entry is reference + MAX_CHASE_PERCENT.
    ENTRY_BAND_PERCENT: float = 0.5
    MAX_CHASE_PERCENT: float = 2.0
    ALERT_LEAD_MINUTES: str = "60,30,0"

    # Selection
    MIN_SIGNAL_SCORE: float = 60.0
    MIN_CONFIDENCE_SCORE: float = 50.0

    # Position sizing (informational only; the user places trades)
    ACCOUNT_SIZE: float = 0.0
    MAX_POSITION_RISK_PERCENT: float = 1.0
    # Never suggest more than this share of the stock's average daily volume.
    MAX_POSITION_ADV_PERCENT: float = 1.0

    # Runtime
    MARKET_TIMEZONE: str = "America/New_York"
    USER_TIMEZONE: str = "America/New_York"
    DATA_PROVIDER: str = "mock"
    ALPACA_DATA_FEED: str = "iex"  # iex (free) or sip (paid)
    DB_PATH: str = "data/apex_stocks.db"
    NOTIFIER: str = "console"

    @property
    def alert_leads(self) -> list[int]:
        """Minutes before the planned exit at which to alert, largest first."""
        leads = sorted({int(x) for x in self.ALERT_LEAD_MINUTES.split(",") if x.strip()}, reverse=True)
        if any(m < 0 for m in leads):
            raise ConfigError("ALERT_LEAD_MINUTES must be non-negative")
        return leads

    def validate(self) -> "Config":
        if not 0 < self.MAX_STOCK_PRICE < 20:
            raise ConfigError("MAX_STOCK_PRICE must be above 0 and below 20")
        if self.MIN_STOCK_PRICE >= self.MAX_STOCK_PRICE:
            raise ConfigError("MIN_STOCK_PRICE must be below MAX_STOCK_PRICE")
        if not 0 < self.MAX_HOLDING_HOURS <= 48:
            raise ConfigError("MAX_HOLDING_HOURS must be in (0, 48]")
        if self.EXIT_SESSION_OFFSET < 0:
            raise ConfigError("EXIT_SESSION_OFFSET must be >= 0")
        if self.EXIT_MINUTES_BEFORE_CLOSE < 0:
            raise ConfigError("EXIT_MINUTES_BEFORE_CLOSE must be >= 0")
        if self.PROFIT_TARGET_PERCENT <= 0 or self.STOP_LOSS_PERCENT <= 0:
            raise ConfigError("PROFIT_TARGET_PERCENT and STOP_LOSS_PERCENT must be positive")
        weights = (self.WEIGHT_MOMENTUM, self.WEIGHT_VOLUME, self.WEIGHT_CATALYST,
                   self.WEIGHT_TECHNICAL, self.WEIGHT_RISK)
        if any(w < 0 for w in weights) or abs(sum(weights) - 100) > 1e-6:
            raise ConfigError("scoring weights must be non-negative and sum to 100")
        self.alert_leads  # parses and checks
        return self


def _coerce(name: str, raw: str, typ: type):
    try:
        if typ is bool:
            low = raw.strip().lower()
            if low in ("1", "true", "yes", "on"):
                return True
            if low in ("0", "false", "no", "off"):
                return False
            raise ValueError(raw)
        return typ(raw)
    except ValueError as e:
        raise ConfigError(f"{name}={raw!r} is not a valid {typ.__name__}") from e


def load_config(env: dict[str, str] | None = None) -> Config:
    """Build a Config from defaults overridden by environment variables."""
    env = os.environ if env is None else env
    hints = get_type_hints(Config)
    overrides = {
        f.name: _coerce(f.name, env[f.name], hints[f.name])
        for f in fields(Config)
        if f.name in env and env[f.name] != ""
    }
    return Config(**overrides).validate()
