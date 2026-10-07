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

    # Trade plan
    PROFIT_TARGET_PERCENT: float = 7.0
    STOP_LOSS_PERCENT: float = 4.0
    MAX_HOLDING_HOURS: float = 48.0
    # Which session to exit in: 0 = same day as entry, 1 = next trading day.
    EXIT_SESSION_OFFSET: int = 1
    EXIT_MINUTES_BEFORE_CLOSE: int = 30
    ALERT_LEAD_MINUTES: str = "60,30,0"

    # Selection
    MIN_SIGNAL_SCORE: float = 60.0
    MIN_CONFIDENCE_SCORE: float = 50.0

    # Position sizing (informational only; the user places trades)
    ACCOUNT_SIZE: float = 0.0
    MAX_POSITION_RISK_PERCENT: float = 1.0

    # Runtime
    MARKET_TIMEZONE: str = "America/New_York"
    USER_TIMEZONE: str = "America/New_York"
    DATA_PROVIDER: str = "mock"
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
