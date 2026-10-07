"""U.S. equity market calendar and trade-timing rules.

All market times are computed in America/New_York from the NYSE calendar
(holidays, early closes and DST included). Nothing here assumes 9:30 or
16:00; session times always come from the calendar.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from .config import Config

ET = ZoneInfo("America/New_York")
REGULAR_CLOSE = dt.time(16, 0)


class CalendarUnavailable(RuntimeError):
    """The calendar cannot answer for this date. Callers must not guess."""


class ExitPlanError(ValueError):
    """No exit time satisfies the holding-period rules for this entry."""


@dataclass(frozen=True)
class Session:
    date: dt.date
    open: dt.datetime   # tz-aware, America/New_York
    close: dt.datetime  # tz-aware, America/New_York

    @property
    def early_close(self) -> bool:
        return self.close.timetz().replace(tzinfo=None) < REGULAR_CLOSE


@dataclass(frozen=True)
class AlertTime:
    lead_minutes: int
    at: dt.datetime  # tz-aware, America/New_York


@lru_cache(maxsize=1)
def _xnys():
    return xcals.get_calendar("XNYS")


def _to_et(ts: pd.Timestamp) -> dt.datetime:
    return ts.tz_convert(ET).to_pydatetime()


def _require_aware(t: dt.datetime) -> dt.datetime:
    if t.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return t.astimezone(ET)


class MarketCalendar:
    def __init__(self, cal=None):
        self._cal = cal or _xnys()

    # ── sessions ───────────────────────────────────────────────
    def _check_range(self, d: dt.date) -> None:
        first = self._cal.first_session.date()
        last = self._cal.last_session.date()
        if not first <= d <= last:
            raise CalendarUnavailable(f"{d} is outside the calendar's range {first}..{last}")

    def is_trading_day(self, d: dt.date) -> bool:
        self._check_range(d)
        return bool(self._cal.is_session(pd.Timestamp(d)))

    def session(self, d: dt.date) -> Session | None:
        if not self.is_trading_day(d):
            return None
        ts = pd.Timestamp(d)
        return Session(d, _to_et(self._cal.session_open(ts)), _to_et(self._cal.session_close(ts)))

    def next_session(self, d: dt.date, include_today: bool = True) -> Session:
        """The first session on or after ``d`` (strictly after if include_today is False)."""
        cur = d if include_today else d + dt.timedelta(days=1)
        for _ in range(15):  # no U.S. market closure has lasted this long since 2001
            s = self.session(cur)
            if s:
                return s
            cur += dt.timedelta(days=1)
        raise CalendarUnavailable(f"no session within 15 days of {d}")

    def sessions_from(self, d: dt.date, count: int) -> list[Session]:
        out = [self.next_session(d)]
        while len(out) < count:
            out.append(self.next_session(out[-1].date, include_today=False))
        return out

    def market_status(self, now: dt.datetime) -> str:
        """One of: closed, pre-market, open, after-hours."""
        now = _require_aware(now)
        s = self.session(now.date())
        if s is None:
            return "closed"
        if now < s.open:
            return "pre-market"
        if now < s.close:
            return "open"
        return "after-hours"

    # ── trade timing ───────────────────────────────────────────
    def plan_exit(self, entry: dt.datetime, cfg: Config) -> dt.datetime:
        """Planned exit for a position entered at ``entry``.

        Exit points are ``EXIT_MINUTES_BEFORE_CLOSE`` before each session's
        close (early closes included). The preferred point is in the session
        ``EXIT_SESSION_OFFSET`` sessions after entry. If that point is more than
        ``MAX_HOLDING_HOURS`` after entry (a weekend or holiday in between), the
        latest earlier point inside the limit is used instead. The holding
        period never exceeds the limit.
        """
        entry = _require_aware(entry)
        entry_session = self.session(entry.date())
        if entry_session is None or not entry_session.open <= entry < entry_session.close:
            raise ExitPlanError(f"entry {entry.isoformat()} is not during a regular session")

        deadline = entry + dt.timedelta(hours=cfg.MAX_HOLDING_HOURS)
        sessions = self.sessions_from(entry.date(), cfg.EXIT_SESSION_OFFSET + 1)
        before = dt.timedelta(minutes=cfg.EXIT_MINUTES_BEFORE_CLOSE)
        for s in reversed(sessions):
            point = s.close - before
            if entry < point <= deadline and point > s.open:
                return point
        raise ExitPlanError(
            f"no exit point between entry {entry.isoformat()} and the "
            f"{cfg.MAX_HOLDING_HOURS:g}h limit"
        )

    def alert_times(self, exit_at: dt.datetime, cfg: Config) -> list[AlertTime]:
        exit_at = _require_aware(exit_at)
        return [AlertTime(m, exit_at - dt.timedelta(minutes=m)) for m in cfg.alert_leads]


def format_dual(t: dt.datetime, user_tz: str) -> str:
    """'Thu Oct 8, 3:30 PM ET' plus the user's local time when it differs."""
    t = _require_aware(t)
    et = t.strftime("%a %b %-d, %-I:%M %p ET")
    local_zone = ZoneInfo(user_tz)
    if local_zone.key == ET.key:
        return et
    local = t.astimezone(local_zone)
    return f"{et} ({local.strftime('%-I:%M %p')} {local.tzname()})"
