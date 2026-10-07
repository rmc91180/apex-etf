import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from apex_stocks.config import Config
from apex_stocks.market_calendar import (
    ET, CalendarUnavailable, ExitPlanError, MarketCalendar, format_dual,
)

cal = MarketCalendar()
CFG = Config()


def et(y, m, d, h=9, mi=30):
    return dt.datetime(y, m, d, h, mi, tzinfo=ET)


def test_weekends_and_holidays_are_not_trading_days():
    assert cal.is_trading_day(dt.date(2026, 10, 7))       # Wednesday
    assert not cal.is_trading_day(dt.date(2026, 10, 10))  # Saturday
    assert not cal.is_trading_day(dt.date(2026, 11, 26))  # Thanksgiving
    assert not cal.is_trading_day(dt.date(2026, 12, 25))  # Christmas
    assert not cal.is_trading_day(dt.date(2027, 3, 26))   # Good Friday


def test_session_times_come_from_calendar_including_early_close():
    s = cal.session(dt.date(2026, 11, 27))  # day after Thanksgiving
    assert s.open == et(2026, 11, 27, 9, 30)
    assert s.close == et(2026, 11, 27, 13, 0)
    assert s.early_close
    assert not cal.session(dt.date(2026, 10, 7)).early_close


def test_daylight_saving_shift_keeps_eastern_open():
    before = cal.session(dt.date(2026, 3, 6))  # EST
    after = cal.session(dt.date(2026, 3, 9))   # EDT
    assert before.open.time() == after.open.time() == dt.time(9, 30)
    utc = dt.timezone.utc
    assert before.open.astimezone(utc).hour == 14
    assert after.open.astimezone(utc).hour == 13


def test_next_session_skips_closures():
    assert cal.next_session(dt.date(2026, 11, 26)).date == dt.date(2026, 11, 27)
    assert cal.next_session(dt.date(2026, 10, 9), include_today=False).date == dt.date(2026, 10, 12)


def test_market_status():
    assert cal.market_status(et(2026, 10, 7, 8, 0)) == "pre-market"
    assert cal.market_status(et(2026, 10, 7, 10, 0)) == "open"
    assert cal.market_status(et(2026, 10, 7, 16, 30)) == "after-hours"
    assert cal.market_status(et(2026, 10, 10, 12, 0)) == "closed"
    assert cal.market_status(et(2026, 11, 27, 13, 30)) == "after-hours"


def test_outside_calendar_range_fails_instead_of_guessing():
    with pytest.raises(CalendarUnavailable):
        cal.is_trading_day(dt.date(2090, 1, 2))


# ── exit planning ──────────────────────────────────────────────

def test_default_exit_is_next_session_before_close():
    assert cal.plan_exit(et(2026, 10, 7), CFG) == et(2026, 10, 8, 15, 30)


def test_friday_entry_exits_same_day_because_monday_exceeds_48h():
    assert cal.plan_exit(et(2026, 10, 9), CFG) == et(2026, 10, 9, 15, 30)


def test_entry_before_holiday_falls_back_within_limit():
    # Wed before Thanksgiving: Friday's early-close exit (12:30) is 51h away.
    assert cal.plan_exit(et(2026, 11, 25), CFG) == et(2026, 11, 25, 15, 30)


def test_entry_on_early_close_day_uses_early_close():
    assert cal.plan_exit(et(2026, 11, 27), CFG) == et(2026, 11, 27, 12, 30)


def test_thursday_entry_exits_friday():
    assert cal.plan_exit(et(2026, 10, 8), CFG) == et(2026, 10, 9, 15, 30)


def test_same_session_offset():
    cfg = Config(EXIT_SESSION_OFFSET=0)
    assert cal.plan_exit(et(2026, 10, 7, 10, 0), cfg) == et(2026, 10, 7, 15, 30)


def test_two_session_offset_is_capped_at_48h():
    cfg = Config(EXIT_SESSION_OFFSET=2)
    # Mon 9:30 -> Wed 15:30 would be 54h, so Tue 15:30 is used.
    assert cal.plan_exit(et(2026, 10, 5), cfg) == et(2026, 10, 6, 15, 30)
    # Mon 15:45 -> Wed 15:30 is 47h45m, allowed.
    assert cal.plan_exit(et(2026, 10, 5, 15, 45), cfg) == et(2026, 10, 7, 15, 30)


def test_entry_outside_session_is_rejected():
    with pytest.raises(ExitPlanError):
        cal.plan_exit(et(2026, 10, 7, 8, 0), CFG)
    with pytest.raises(ExitPlanError):
        cal.plan_exit(et(2026, 10, 10, 10, 0), CFG)


def test_no_valid_exit_raises():
    with pytest.raises(ExitPlanError):
        cal.plan_exit(et(2026, 10, 7, 15, 45), Config(EXIT_SESSION_OFFSET=0))


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValueError):
        cal.plan_exit(dt.datetime(2026, 10, 7, 9, 30), CFG)


@pytest.mark.parametrize("offset", [0, 1, 2, 3])
def test_holding_period_never_exceeds_limit_over_a_year(offset):
    cfg = Config(EXIT_SESSION_OFFSET=offset)
    limit = dt.timedelta(hours=cfg.MAX_HOLDING_HOURS)
    d, checked = dt.date(2026, 1, 1), 0
    while d < dt.date(2027, 1, 1):
        s = cal.session(d)
        if s:
            exit_at = cal.plan_exit(s.open, cfg)
            assert s.open < exit_at <= s.open + limit
            exit_session = cal.session(exit_at.date())
            assert exit_session.open < exit_at < exit_session.close
            checked += 1
        d += dt.timedelta(days=1)
    assert checked > 240


def test_alert_times_before_exit():
    exit_at = et(2026, 10, 8, 15, 30)
    alerts = cal.alert_times(exit_at, CFG)
    assert [(a.lead_minutes, a.at) for a in alerts] == [
        (60, et(2026, 10, 8, 14, 30)), (30, et(2026, 10, 8, 15, 0)), (0, exit_at)]


def test_alert_times_on_early_close():
    exit_at = cal.plan_exit(et(2026, 11, 27), CFG)
    assert [a.at.time() for a in cal.alert_times(exit_at, CFG)] == [
        dt.time(11, 30), dt.time(12, 0), dt.time(12, 30)]


def test_format_dual_shows_user_timezone():
    t = et(2026, 10, 8, 15, 30)
    assert format_dual(t, "America/New_York") == "Thu Oct 8, 3:30 PM ET"
    assert format_dual(t, "America/Los_Angeles") == "Thu Oct 8, 3:30 PM ET (12:30 PM PDT)"
    assert format_dual(t.astimezone(ZoneInfo("UTC")), "Asia/Jerusalem") == "Thu Oct 8, 3:30 PM ET (10:30 PM IDT)"
