import datetime as dt
import sqlite3

import pytest

from apex_stocks.market_calendar import ET
from apex_stocks.storage import MIGRATIONS, Store

NOW = dt.datetime(2026, 10, 7, 8, 15, tzinfo=ET)
DAY = dt.date(2026, 10, 7)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db" / "apex.db")
    yield s
    s.close()


def rec(**over):
    base = dict(symbol="MOCKA", company="Mock Alpha", created_at=NOW, reference_price=8.42,
                entry_low=8.35, entry_high=8.50, max_entry=8.60, target=9.01, stop=8.08,
                planned_entry_at=NOW.replace(hour=9, minute=30),
                planned_exit_at=dt.datetime(2026, 10, 8, 15, 30, tzinfo=ET),
                signal_score=82, confidence=71, inputs={"factors": {"momentum": 0.7}})
    base.update(over)
    return base


def test_migrations_apply_once_and_reopen_cleanly(tmp_path):
    path = tmp_path / "apex.db"
    Store(path).close()
    s = Store(path)
    assert s.schema_version == len(MIGRATIONS)
    s.close()


def test_one_recommendation_per_trading_day(store):
    run = store.start_run(DAY, NOW, "mock", {})
    assert not store.has_recommendation(DAY)
    rec_id = store.save_recommendation(run, DAY, rec())
    assert store.has_recommendation(DAY)
    assert store.get_recommendation(rec_id)["symbol"] == "MOCKA"
    with pytest.raises(sqlite3.IntegrityError):
        store.save_recommendation(run, DAY, rec(symbol="MOCKB"))


def test_alerts_are_due_in_order_and_sent_once(store):
    run = store.start_run(DAY, NOW, "mock", {})
    rec_id = store.save_recommendation(run, DAY, rec())
    t60 = dt.datetime(2026, 10, 8, 14, 30, tzinfo=ET)
    t30 = dt.datetime(2026, 10, 8, 15, 0, tzinfo=ET)
    store.schedule_alert(rec_id, "exit_t30", t30)
    store.schedule_alert(rec_id, "exit_t60", t60)
    store.schedule_alert(rec_id, "exit_t60", t60)  # idempotent

    assert store.due_alerts(t60 - dt.timedelta(minutes=1)) == []
    assert [a["kind"] for a in store.due_alerts(t30)] == ["exit_t60", "exit_t30"]

    assert store.mark_alert_sent(rec_id, "exit_t60", t60, "msg")
    assert not store.mark_alert_sent(rec_id, "exit_t60", t60, "msg")
    assert [a["kind"] for a in store.due_alerts(t30)] == ["exit_t30"]


def test_run_lifecycle_and_logs(store):
    run = store.start_run(DAY, NOW, "mock", {"MAX_STOCK_PRICE": 19.99})
    store.finish_run(run, NOW, "no_trade", reason="no candidate passed", market_status="pre-market",
                     universe_size=8)
    store.log("WARN", "data_quality", {"symbol": "THIN"}, ts=NOW)
    row = store.db.execute("SELECT status, reason FROM runs WHERE id=?", (run,)).fetchone()
    assert tuple(row) == ("no_trade", "no candidate passed")
    assert store.db.execute("SELECT count(*) FROM system_logs").fetchone()[0] == 1


def test_naive_timestamps_rejected(store):
    with pytest.raises(ValueError):
        store.start_run(DAY, dt.datetime(2026, 10, 7, 8, 0), "mock", {})
