"""SQLite persistence: recommendations, alerts, outcomes and the audit trail.

The database file is committed back to the repo by the scheduled workflows,
so history survives between runs. Schema changes go in MIGRATIONS as new
entries; never edit an applied one.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

MIGRATIONS: list[str] = [
    # 1: initial schema
    """
    CREATE TABLE runs (
        id            TEXT PRIMARY KEY,
        trade_date    TEXT NOT NULL,
        started_at    TEXT NOT NULL,
        finished_at   TEXT,
        market_status TEXT,
        provider      TEXT NOT NULL,
        status        TEXT NOT NULL,          -- running, recommended, no_trade, failed, skipped
        reason        TEXT,
        universe_size INTEGER,
        config_json   TEXT NOT NULL
    );
    CREATE TABLE candidates (
        run_id        TEXT NOT NULL REFERENCES runs(id),
        symbol        TEXT NOT NULL,
        passed        INTEGER NOT NULL,
        signal_score  REAL,
        confidence    REAL,
        rejections    TEXT,                   -- JSON list of filter reasons
        factors_json  TEXT,                   -- JSON per-factor sub-scores and inputs
        PRIMARY KEY (run_id, symbol)
    );
    CREATE TABLE recommendations (
        id               TEXT PRIMARY KEY,
        run_id           TEXT NOT NULL REFERENCES runs(id),
        trade_date       TEXT NOT NULL UNIQUE, -- at most one pick per trading day
        symbol           TEXT NOT NULL,
        company          TEXT,
        created_at       TEXT NOT NULL,
        reference_price  REAL NOT NULL,        -- pre-market price when generated
        entry_low        REAL NOT NULL,
        entry_high       REAL NOT NULL,
        max_entry        REAL NOT NULL,
        target           REAL NOT NULL,
        stop             REAL NOT NULL,
        planned_entry_at TEXT NOT NULL,
        planned_exit_at  TEXT NOT NULL,
        signal_score     REAL NOT NULL,
        confidence       REAL NOT NULL,
        catalyst         TEXT,
        rationale        TEXT,
        risks            TEXT,
        inputs_json      TEXT NOT NULL,        -- everything the decision used, for audit
        status           TEXT NOT NULL DEFAULT 'open'  -- open, closed, cancelled
    );
    CREATE TABLE entries (
        recommendation_id  TEXT PRIMARY KEY REFERENCES recommendations(id),
        recorded_at        TEXT NOT NULL,
        official_open      REAL,
        first_trade        REAL,
        bid                REAL,
        ask                REAL,
        hypothetical_fill  REAL,
        user_fill          REAL,               -- only if the user reports one
        within_entry_range INTEGER
    );
    CREATE TABLE alerts (
        recommendation_id TEXT NOT NULL REFERENCES recommendations(id),
        kind              TEXT NOT NULL,       -- exit_t60, exit_t30, exit_t0, target, stop
        due_at            TEXT,
        sent_at           TEXT,
        message           TEXT,
        PRIMARY KEY (recommendation_id, kind)
    );
    CREATE TABLE outcomes (
        recommendation_id TEXT PRIMARY KEY REFERENCES recommendations(id),
        closed_at         TEXT NOT NULL,
        exit_price        REAL NOT NULL,
        exit_reason       TEXT NOT NULL,       -- target, stop, time
        return_pct        REAL NOT NULL,
        max_favorable_pct REAL,
        max_adverse_pct   REAL,
        holding_minutes   INTEGER,
        user_exit_price   REAL
    );
    CREATE TABLE news (
        id           TEXT PRIMARY KEY,
        symbols      TEXT NOT NULL,
        headline     TEXT NOT NULL,
        source       TEXT,
        url          TEXT,
        published_at TEXT NOT NULL,
        provider     TEXT,
        fetched_at   TEXT NOT NULL
    );
    CREATE TABLE system_logs (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     TEXT NOT NULL,
        level  TEXT NOT NULL,
        event  TEXT NOT NULL,
        detail TEXT
    );
    """,
]


def _iso(t: dt.datetime) -> str:
    if t.tzinfo is None:
        raise ValueError("datetimes must be timezone-aware")
    return t.astimezone(dt.timezone.utc).isoformat()


def _from_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    def close(self) -> None:
        self.db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self.db:
            yield self.db

    def _migrate(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        for i, sql in enumerate(MIGRATIONS[version:], start=version + 1):
            with self.db:
                self.db.executescript(sql)
                self.db.execute(f"PRAGMA user_version = {i}")

    @property
    def schema_version(self) -> int:
        return self.db.execute("PRAGMA user_version").fetchone()[0]

    # ── runs and audit ─────────────────────────────────────────
    def start_run(self, trade_date: dt.date, started_at: dt.datetime, provider: str, config: dict[str, Any]) -> str:
        run_id = str(uuid.uuid4())
        with self.tx() as db:
            db.execute(
                "INSERT INTO runs (id, trade_date, started_at, provider, status, config_json) VALUES (?,?,?,?,?,?)",
                (run_id, trade_date.isoformat(), _iso(started_at), provider, "running", json.dumps(config)),
            )
        return run_id

    def finish_run(self, run_id: str, finished_at: dt.datetime, status: str, reason: str | None = None,
                   market_status: str | None = None, universe_size: int | None = None) -> None:
        with self.tx() as db:
            db.execute(
                "UPDATE runs SET finished_at=?, status=?, reason=?, market_status=?, universe_size=? WHERE id=?",
                (_iso(finished_at), status, reason, market_status, universe_size, run_id),
            )

    def log(self, level: str, event: str, detail: Any = None, ts: dt.datetime | None = None) -> None:
        ts = ts or dt.datetime.now(dt.timezone.utc)
        with self.tx() as db:
            db.execute(
                "INSERT INTO system_logs (ts, level, event, detail) VALUES (?,?,?,?)",
                (_iso(ts), level, event, None if detail is None else json.dumps(detail, default=str)),
            )

    # ── recommendations ────────────────────────────────────────
    def has_recommendation(self, trade_date: dt.date) -> bool:
        row = self.db.execute("SELECT 1 FROM recommendations WHERE trade_date=?", (trade_date.isoformat(),)).fetchone()
        return row is not None

    def save_recommendation(self, run_id: str, trade_date: dt.date, rec: dict[str, Any]) -> str:
        """Store a recommendation. Raises sqlite3.IntegrityError if the day already has one."""
        rec_id = str(uuid.uuid4())
        with self.tx() as db:
            db.execute(
                """INSERT INTO recommendations (id, run_id, trade_date, symbol, company, created_at,
                   reference_price, entry_low, entry_high, max_entry, target, stop,
                   planned_entry_at, planned_exit_at, signal_score, confidence,
                   catalyst, rationale, risks, inputs_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec_id, run_id, trade_date.isoformat(), rec["symbol"], rec.get("company"),
                 _iso(rec["created_at"]), rec["reference_price"], rec["entry_low"], rec["entry_high"],
                 rec["max_entry"], rec["target"], rec["stop"], _iso(rec["planned_entry_at"]),
                 _iso(rec["planned_exit_at"]), rec["signal_score"], rec["confidence"],
                 rec.get("catalyst"), rec.get("rationale"), rec.get("risks"),
                 json.dumps(rec.get("inputs", {}), default=str)),
            )
        return rec_id

    def get_recommendation(self, rec_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM recommendations WHERE id=?", (rec_id,)).fetchone()

    def open_recommendations(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM recommendations WHERE status='open' ORDER BY trade_date").fetchall()

    # ── alerts ─────────────────────────────────────────────────
    def schedule_alert(self, rec_id: str, kind: str, due_at: dt.datetime | None) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR IGNORE INTO alerts (recommendation_id, kind, due_at) VALUES (?,?,?)",
                (rec_id, kind, None if due_at is None else _iso(due_at)),
            )

    def due_alerts(self, now: dt.datetime) -> list[sqlite3.Row]:
        return self.db.execute(
            """SELECT a.*, r.symbol FROM alerts a JOIN recommendations r ON r.id = a.recommendation_id
               WHERE a.sent_at IS NULL AND a.due_at IS NOT NULL AND a.due_at <= ? AND r.status = 'open'
               ORDER BY a.due_at""",
            (_iso(now),),
        ).fetchall()

    def mark_alert_sent(self, rec_id: str, kind: str, sent_at: dt.datetime, message: str) -> bool:
        """Mark sent. Returns False if it was already sent, so callers never double-send."""
        with self.tx() as db:
            cur = db.execute(
                "UPDATE alerts SET sent_at=?, message=? WHERE recommendation_id=? AND kind=? AND sent_at IS NULL",
                (_iso(sent_at), message, rec_id, kind),
            )
        return cur.rowcount == 1

    # ── news ───────────────────────────────────────────────────
    def save_news(self, items, fetched_at: dt.datetime) -> None:
        with self.tx() as db:
            db.executemany(
                "INSERT OR IGNORE INTO news (id, symbols, headline, source, url, published_at, provider, fetched_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(n.id, ",".join(n.symbols), n.headline, n.source, n.url, _iso(n.published_at), n.provider,
                  _iso(fetched_at)) for n in items],
            )
