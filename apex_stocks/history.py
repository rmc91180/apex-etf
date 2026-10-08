"""Download and cache the historical data a backtest needs.

Everything is cached under a directory (gzip JSON / numpy), so a rerun of
the same period costs no API calls. Daily bars are unadjusted ("raw"): an
adjusted series would rewrite old prices using splits that happened later,
which is a form of look-ahead.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .market_calendar import ET
from .models import NewsItem, Security
from .universe import static_rejections

DAILY_FIELDS = ("o", "h", "l", "c", "v")


def _read(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(data, f)
    tmp.replace(path)


def _et_date(ts: str) -> dt.date:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(ET).date()


class DailyTable:
    """Compact daily OHLCV per symbol: dates as ordinals plus float arrays."""

    def __init__(self, data: dict[str, dict[str, np.ndarray]]):
        self.data = data

    def symbols(self) -> list[str]:
        return list(self.data)

    def window(self, sym: str, start: dt.date, end: dt.date):
        d = self.data.get(sym)
        if d is None:
            return None
        lo = np.searchsorted(d["d"], start.toordinal(), "left")
        hi = np.searchsorted(d["d"], end.toordinal(), "right")
        return {k: v[lo:hi] for k, v in d.items()}

    def on(self, sym: str, day: dt.date) -> dict | None:
        w = self.window(sym, day, day)
        if w is None or len(w["d"]) == 0:
            return None
        return {k: float(v[0]) for k, v in w.items()}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flat = {f"{s}|{k}": v for s, d in self.data.items() for k, v in d.items()}
        np.savez_compressed(path, **flat)

    @classmethod
    def load(cls, path: Path) -> "DailyTable":
        z = np.load(path)
        data: dict[str, dict[str, np.ndarray]] = {}
        for key in z.files:
            s, k = key.rsplit("|", 1)
            data.setdefault(s, {})[k] = z[key]
        return cls(data)


class HistoryCache:
    def __init__(self, root: str | Path, source=None):
        self.root = Path(root)
        self.source = source  # an AlpacaProvider, or None to run from cache only
        self.api_calls_skipped = 0

    def _need_source(self, what: str):
        if self.source is None:
            raise RuntimeError(f"{what} is not cached and no data source was given")
        return self.source

    # ── universe ───────────────────────────────────────────────
    def universe(self) -> list[Security]:
        path = self.root / "universe.json.gz"
        if path.exists():
            return [Security(**s) for s in _read(path)]
        src = self._need_source("universe")
        # Include delisted symbols so the backtest isn't limited to today's survivors.
        secs = src.list_universe("active") + src.list_universe("inactive")
        seen, unique = set(), []
        for s in secs:
            if s.symbol not in seen:
                seen.add(s.symbol)
                unique.append(s)
        secs = unique
        _write(path, [asdict(s) for s in secs])
        return secs

    # ── daily bars ─────────────────────────────────────────────
    def daily(self, symbols: Sequence[str], start: dt.date, end: dt.date, max_price: float,
              min_dollar_volume: float, always_keep: Sequence[str] = ()) -> DailyTable:
        """Daily bars for symbols that were ever cheap and liquid enough to matter in [start, end]."""
        path = self.root / f"daily_{start}_{end}.npz"
        if path.exists():
            return DailyTable.load(path)
        src = self._need_source("daily bars")
        kept: dict[str, dict[str, np.ndarray]] = {}
        syms = list(dict.fromkeys(list(symbols) + list(always_keep)))
        for i in range(0, len(syms), 200):
            raw = src.bars_raw(syms[i:i + 200], "1Day", dt.datetime.combine(start, dt.time(0), ET),
                               dt.datetime.combine(end, dt.time(23, 59), ET), adjustment="raw")
            for sym, bars in raw.items():
                if len(bars) < 25:
                    continue
                arr = {"d": np.array([_et_date(b["t"]).toordinal() for b in bars], dtype=np.int32)}
                for k in DAILY_FIELDS:
                    arr[k] = np.array([b[k] for b in bars], dtype=np.float64)
                dollar = np.convolve(arr["c"] * arr["v"], np.ones(20) / 20, mode="valid")
                if sym in always_keep or (arr["c"].min() <= max_price * 1.25
                                          and dollar.max() >= min_dollar_volume * 0.5):
                    kept[sym] = arr
        table = DailyTable(kept)
        table.save(path)
        return table

    # ── per-day data ───────────────────────────────────────────
    def premarket(self, day: dt.date, symbols: Sequence[str], start: dt.datetime, end: dt.datetime) -> dict[str, list]:
        """5-minute pre-market bars per symbol for one day, cached incrementally."""
        path = self.root / "premarket" / f"{day}.json.gz"
        cached = _read(path) if path.exists() else {"window": [start.isoformat(), end.isoformat()], "bars": {}}
        missing = [s for s in symbols if s not in cached["bars"]]
        if missing:
            raw = self._need_source("pre-market bars").bars_raw(missing, "5Min", start, end)
            for s in missing:
                cached["bars"][s] = raw.get(s, [])
            _write(path, cached)
        return {s: cached["bars"][s] for s in symbols}

    def news(self, day: dt.date, symbols: Sequence[str], since: dt.datetime, until: dt.datetime) -> list[NewsItem]:
        path = self.root / "news" / f"{day}.json.gz"
        cached = _read(path) if path.exists() else {"symbols": [], "items": []}
        have = set(cached["symbols"])
        missing = [s for s in symbols if s not in have]
        if missing:
            items = self._need_source("news").get_news(missing, since, until)
            known = {i["id"] for i in cached["items"]}
            for n in items:
                if n.id not in known:
                    d = asdict(n)
                    d["published_at"] = n.published_at.isoformat()
                    cached["items"].append(d)
            cached["symbols"] = sorted(have | set(missing))
            _write(path, cached)
        wanted = set(symbols)
        out = []
        for d in cached["items"]:
            if wanted & set(d["symbols"]):
                d = dict(d, symbols=tuple(d["symbols"]), published_at=dt.datetime.fromisoformat(d["published_at"]))
                out.append(NewsItem(**d))
        return out

    def intraday(self, sym: str, start: dt.datetime, end: dt.datetime) -> list[dict]:
        path = self.root / "intraday" / f"{sym}_{start:%Y%m%d%H%M}_{end:%Y%m%d%H%M}.json.gz"
        if path.exists():
            return _read(path)
        bars = self._need_source("intraday bars").bars_raw([sym], "5Min", start, end).get(sym, [])
        _write(path, bars)
        return bars


def eligible_symbols(universe: list[Security], cfg) -> list[str]:
    return [s.symbol for s in universe if not static_rejections(s, cfg)]
