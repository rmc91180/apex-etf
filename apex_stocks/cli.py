"""Command line entry points.

    python -m apex_stocks.cli scan [--at 2026-10-08T08:00]   # dry run: prints, sends nothing
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from . import messages
from .config import load_config
from .engine import select
from .market_calendar import ET
from .providers import get_provider


def _scan(args) -> int:
    cfg = load_config()
    provider = get_provider(cfg.DATA_PROVIDER, feed=cfg.ALPACA_DATA_FEED)
    now = dt.datetime.fromisoformat(args.at).replace(tzinfo=ET) if args.at else dt.datetime.now(ET)
    if args.at and hasattr(provider, "_now"):
        provider._now = lambda: now
    r = select(provider, cfg, now)
    print(f"status={r.status} provider={r.provider} live={r.live_data} universe={r.universe_size} "
          f"scored={len(r.ranked)} reason={r.reason}")
    if r.issues:
        print("issues:", "; ".join(r.issues))
    print("\nTop candidates:")
    for c in r.ranked[:args.top]:
        print(f"  {c.symbol:6} signal {c.score.signal:5.1f}  conf {c.score.confidence:5.1f}  "
              f"{json.dumps(c.score.components)}  catalyst={c.features.catalyst_category}")
    rejected = [c for c in r.candidates if c.rejections]
    print(f"\nRejected after scoring stage: {len(rejected)}")
    for c in rejected[:args.top]:
        print(f"  {c.symbol:6} {'; '.join(c.rejections)}")
    print("\n--- message preview ---")
    print(messages.morning_signal(r, cfg.USER_TIMEZONE) if r.pick else messages.no_trade(r))
    return 0 if r.status != "failed" else 1


def _backtest(args) -> int:
    import dataclasses as _dc
    from pathlib import Path

    from . import backtest
    from .history import HistoryCache
    from .providers.alpaca import AlpacaProvider

    cfg = load_config()
    source = None if args.cache_only else AlpacaProvider()
    cache = HistoryCache(args.cache, source)
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    records = backtest.run(cache, cfg, start, end)
    summary = backtest.summarize(records, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(backtest.report_markdown(summary))
    with open(out / "days.csv", "w") as f:
        f.write("day,status,symbol,signal,confidence,catalyst,regime,entry,exit,exit_reason,gross_pct,mfe_pct,mae_pct,skip_reason\n")
        for r in records:
            t = r.trade
            f.write(",".join(str(x) for x in (
                r.day, r.status, r.symbol or "", r.signal or "", r.confidence or "", r.catalyst or "", r.regime or "",
                t.entry if t else "", t.exit if t else "", t.exit_reason if t else "",
                round(t.net_return(0) * 100, 3) if t else "", t.mfe_pct if t else "", t.mae_pct if t else "",
                (r.skip_reason or "").replace(",", ";"))) + "\n")
    print(backtest.report_markdown(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apex_stocks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="run the pre-market selection and print the result (sends nothing)")
    s.add_argument("--at", help="pretend the scan runs at this ET time, e.g. 2026-10-08T08:00 (must be pre-market)")
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=_scan)
    b = sub.add_parser("backtest", help="walk-forward backtest over a date range")
    b.add_argument("--start", required=True)
    b.add_argument("--end", required=True)
    b.add_argument("--cache", default=".cache/backtest")
    b.add_argument("--out", default="backtest-results")
    b.add_argument("--cache-only", action="store_true", help="use cached data only; make no API calls")
    b.set_defaults(fn=_backtest)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
