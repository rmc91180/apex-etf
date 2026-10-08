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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apex_stocks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="run the pre-market selection and print the result (sends nothing)")
    s.add_argument("--at", help="pretend the scan runs at this ET time, e.g. 2026-10-08T08:00 (must be pre-market)")
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=_scan)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
