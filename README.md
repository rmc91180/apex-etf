# APEX

This repo holds two tools:

- **APEX ETF scanner** (`apex_scan.py`, `apex_telegram.py`): the existing ETF morning scan and position alerts.
- **APEX Stocks** (`apex_stocks/`): a daily signal for one U.S. common stock under $20, with Telegram alerts before the planned exit. It is being built in steps; see below.

## APEX Stocks

The tool recommends; you place trades yourself. It never claims a stock will rise, never invents data, and returns **NO TRADE TODAY** when nothing passes its quality bar.

### What exists now (step 1: foundation)

| Module | What it does |
| --- | --- |
| `config.py` | Every threshold, overridable by an environment variable of the same name. |
| `market_calendar.py` | NYSE sessions, holidays, early closes and DST in America/New_York; plans the exit time and the alert times. |
| `providers/` | The market-data interface the strategy talks to, plus a deterministic mock provider. |
| `storage.py` | SQLite schema for runs, candidates, recommendations, entries, alerts, outcomes, news and logs. |
| `notify.py` | Telegram and console notifiers; messages built from mock data are always labelled as tests. |

### Exit timing rules

- The exit is `EXIT_MINUTES_BEFORE_CLOSE` (30) minutes before the close of the session `EXIT_SESSION_OFFSET` (1) sessions after entry, so a Wednesday pick exits Thursday at 3:30 PM ET.
- The holding period never exceeds `MAX_HOLDING_HOURS` (48). When a weekend or holiday would push the exit past that, the latest earlier exit point is used. In practice a **Friday pick exits the same Friday**, as does a pick on the day before a market holiday.
- Early closes are respected (for example, a 12:30 PM exit on the day after Thanksgiving).
- Alerts go out 60 and 30 minutes before the exit and at the exit (`ALERT_LEAD_MINUTES`).

### Running the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

### Roadmap

1. Foundation (this step).
2. Signal engine: filters, catalyst-first scoring, NO TRADE TODAY logic, walk-forward backtest with slippage.
3. Scheduled pre-market run and Telegram alerts via GitHub Actions, with outcome tracking.
