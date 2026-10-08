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

### Step 2: the signal engine

`engine.select()` runs the pre-market selection and never writes or sends anything, so the same code runs live and in backtests:

1. **Eligibility:** common stocks on NYSE, NASDAQ or NYSE American that are tradable and not halted. ETFs, ETNs, funds, preferreds, warrants, rights, units and (unless `ALLOW_SPACS`) SPACs are excluded.
2. **Price and liquidity:** pre-market price between `MIN_STOCK_PRICE` and `MAX_STOCK_PRICE`, at least `MIN_AVG_DOLLAR_VOLUME` and `MIN_AVG_SHARE_VOLUME` over 20 days, and enough history.
3. **Safety rejections:** wide spread, stale quote, split-like jumps in history, abnormal gaps, a big run-up with no new catalyst, and material negative news (offerings, bankruptcy, FDA rejections, guidance cuts and so on).
4. **Signal score (0–100):** momentum 25, volume 20, catalyst 20, technical 20, risk/liquidity 15 (weights configurable). Without a news catalyst the score is capped at `NO_CATALYST_SCORE_CAP`.
5. **Confidence score (0–100):** quote freshness, spread, pre-market liquidity, history and known fundamentals. A high signal cannot make up for low confidence.
6. **Pick or NO TRADE TODAY:** the top candidate must clear both `MIN_SIGNAL_SCORE` and `MIN_CONFIDENCE_SCORE`.

Catalysts come from keyword rules on Alpaca/Benzinga headlines (`catalysts.py`), so every classification is auditable.

Dry-run the scan (prints the result, sends nothing):

```bash
DATA_PROVIDER=mock python -m apex_stocks.cli scan --at 2026-10-08T08:00
```

With Alpaca, set `DATA_PROVIDER=alpaca`, `ALPACA_KEY` and `ALPACA_SECRET`, or run the **APEX Stocks Dry-Run Scan** workflow from the Actions tab.

### Backtest

`backtest.py` replays history day by day through the same `engine.select()`, using only what was knowable at the scan time (`SCAN_TIME_ET`, 8:45 by default): daily bars before the trade date, pre-market bars delayed 16 minutes as on the free plan, and news published before the scan. Delisted symbols are included where Alpaca lists them.

- **Trade simulation:** buy at the open (skipped if the open is above the max entry or at or below the stop), then exit at the target, the stop or the planned exit time on 5-minute bars. If the target and the stop fall in the same bar, the stop is assumed.
- **Slippage sensitivity:** 0, 0.1, 0.25, 0.5 and 1.0% per side.
- **Walk-forward:** score thresholds are chosen on 6 months of history and applied to the next 2 unseen months, rolling forward. A fold that finds nothing better than zero stays out of the market. The headline result is out-of-sample.
- **Comparisons:** every top pick with no thresholds, a baseline of all scored candidates, results by signal-score bucket, by market regime (SPY vs its 50-day average) and by month.

Run it from the **APEX Stocks Backtest** workflow (Actions tab), or locally with Alpaca keys:

```bash
python -m apex_stocks.cli backtest --start 2025-04-01 --end 2026-09-30
```

Data is cached in `.cache/backtest`, so reruns over the same period make no API calls.

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

1. Foundation (done).
2. Signal engine and walk-forward backtest (done).
3. Scheduled pre-market run and Telegram alerts via GitHub Actions, with outcome tracking.
