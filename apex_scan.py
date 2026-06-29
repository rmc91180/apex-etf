import os, json, requests, datetime, sys, time, re
from zoneinfo import ZoneInfo

AV_KEY        = os.environ["AV_KEY"]
CLAUDE_KEY    = os.environ["CLAUDE_KEY"]
TG_TOKEN      = os.environ["TELEGRAM_TOKEN"]
TG_CHAT_ID    = os.environ["TELEGRAM_CHAT_ID"]
AV_BASE       = "https://www.alphavantage.co/query"
CLAUDE_URL    = "https://api.anthropic.com/v1/messages"

ET = ZoneInfo("America/New_York")

WATCHLIST = [
  "SLV","GLD","USLV","NUGT","GDXJ",
  "SOXL","TQQQ","TECL","FNGU","LABU",
  "USO","GUSH","BOIL","DRIP","UNG","KOLD",
  "UVXY","SPXL","TNA","FAS","TMF",
  "SPXS","SQQQ","TZA","TBT",
  "SPMO","MTUM","YINN","YANG","IBIT"
]


def get_mode():
    mode = os.environ.get("RUN_MODE", "auto")
    if mode != "auto":
        return mode
    # Auto-detect from current ET time
    now = datetime.datetime.now(ET)
    hour, minute = now.hour, now.minute
    # 9:35am = morning scan
    if hour == 9 and 30 <= minute <= 45:
        return "scan"
    # All other times = position alerts
    return "alerts"


def tg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)


def av_quote(ticker):
    r = requests.get(AV_BASE, params={"function":"GLOBAL_QUOTE","symbol":ticker,"apikey":AV_KEY}, timeout=10)
    d = r.json()
    if "Note" in d or "Information" in d:
        raise Exception("AV rate limit hit")
    g = d.get("Global Quote", {})
    if not g.get("05. price"):
        return None
    return {
        "price":   float(g["05. price"]),
        "open":    float(g["02. open"]),
        "high":    float(g["03. high"]),
        "low":     float(g["04. low"]),
        "change":  float(g["09. change"]),
        "chg_pct": float(g["10. change percent"].replace("%","")),
        "volume":  int(g["06. volume"]),
    }


def claude(prompt, tokens=1200):
    r = requests.post(CLAUDE_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": CLAUDE_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": tokens,
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=90
    )
    d = r.json()
    if r.status_code != 200:
        raise Exception(f"Claude HTTP {r.status_code}: {d}")

    # Collect only text blocks — web search produces tool_use/tool_result blocks
    # before the final text block; we want only text blocks
    text_blocks = [b["text"] for b in d.get("content", []) if b.get("type") == "text"]
    if not text_blocks:
        raise Exception(f"No text block in Claude response. Content types: {[b.get('type') for b in d.get('content', [])]}")

    text = " ".join(text_blocks)

    # Strip markdown fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Find the JSON object — handle any preamble before the opening brace
    start = text.find('{')
    end = text.rfind('}')
    if start == -1 or end == -1:
        raise Exception(f"No JSON object found in response: {text[:200]}")

    return json.loads(text[start:end+1])


def run_morning_scan():
    today = datetime.datetime.now(ET).strftime("%A, %B %d, %Y")

    # Fetch 5 proxy quotes (5 AV calls)
    proxies = ["TQQQ","SOXL","SPXL","UVXY","GLD"]
    quotes = {}
    for ticker in proxies:
        try:
            q = av_quote(ticker)
            if q:
                quotes[ticker] = q
        except Exception as e:
            print(f"Quote failed {ticker}: {e}")
        time.sleep(13)  # AV free tier: max 5 calls/minute

    if len(quotes) < 2:
        msg = (
            f"⚠️ <b>APEX Morning Scan — {today}</b>\n"
            f"Scan aborted: only {len(quotes)}/5 AV quotes succeeded (rate limit).\n"
            f"Try again in a few minutes or check your AV key."
        )
        tg(msg)
        print(f"Insufficient quotes ({len(quotes)}). Aborting.")
        return

    live_ctx = ", ".join(
        f"{t}:${q['price']:.2f}({q['chg_pct']:+.2f}%)"
        for t,q in quotes.items()
    ) or "unavailable"

    prompt = f"""You are a professional ETF day trader. Today is {today}.

Scan the full watchlist and identify the single best short-term trade for the next 7-14 days.

Full watchlist: {", ".join(WATCHLIST)}

Live proxy quotes (for regime context): {live_ctx}

These are short-term momentum trades with a 7-14 day holding window. Prioritize setups with clear near-term catalysts, strong recent momentum, and defined technical levels. Do not recommend anything based on long-term fundamentals.

RULES:
- Timeframe is 7-14 days. The setup must be actionable TODAY.
- If 3+ of these align (trend direction, momentum, regime fit, not at resistance) — trade is on
- Prefer high-momentum names with recent price expansion
- Leveraged ETFs (2x/3x) are acceptable and preferred when momentum is clear
- In Risk-On regime: prefer Macro Long, Tech, Momentum themes
- In Risk-Off regime: prefer Macro Short, Metals, Volatility, TMF
- In High Volatility Chop: prefer UVXY, inverse ETFs, or flag no-trade
- Confidence 55-74 = smaller size. Confidence 75+ = full size
- ALWAYS produce a primary trade recommendation
- Entry, stop, and target must be exact dollar numbers
- Use web search for current market conditions and news
- Today is {today} — do not reference any other date

Respond ONLY in valid JSON:
{{"primary":{{"ticker":"","direction":"LONG","entry":0.00,"stop":0.00,"target":0.00,"stopPct":0.0,"rrRatio":"1:2.0","confidence":0,"rationale":"2-3 specific sentences","catalyst":"1 sentence on the near-term trigger"}},"secondary":{{"ticker":"","direction":"LONG","entry":0.00,"stop":0.00,"target":0.00,"confidence":0,"rationale":"1 sentence"}},"avoid":[{{"ticker":"","reason":"brief"}}],"regime":"Risk-On Rally|Risk-Off Safety|High Volatility Chop|Sector Rotation","regimeDirective":"1 sentence what to do today","generatedAt":"{today}"}}"""

    result = claude(prompt, tokens=1400)
    p = result.get("primary", {})
    s = result.get("secondary", {})
    avoid = result.get("avoid", [])
    avoid_str = ", ".join(f"{a['ticker']} ({a['reason']})" for a in avoid[:3]) if avoid else "none"

    msg = (
        f"📊 <b>APEX Morning Scan — {today}</b>\n"
        f"Regime: {result.get('regime','?')}\n"
        f"{result.get('regimeDirective','')}\n\n"
        f"<b>PRIMARY: {p.get('ticker','')} {p.get('direction','')}</b>\n"
        f"Entry: ${p.get('entry',0):.2f} | Stop: ${p.get('stop',0):.2f} | Target: ${p.get('target',0):.2f}\n"
        f"R/R: {p.get('rrRatio','1:2')} | Conf: {p.get('confidence',0)}%\n"
        f"Catalyst: {p.get('catalyst','')}\n"
        f"{p.get('rationale','')}\n\n"
        f"<b>SECONDARY: {s.get('ticker','')} {s.get('direction','')}</b>\n"
        f"Entry: ${s.get('entry',0):.2f} | Stop: ${s.get('stop',0):.2f} | Target: ${s.get('target',0):.2f}\n"
        f"{s.get('rationale','')}\n\n"
        f"⛔ Avoid: {avoid_str}"
    )
    tg(msg)
    print("Morning scan sent.")


def run_position_alerts():
    if not os.path.exists("positions.json"):
        print("No positions file. Skipping alerts.")
        return

    with open("positions.json") as f:
        positions = json.load(f)

    if not positions:
        print("No open positions. Skipping alerts.")
        return

    today = datetime.datetime.now(ET).strftime("%A, %B %d, %Y")
    alerts_sent = 0

    for p in positions:
        ticker  = p["ticker"]
        entry   = float(p["entry"])
        shares  = int(p["shares"])
        stop_pct = float(p.get("stop_pct", 6.0))
        high    = float(p.get("high", entry))

        stop   = round(high * (1 - stop_pct / 100), 2)
        target = round(entry + (entry - stop) * 2.0, 2)

        try:
            q = av_quote(ticker)
            time.sleep(13)  # AV free tier: max 5 calls/minute
            if not q:
                continue
            price = q["price"]
            dist_pct = ((price - stop) / stop * 100)
            pnl = round((price - entry) * shares, 2)
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"

            alert = None

            if price <= stop:
                alert = (
                    f"🚨 <b>{ticker} STOP HIT</b>\n"
                    f"Price: ${price:.2f} | Stop: ${stop:.2f}\n"
                    f"P&L: {pnl_str}\n"
                    f"Close your position immediately."
                )
            elif 0 <= dist_pct < 2.0:
                alert = (
                    f"⚠️ <b>{ticker} NEAR STOP</b>\n"
                    f"Price: ${price:.2f} | Stop: ${stop:.2f} | Gap: {dist_pct:.1f}%\n"
                    f"Consider tightening or exiting."
                )
            elif price >= target:
                alert = (
                    f"🎯 <b>{ticker} TARGET HIT</b>\n"
                    f"Price: ${price:.2f} | Target: ${target:.2f}\n"
                    f"P&L: {pnl_str}\n"
                    f"Consider taking profit."
                )
            elif q["chg_pct"] <= -3.0:
                alert = (
                    f"📉 <b>{ticker} SHARP DROP</b>\n"
                    f"Price: ${price:.2f} ({q['chg_pct']:+.2f}% today)\n"
                    f"Stop: ${stop:.2f} | P&L: {pnl_str}\n"
                    f"Monitor closely."
                )

            if alert:
                tg(alert)
                alerts_sent += 1
                print(f"Alert sent: {ticker}")
            else:
                print(f"{ticker}: ${price:.2f} — OK (stop ${stop:.2f}, target ${target:.2f})")

        except Exception as e:
            print(f"Alert check failed {ticker}: {e}")

    if alerts_sent == 0:
        print("All positions healthy. No alerts sent.")


if __name__ == "__main__":
    mode = get_mode()
    print(f"Running in mode: {mode}")

    if mode == "scan":
        run_morning_scan()
    elif mode == "alerts":
        run_position_alerts()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
