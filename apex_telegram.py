import os, json, requests

TG_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TG_CHAT_ID = str(os.environ["TELEGRAM_CHAT_ID"])
POSITIONS_FILE = "positions.json"
OFFSET_FILE    = "telegram_offset.json"

def tg_send(msg):
    requests.post(
        f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
        json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "HTML"},
        timeout=10
    )

def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_updates(offset):
    params = {"timeout": 0}
    if offset is not None:
        params["offset"] = offset
    r = requests.get(f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates", params=params, timeout=15)
    return r.json().get("result", [])

def handle_command(text, positions):
    # Returns (positions, reply_message, changed_bool)
    parts = text.strip().split()
    if not parts:
        return positions, None, False
    cmd = parts[0].lower()

    if cmd == "open":
        # open TICKER ENTRY SHARES [STOPPCT]
        if len(parts) < 4:
            return positions, "⚠️ Usage: open TICKER ENTRY SHARES [STOP%]\nExample: open SOXL 28.40 18 6", False
        ticker = parts[1].upper()
        try:
            entry = float(parts[2])
            shares = int(parts[3])
            stop_pct = float(parts[4]) if len(parts) >= 5 else 6.0
        except ValueError:
            return positions, "⚠️ Entry must be a number, shares an integer, stop a number.", False
        # Remove any existing position with same ticker, then add
        positions = [p for p in positions if p["ticker"] != ticker]
        positions.append({
            "ticker": ticker,
            "entry": entry,
            "shares": shares,
            "stop_pct": stop_pct,
            "high": entry
        })
        stop = round(entry * (1 - stop_pct/100), 2)
        target = round(entry + (entry - stop) * 2.0, 2)
        return positions, f"✅ <b>Opened {ticker}</b>\nEntry: ${entry:.2f} | Shares: {shares}\nStop: ${stop:.2f} ({stop_pct:.0f}%) | Target: ${target:.2f}", True

    elif cmd == "close":
        # close TICKER
        if len(parts) < 2:
            return positions, "⚠️ Usage: close TICKER", False
        ticker = parts[1].upper()
        before = len(positions)
        positions = [p for p in positions if p["ticker"] != ticker]
        if len(positions) == before:
            return positions, f"⚠️ No open position found for {ticker}.", False
        return positions, f"✅ <b>Closed {ticker}</b> — removed from monitoring.", True

    elif cmd == "high":
        # high TICKER PRICE  — updates trailing-stop high
        if len(parts) < 3:
            return positions, "⚠️ Usage: high TICKER PRICE", False
        ticker = parts[1].upper()
        try:
            new_high = float(parts[2])
        except ValueError:
            return positions, "⚠️ Price must be a number.", False
        found = False
        for p in positions:
            if p["ticker"] == ticker:
                p["high"] = max(p.get("high", p["entry"]), new_high)
                found = True
                stop = round(p["high"] * (1 - p["stop_pct"]/100), 2)
                reply = f"✅ <b>{ticker} high updated</b> to ${p['high']:.2f}\nTrailing stop now: ${stop:.2f}"
        if not found:
            return positions, f"⚠️ No open position found for {ticker}.", False
        return positions, reply, True

    elif cmd == "list":
        if not positions:
            return positions, "📋 No open positions.", False
        lines = ["📋 <b>Open Positions</b>"]
        for p in positions:
            stop = round(p["high"] * (1 - p["stop_pct"]/100), 2)
            target = round(p["entry"] + (p["entry"] - stop) * 2.0, 2)
            lines.append(f"• {p['ticker']}: entry ${p['entry']:.2f}, {p['shares']} sh, stop ${stop:.2f}, target ${target:.2f}")
        return positions, "\n".join(lines), False

    elif cmd == "help":
        return positions, ("📖 <b>APEX Commands</b>\n"
            "open TICKER ENTRY SHARES [STOP%]\n"
            "close TICKER\n"
            "high TICKER PRICE\n"
            "list\n"
            "help"), False

    # Unknown command — ignore silently (could be the bot's own messages or noise)
    return positions, None, False

def main():
    offset_data = load_json(OFFSET_FILE, {"offset": None})
    offset = offset_data.get("offset")

    updates = get_updates(offset)
    if not updates:
        print("No new updates.")
        return

    positions = load_json(POSITIONS_FILE, [])
    changed_any = False
    last_update_id = offset

    for u in updates:
        last_update_id = u["update_id"] + 1
        msg = u.get("message") or u.get("channel_post")
        if not msg:
            continue
        # Only accept commands from the configured chat ID
        chat_id = str(msg.get("chat", {}).get("id", ""))
        if chat_id != TG_CHAT_ID:
            print(f"Ignoring message from unauthorized chat {chat_id}")
            continue
        text = msg.get("text", "")
        if not text:
            continue
        positions, reply, changed = handle_command(text, positions)
        if changed:
            changed_any = True
        if reply:
            tg_send(reply)

    # Always advance the offset so we don't reprocess
    save_json(OFFSET_FILE, {"offset": last_update_id})

    if changed_any:
        save_json(POSITIONS_FILE, positions)
        print(f"Positions updated. {len(positions)} open.")
    else:
        print("No position changes.")

if __name__ == "__main__":
    main()
