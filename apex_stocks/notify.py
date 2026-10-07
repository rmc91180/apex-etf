"""Notification delivery. Telegram for real alerts, console for dry runs.

When the data behind a message is synthetic, every message is labelled as a
test so it can never be mistaken for a real signal.
"""
from __future__ import annotations

import os
import time
from typing import Protocol

import requests

TEST_LABEL = "🧪 TEST: synthetic data, not a trading signal\n\n"


class NotificationError(RuntimeError):
    pass


class Notifier(Protocol):
    def send(self, text: str) -> None: ...


class ConsoleNotifier:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.sent: list[str] = []

    def send(self, text: str) -> None:
        text = (TEST_LABEL if self.test_mode else "") + text
        self.sent.append(text)
        print(text)


class TelegramNotifier:
    API = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str, test_mode: bool = False,
                 retries: int = 3, backoff: float = 2.0, session: requests.Session | None = None):
        if not token or not chat_id:
            raise NotificationError("TELEGRAM_TOKEN and TELEGRAM_CHAT_ID are required")
        self.token, self.chat_id = token, chat_id
        self.test_mode = test_mode
        self.retries, self.backoff = retries, backoff
        self.http = session or requests.Session()

    def send(self, text: str) -> None:
        text = (TEST_LABEL if self.test_mode else "") + text
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = self.http.post(self.API.format(token=self.token),
                                   json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                                         "disable_web_page_preview": True},
                                   timeout=10)
                if r.status_code == 200 and r.json().get("ok"):
                    return
                last_err = NotificationError(f"Telegram HTTP {r.status_code}: {r.text[:200]}")
                if 400 <= r.status_code < 500 and r.status_code != 429:
                    break  # bad token or chat id: retrying will not help
            except requests.RequestException as e:
                last_err = e
            if attempt < self.retries - 1:
                time.sleep(self.backoff * (2 ** attempt))
        raise NotificationError(f"Telegram send failed: {last_err}")


def get_notifier(name: str, test_mode: bool) -> Notifier:
    if name == "console":
        return ConsoleNotifier(test_mode=test_mode)
    if name == "telegram":
        return TelegramNotifier(os.environ.get("TELEGRAM_TOKEN", ""), os.environ.get("TELEGRAM_CHAT_ID", ""),
                                test_mode=test_mode)
    raise ValueError(f"unknown NOTIFIER {name!r}")
