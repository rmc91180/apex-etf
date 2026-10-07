import pytest

from apex_stocks.notify import TEST_LABEL, ConsoleNotifier, NotificationError, TelegramNotifier


class FakeResponse:
    def __init__(self, status, ok=True):
        self.status_code, self._ok, self.text = status, ok, "err"

    def json(self):
        return {"ok": self._ok}


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, timeout):
        self.calls.append(json)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def test_console_labels_test_mode():
    n = ConsoleNotifier(test_mode=True)
    n.send("hello")
    assert n.sent == [TEST_LABEL + "hello"]


def test_telegram_retries_then_succeeds():
    import requests
    s = FakeSession([requests.ConnectionError("boom"), FakeResponse(500), FakeResponse(200)])
    TelegramNotifier("t", "c", session=s, backoff=0).send("hi")
    assert len(s.calls) == 3 and s.calls[0]["text"] == "hi"


def test_telegram_bad_credentials_fail_fast():
    s = FakeSession([FakeResponse(401, ok=False), FakeResponse(200)])
    with pytest.raises(NotificationError):
        TelegramNotifier("t", "c", session=s, backoff=0).send("hi")
    assert len(s.calls) == 1


def test_telegram_requires_credentials():
    with pytest.raises(NotificationError):
        TelegramNotifier("", "c")
