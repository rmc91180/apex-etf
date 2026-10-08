from .base import DataUnavailable, MarketDataProvider
from .mock import MockProvider


def get_provider(name: str, feed: str = "iex") -> MarketDataProvider:
    if name == "mock":
        return MockProvider()
    if name == "alpaca":
        from .alpaca import AlpacaProvider
        return AlpacaProvider(feed=feed)
    raise ValueError(f"unknown DATA_PROVIDER {name!r}")


__all__ = ["DataUnavailable", "MarketDataProvider", "MockProvider", "get_provider"]
