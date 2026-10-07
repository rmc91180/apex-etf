from .base import DataUnavailable, MarketDataProvider
from .mock import MockProvider


def get_provider(name: str) -> MarketDataProvider:
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown DATA_PROVIDER {name!r}")


__all__ = ["DataUnavailable", "MarketDataProvider", "MockProvider", "get_provider"]
