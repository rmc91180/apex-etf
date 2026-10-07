import pytest

from apex_stocks.config import Config, ConfigError, load_config


def test_defaults_are_valid():
    cfg = load_config({})
    assert cfg.MAX_STOCK_PRICE == 19.99
    assert cfg.MAX_HOLDING_HOURS == 48
    assert cfg.alert_leads == [60, 30, 0]


def test_env_overrides_are_typed():
    cfg = load_config({"MAX_STOCK_PRICE": "15", "EXIT_SESSION_OFFSET": "0", "ALLOW_SPACS": "yes"})
    assert cfg.MAX_STOCK_PRICE == 15.0
    assert cfg.EXIT_SESSION_OFFSET == 0
    assert cfg.ALLOW_SPACS is True


@pytest.mark.parametrize("env", [
    {"MAX_STOCK_PRICE": "20"},
    {"MAX_STOCK_PRICE": "abc"},
    {"MAX_HOLDING_HOURS": "49"},
    {"STOP_LOSS_PERCENT": "0"},
    {"ALLOW_SPACS": "maybe"},
    {"ALERT_LEAD_MINUTES": "60,-5"},
])
def test_invalid_values_are_rejected(env):
    with pytest.raises(ConfigError):
        load_config(env)


def test_alert_leads_are_sorted_and_deduplicated():
    assert Config(ALERT_LEAD_MINUTES="30, 60,0,30").alert_leads == [60, 30, 0]
