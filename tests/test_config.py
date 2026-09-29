import pytest

from ctp import ConfigError, CTPConfig


def test_defaults_are_valid():
    cfg = CTPConfig()
    assert cfg.storage == "auto"
    assert cfg.min_buffer_bytes <= cfg.initial_buffer_bytes <= cfg.max_cache_bytes


@pytest.mark.parametrize(
    "kwargs",
    [
        {"storage": "tape"},
        {"on_error": "explode"},
        {"ahead_seconds": -1},
        {"max_cache_mb": 0},
        {"chunk_size": 0},
        {"timeout": 0},
        {"max_retries": -1},
        {"min_buffer_mb": 0},
    ],
)
def test_invalid_values_rejected(kwargs):
    with pytest.raises(ConfigError):
        CTPConfig(**kwargs)


def test_buffer_bounds_never_exceed_cap():
    cfg = CTPConfig(max_cache_mb=4, min_buffer_mb=8, initial_buffer_mb=32)
    assert cfg.min_buffer_bytes == cfg.max_cache_bytes
    assert cfg.initial_buffer_bytes == cfg.max_cache_bytes


def test_replace_returns_new_config():
    cfg = CTPConfig()
    other = cfg.replace(ahead_seconds=5)
    assert other.ahead_seconds == 5 and cfg.ahead_seconds == 60


def test_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CTP_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CTP_AHEAD_SECONDS", "12.5")
    monkeypatch.setenv("CTP_MAX_CACHE_MB", "256")
    cfg = CTPConfig.from_env(storage="disk")
    assert cfg.cache_dir == str(tmp_path)
    assert cfg.ahead_seconds == 12.5
    assert cfg.max_cache_mb == 256
    assert cfg.storage == "disk"


def test_from_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("CTP_MAX_CACHE_MB", "lots")
    with pytest.raises(ConfigError):
        CTPConfig.from_env()
