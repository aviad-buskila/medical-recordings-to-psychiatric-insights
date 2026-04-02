import pytest

from src.config.settings import Settings


def test_settings_dsn_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_DB", "db")
    monkeypatch.setenv("POSTGRES_USER", "user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "pass")
    settings = Settings()
    assert "postgresql://user:pass@localhost" in settings.postgres_dsn
