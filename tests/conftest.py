"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def _postgres_env_for_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal env so ``get_settings()`` works when CLI or code loads settings."""
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
