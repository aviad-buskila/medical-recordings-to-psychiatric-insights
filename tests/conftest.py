"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.config.settings import get_settings


@pytest.fixture(autouse=True)
def _test_env_for_settings(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Minimal env so ``get_settings()`` works when CLI or code loads settings."""
    artifacts_dir = tmp_path_factory.mktemp("test_artifacts")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_DB", "test_db")
    monkeypatch.setenv("POSTGRES_USER", "test_user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test_pass")
    # Keep test runs from writing noise under repository data/processed.
    monkeypatch.setenv("EVAL_REPORTS_DIR", str(artifacts_dir / "eval_reports"))
    monkeypatch.setenv("INSIGHTS_EXTRACT_DIR", str(artifacts_dir / "insights_extract"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
