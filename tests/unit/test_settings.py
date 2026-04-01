import os

from src.config.settings import Settings


def test_settings_dsn_builds() -> None:
    os.environ["POSTGRES_HOST"] = "localhost"
    os.environ["POSTGRES_DB"] = "db"
    os.environ["POSTGRES_USER"] = "user"
    os.environ["POSTGRES_PASSWORD"] = "pass"
    settings = Settings()
    assert "postgresql://user:pass@localhost" in settings.postgres_dsn
