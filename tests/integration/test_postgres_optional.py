"""Optional: verify Postgres is reachable when running locally (skipped if down)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg")


@pytest.mark.postgres
def test_postgres_accepts_connection_if_available() -> None:
    """Skip when Docker is not running or credentials differ from .env."""
    import psycopg

    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    db = os.environ.get("POSTGRES_DB", "test_db")
    user = os.environ.get("POSTGRES_USER", "test_user")
    password = os.environ.get("POSTGRES_PASSWORD", "test_pass")

    dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    try:
        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        assert row == (1,)
    except Exception as exc:
        pytest.skip(f"Postgres not reachable (expected in CI without DB): {exc}")
