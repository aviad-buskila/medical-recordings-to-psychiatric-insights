"""Helpers for integration tests (subprocess CLI, shared paths)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Project root (contains src/)
ROOT = Path(__file__).resolve().parents[2]


def base_cli_env() -> dict[str, str]:
    """Env vars required for Settings when spawning the CLI subprocess."""
    return {
        "POSTGRES_HOST": os.environ.get("POSTGRES_HOST", "localhost"),
        "POSTGRES_PORT": os.environ.get("POSTGRES_PORT", "5432"),
        "POSTGRES_DB": os.environ.get("POSTGRES_DB", "test_db"),
        "POSTGRES_USER": os.environ.get("POSTGRES_USER", "test_user"),
        "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD", "test_pass"),
    }


def run_main_cli(
    *args: str,
    extra_env: dict[str, str] | None = None,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    """Run ``python -m src.cli.main`` the same way a user would (cwd = repo root)."""
    env = {**os.environ, **base_cli_env(), **(extra_env or {})}
    return subprocess.run(
        [sys.executable, "-m", "src.cli.main", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )
