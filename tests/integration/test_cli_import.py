"""Tests for cli import behavior."""

def test_cli_module_imports() -> None:
    from src.cli.main import cli  # noqa: F401
