"""Run the real CLI entrypoint in a subprocess (integration)."""

from __future__ import annotations

from tests.integration.conftest import run_main_cli


def test_subprocess_help_exits_zero() -> None:
    proc = run_main_cli("--help")
    assert proc.returncode == 0
    out = proc.stdout + proc.stderr
    assert "show-alignment" in out or "Commands:" in out or "Usage:" in out


def test_subprocess_validate_dataset_help() -> None:
    proc = run_main_cli("validate-dataset", "--help")
    assert proc.returncode == 0
    assert "validate-dataset" in proc.stdout.lower() or "dataset" in proc.stdout.lower()


def test_subcommand_help_run_eval() -> None:
    proc = run_main_cli("run-eval", "--help")
    assert proc.returncode == 0
    assert "--run-id" in proc.stdout


def test_subcommand_help_run_bertscore() -> None:
    proc = run_main_cli("run-bertscore", "--help")
    assert proc.returncode == 0
    assert "--run-id" in proc.stdout or "run-id" in proc.stdout


def test_subcommand_help_show_alignment() -> None:
    proc = run_main_cli("show-alignment", "--help")
    assert proc.returncode == 0
    assert "--chunk-columns" in proc.stdout
