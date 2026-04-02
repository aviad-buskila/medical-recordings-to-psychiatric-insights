"""CLI smoke tests and validation (Click CliRunner)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli.main import cli


def test_cli_help_lists_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in (
        "validate-dataset",
        "run-stt",
        "run-bertscore",
        "run-eval",
        "run-all",
        "show-alignment",
        "run-llm-judge",
    ):
        assert name in result.output


def test_run_eval_rejects_ref_without_run_id() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--ref-run-id", "abc"])
    assert result.exit_code != 0
    assert "run-id" in result.output.lower()


def test_run_bertscore_rejects_ref_without_run_id() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-bertscore", "--ref-run-id", "abc"])
    assert result.exit_code != 0


def test_run_eval_rejects_non_positive_limit() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--limit", "0"])
    assert result.exit_code != 0


def test_run_eval_rejects_non_positive_workers() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--workers", "0"])
    assert result.exit_code != 0


def test_show_alignment_rejects_low_chunk_columns() -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["show-alignment", "--run-id", "x", "--chunk-columns", "2"],
    )
    assert result.exit_code != 0


@patch("src.cli.main.run_alignment_report")
@patch("src.cli.main.make_eval_report_path")
def test_show_alignment_writes_report_artifact(
    mock_make_path: MagicMock,
    mock_report: MagicMock,
    tmp_path: Path,
) -> None:
    from src.cli import main as cli_main

    mock_make_path.return_value = tmp_path / "show-alignment_fixed.txt"
    mock_report.return_value = "ALIGNMENT_TEXT"
    # Make command-line deterministic for the written file.
    cli_main.sys.argv = ["python", "-m", "src.cli.main", "show-alignment", "--run-id", "rid"]

    runner = CliRunner()
    result = runner.invoke(cli, ["show-alignment", "--run-id", "rid", "--chunk-columns", "48"])
    assert result.exit_code == 0

    report_file = tmp_path / "show-alignment_fixed.txt"
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Command line:" in content
    assert "show-alignment output" in content
    assert "ALIGNMENT_TEXT" in content


@patch("src.cli.main.run_bertscore_eval")
def test_run_bertscore_cli_invokes_eval(mock_eval: MagicMock) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run-bertscore", "--run-id", "rid", "--limit", "5", "--no-rescale"],
    )
    assert result.exit_code == 0
    assert "BERTScore evaluation completed" in result.output
    mock_eval.assert_called_once()
    kwargs = mock_eval.call_args.kwargs
    assert kwargs["run_id"] == "rid"
    assert kwargs["limit"] == 5
    assert kwargs["rescale_with_baseline"] is False


@patch("src.cli.main.evaluate_stt_against_gold")
def test_run_eval_cli_invokes_stt_eval(mock_eval: MagicMock) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run-eval", "--limit", "3"])
    assert result.exit_code == 0
    assert "Evaluation completed" in result.output
    kwargs = mock_eval.call_args.kwargs
    assert kwargs["limit"] == 3
    assert kwargs["run_id"] is None
    assert kwargs["ref_run_id"] is None
    assert kwargs["workers"] == 1
    assert "reporter" in kwargs
